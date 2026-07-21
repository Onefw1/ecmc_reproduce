import torch
import torch.nn as nn
import lightning.pytorch as pl
from module.Qformer import BertConfig, BertLMHeadModel
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
    BertTokenizer, 
    BertModel,
    LlamaTokenizer
)
from module.modeling_llama import LlamaForCausalLM
import torch.nn.functional as F
import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from MMDAmodel import *
import json
from torch.cuda.amp import autocast
class ECMCLLaMA(ECMC):
    def __init__(
        self,
        llama_ckpt="weights"):
        super().__init__()
        #llama
        self.llama_model=LlamaForCausalLM.from_pretrained(llama_ckpt, torch_dtype="auto")
        self.llama_tokenizer=LlamaTokenizer.from_pretrained(llama_ckpt)
        if self.llama_tokenizer.pad_token_id is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.unk_token
        for param in self.llama_model.parameters():
            param.requires_grad = False
        self.llama_model.eval() 

        self.emo_llama_project=nn.Linear(768,4096)
        self.cog_llama_project=nn.Linear(768,4096)

    def forward(self, audio, video, text, emo_category, cognition_category, emotion_cap, cognition_cap):
        # 调用 Stage 1 的 BridgeNet 前向过程，得到情绪和认知两路融合特征；第一个返回值是 Stage 1 对比损失，Stage 2 不直接使用。
        _, combined_feature_emo, combined_feature_cog = super().forward(audio, video, text, emo_category, cognition_category)
        #两个投影层，(B,96,768) →  (B,96,4096)
        # 将情绪融合特征从 768 维投影到 LLaMA 的 4096 维 embedding 空间。
        emo_input = self.emo_llama_project(combined_feature_emo)
        # 将认知融合特征从 768 维投影到 LLaMA 的 4096 维 embedding 空间。
        cog_input = self.cog_llama_project(combined_feature_cog)
        #拼接，得到(B,192,4096)
        # 在序列维度拼接情绪 token 和认知 token，得到多模态 soft tokens。
        mm_input = torch.cat([emo_input, cog_input], dim=1)

        # 记录 batch size，后面用于把同一个 prompt 复制到每个样本。
        batchsize = emo_input.shape[0]

        # 构造固定任务提示词，告诉 LLaMA 需要输出情绪和认知分析。
        prompt = '''I provide you with a conversation between a doctor and a user. Please analyze the emotional state and the signs of cognitive impairment. 

        For emotion, What are the facial expressions used by the person in the video? What is the intended meaning behind his words? Which emotion does this reflect?

        For cognition impairment, it includes four domains:1. Orientation 2. Memory 3. Attention 4. Language ability.
        Please provide a brief analysis about cognitive impairment (1–3 sentences) that considers both the user's speech and the video emotion

        Output Format:
        Emotion:
        <your description here>

        Cognition:
        1. ...
        2. ...
        3. ...
        '''

        # 使用 LLaMA tokenizer 将 prompt 文本转换成 token id。
        prompts_id = self.llama_tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        ).input_ids.to(self.device)
        # 将单条 prompt token id 复制到 batch 内每个样本。
        prompts_id = prompts_id.expand(batchsize, -1)
        # 通过 LLaMA 的词嵌入层把 prompt token id 转成 embedding。
        prompts_embeds = self.llama_model.model.embed_tokens(prompts_id)

        # Stage 2 训练必须提供 emotion caption，否则无法构造生成监督信号。
        assert emotion_cap is not None
        # 将每个样本的情绪 caption 和认知 caption 拼成目标输出文本。
        text_cap = [f"Emotion:\n{e}\n\nCognition:\n{c}" for e, c in zip(emotion_cap, cognition_cap)]
        # 将目标 caption 文本转换成 token id 和 attention mask。
        text_tokens = self.llama_tokenizer(
            text_cap,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        # 将目标 caption token id 转成 LLaMA embedding，用于 teacher forcing 输入。
        text_embeds = self.llama_model.model.embed_tokens(text_tokens.input_ids)
        # 构造语言模型标签；padding token 替换为 -100，交叉熵损失会忽略这些位置。
        targets = text_tokens.input_ids.masked_fill(text_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100)

        # 构造每个样本的 BOS token id，作为 LLaMA 输入序列起始 token。
        bos = torch.ones([batchsize, 1], dtype=torch.long).to(self.device) * self.llama_tokenizer.bos_token_id
        # 将 BOS token id 转成 embedding。
        bos_embeds = self.llama_model.model.embed_tokens(bos)

        # 拼接完整输入 embedding：BOS + 多模态 soft tokens + prompt embeddings + 目标 caption embeddings。
        input_embeds = torch.cat([bos_embeds, mm_input, prompts_embeds, text_embeds], dim=1)

        # 多模态 soft tokens 都是有效 token，因此构造全 1 attention mask。(B, 192)
        attn_audio = torch.ones(mm_input.shape[:-1], dtype=torch.long).to(self.device)
        # prompt tokens 也全部作为有效输入参与注意力计算。(B, prompt_len)
        attn_prompt = torch.ones(prompts_embeds.shape[:-1], dtype=torch.long).to(self.device)
        # caption 的 attention mask 来自 tokenizer，padding 位置为 0，真实 token 位置为 1。 (B, caption_len)
        attn_text = text_tokens.attention_mask  
        # BOS 只有 1 个 token，复用多模态 mask 的第一列形状构造 BOS mask。(B, 1)
        attn_bos = attn_audio[:, :1]
        # 拼接完整 attention mask，长度必须和 input_embeds 的序列长度一致。
        #告诉LLaMA哪些 token 是有效输入，需要参与注意力计算；哪些 token是padding，需要忽略。
        #attention_mask: (B, 1 + 192 + prompt_len + caption_len)
        attention_mask = torch.cat([attn_bos, attn_audio, attn_prompt, attn_text], dim=1)

        # 将拼好的 embedding 直接输入 LLaMA；这里用 inputs_embeds 接入多模态 soft tokens。
        outputs = self.llama_model(
            inputs_embeds=input_embeds,
            use_cache=False,#关闭kv缓存，避免训练时梯度计算出错。
            attention_mask=attention_mask,
            labels=targets,#因为传了 labels=targets，所以它不仅会输出 logits，还会自动计算语言模型交叉熵损失：预测 token 和真实 caption token 的交叉熵 loss
            return_dict=True
        )
        # 返回 LLaMA 的语言模型损失，Lightning 会用这个标量 loss 自动执行反向传播。
        return outputs.loss

    
    def training_step(self, batch, batch_idx):
        audio, video, text, emo_category, cognition_category, emotion_cap, cognition_cap =batch['audio'],batch['video'],batch['text'],batch['emo_category'],batch['cognition_category'],batch['emotion_cap'],batch['cognition_cap']
        loss=self.forward(audio, video, text, emo_category, cognition_category,emotion_cap, cognition_cap)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=len(audio),sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        audio, video, text, emo_category, cognition_category, emotion_cap, cognition_cap =batch['audio'],batch['video'],batch['text'],batch['emo_category'],batch['cognition_category'],batch['emotion_cap'],batch['cognition_cap']
        loss=self.forward(audio, video, text, emo_category, cognition_category,emotion_cap, cognition_cap)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=len(audio),sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
        filter(lambda p: p.requires_grad, self.parameters()),
        lr=1e-4
    )

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass

    def inference(self, audio, video, text, emo_category, cognition_category):
        self.eval()
        _, combined_feature_emo, combined_feature_cog = super().forward(audio, video, text, emo_category, cognition_category)
        emo_input = self.emo_llama_project(combined_feature_emo)
        cog_input = self.cog_llama_project(combined_feature_cog)
        mm_input = torch.cat([emo_input, cog_input], dim=1)

        batchsize = emo_input.shape[0]

        prompt = '''I provide you with a conversation between a doctor and a user. Please analyze the emotional state and the signs of cognitive impairment. 

        For emotion, What are the facial expressions used by the person in the video? What is the intended meaning behind his words? Which emotion does this reflect?

        For cognition impairment, it includes four domains:1. Orientation 2. Memory 3. Attention 4. Language ability.
        Please provide a brief analysis about cognitive impairment (1–3 sentences) that considers both the user's speech and the video emotion

        Output Format:
        Emotion:
        <your description here>

        Cognition:
        1. ...
        2. ...
        3. ...
        '''
        prompt_ids = self.llama_tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        ).input_ids.to(self.device)
        prompt_ids = prompt_ids.expand(batchsize, -1) 
        prompt_embeds = self.llama_model.model.embed_tokens(prompt_ids)

        bos = torch.ones([batchsize, 1], dtype=torch.long).to(self.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)

        input_embeds = torch.cat([bos_embeds, mm_input, prompt_embeds], dim=1)

        attn_bos = torch.ones((batchsize, 1), dtype=torch.long).to(self.device)
        attn_mm = torch.ones(mm_input.shape[:-1], dtype=torch.long).to(self.device)
        attn_prompt = torch.ones(prompt_embeds.shape[:-1], dtype=torch.long).to(self.device)
        attention_mask = torch.cat([attn_bos, attn_mm, attn_prompt], dim=1)
     
        with torch.no_grad():
            with autocast(dtype=torch.float16):
                outputs = self.llama_model.generate(
                        inputs_embeds=input_embeds,
                        attention_mask=attention_mask,
                        max_new_tokens=256,
                        min_new_tokens=None,
                        do_sample=True,
                        top_k=10,
                        top_p=0.95,
                        num_beams=3,
                        repetition_penalty=9.0,
                        pad_token_id=self.llama_tokenizer.pad_token_id,
                        eos_token_id=self.llama_tokenizer.eos_token_id,
                        early_stopping=True,
                        num_return_sequences=1,
                        no_repeat_ngram_size=2
                    )
        decoded = self.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        final_outputs=decoded
        return final_outputs

    def test_step(self, batch, batch_idx, save_path="predictions.jsonl"):
        audio = batch["audio"]
        video = batch["video"]
        text = batch["text"]
        emo_cat = batch["emo_category"]
        cog_cat = batch["cognition_category"]
        sample_id = batch["ids"]  

        predictions = self.inference(audio, video, text, emo_cat, cog_cat)  

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'a', encoding='utf-8') as f:
            for sid, pred in zip(sample_id, predictions):
                result = {
                    "id": sid,
                    "label": text,
                    "prediction": pred
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
