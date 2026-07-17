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
from transformers import StoppingCriteria, StoppingCriteriaList
import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False

class ECMC(pl.LightningModule):
    def __init__(
        self,
        text2vec_ckpt="weights",
        audio_input_dim=512,
        video_input_dim=2048,
        hidden_size=768,
        cognition_loss_weight=1.0,
        train_qformers=True):
        super(ECMC,self).__init__()
        self.training_step_outputs  = []
        
        #path
        current_directory = os.path.dirname(os.path.abspath(__file__))
        text2vec_ckpt = os.path.join(current_directory, text2vec_ckpt)

        #text2vec
        self.text2vec_model=BertModel.from_pretrained(text2vec_ckpt)
        self.text2vec_tokenizer=BertTokenizer.from_pretrained(text2vec_ckpt)

        for p in self.parameters():
            p.requires_grad = False

        # EgoCom feature widths differ from the 768-D Q-Former input.
        self.audio_adapter = nn.Linear(audio_input_dim, hidden_size)
        self.video_adapter = nn.Linear(video_input_dim, hidden_size)
        self.cognition_loss_weight = cognition_loss_weight
        self.train_qformers = train_qformers

        #Qformer-audio-emo
        self.audio_Qformer,self.audio_query_tokens=self.init_Qformer(num_query_token=32,vision_width=768)
        self.audio_Qformer.cls = None
        self.audio_Qformer.bert.embeddings.word_embeddings = None
        self.audio_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.audio_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.audio_llama_project=nn.Linear(768,4096)

        #Qformer-video-emo
        self.video_Qformer,self.video_query_tokens=self.init_Qformer(num_query_token=32,vision_width=768)
        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        #Qformer-text-emo
        self.text_Qformer,self.text_query_tokens=self.init_Qformer(num_query_token=32,vision_width=768)
        self.text_Qformer.cls = None
        self.text_Qformer.bert.embeddings.word_embeddings = None
        self.text_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.text_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        #Qformer-audio-cong
        self.audio_Qformer_cong,self.audio_query_tokens_cong=self.init_Qformer(num_query_token=32,vision_width=768)
        self.audio_Qformer_cong.cls = None
        self.audio_Qformer_cong.bert.embeddings.word_embeddings = None
        self.audio_Qformer_cong.bert.embeddings.position_embeddings = None
        for layer in self.audio_Qformer_cong.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        
        #Qformer-video-cong
        self.video_Qformer_cong,self.video_query_tokens_cong=self.init_Qformer(num_query_token=32,vision_width=768)
        self.video_Qformer_cong.cls = None
        self.video_Qformer_cong.bert.embeddings.word_embeddings = None
        self.video_Qformer_cong.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer_cong.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        
        #Qformer-text-cong
        self.text_Qformer_cong,self.text_query_tokens_cong=self.init_Qformer(num_query_token=32,vision_width=768)
        self.text_Qformer_cong.cls = None
        self.text_Qformer_cong.bert.embeddings.word_embeddings = None
        self.text_Qformer_cong.bert.embeddings.position_embeddings = None
        for layer in self.text_Qformer_cong.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        if not self.train_qformers:
            qformers = (
                self.audio_Qformer,
                self.video_Qformer,
                self.text_Qformer,
                self.audio_Qformer_cong,
                self.video_Qformer_cong,
                self.text_Qformer_cong,
            )
            for qformer in qformers:
                qformer.requires_grad_(False)
            for query_tokens in (
                self.audio_query_tokens,
                self.video_query_tokens,
                self.text_query_tokens,
                self.audio_query_tokens_cong,
                self.video_query_tokens_cong,
                self.text_query_tokens_cong,
            ):
                query_tokens.requires_grad_(False)
            self.audio_llama_project.requires_grad_(False)
        

    def init_Qformer(self,num_query_token, vision_width, cross_attention_freq=2):
        path=os.path.dirname(os.path.abspath(__file__))
        config_path=os.path.join(path,"weights")# 基于 BERT 配置
        encoder_config = BertConfig.from_pretrained(config_path)
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True # 开启交叉注意力
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        ckpt=os.path.join(path,"pytorch_model.bin")
        Qformer.load_state_dict(torch.load(ckpt),strict=False)

        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
    
    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    
    def forward(self, audio, video, text, emo_category, cognition_category):
        #audio  直接用 DataLoader 读进来的 .npy 数据，shape: (B, 1024, 768)
        audio_feature=self.audio_adapter(audio.float())

        #text2vec 编码文字
        with torch.no_grad():  # 冻结，不训练bert
            #describtion
            describtion=[s+"</s>" for s in text]# 每句末尾加结束符
            describtion_input=self.text2vec_tokenizer(describtion, padding=True, truncation=True, return_tensors='pt').to(self.device)# 文字 → 数字编码
            describtion_feature=self.text2vec_model(**describtion_input)# 数字 → 768维向量
            describtion_feature=self.mean_pooling(describtion_feature,describtion_input['attention_mask']).unsqueeze(1)# 所有词取平均
            describtion_feature = describtion_feature.to(audio_feature.dtype) # (B, 768) → (B, 1, 768)

        #Qformer-emotion-----------------------------------------------------
        #audio → 情感音频 Q-Former
        audio_query_tokens=self.audio_query_tokens.expand(audio_feature.shape[0], -1, -1)
        frame_atts_audio = torch.ones(audio_feature.size()[:-1], dtype=torch.long).to(audio_feature.device)#准备注意力掩码

        #print(audio_query_tokens.shape,audio_feature.shape,frame_atts.shape)
        audio_query_output=self.audio_Qformer.bert(
            query_embeds=audio_query_tokens, #32个专属提问者
            encoder_hidden_states=audio_feature,#[B,1024,768]
            encoder_attention_mask=frame_atts_audio,
            return_dict=True,
            )
        audio_hidden=audio_query_output.last_hidden_state#[B,32,768]

        # 直接用 DataLoader 读进来的 .npy 数据，shape: (B, 512, 768)
        video_feature=self.video_adapter(video.float())
        # video → 情感视频 Q-Former
        video_query_tokens=self.video_query_tokens.expand(video_feature.shape[0], -1, -1)
        frame_atts_video = torch.ones(video_feature.size()[:-1], dtype=torch.long).to(video_feature.device)

        video_query_output=self.video_Qformer.bert(
            query_embeds=video_query_tokens, #[32,768]
            encoder_hidden_states=video_feature,# (B, 512, 768)
            encoder_attention_mask=frame_atts_video,
            return_dict=True,
            )
        video_hidden=video_query_output.last_hidden_state# (B, 32, 768)

        # text → 情感文本 Q-Former
        text_feature=describtion_feature
        text_query_tokens=self.text_query_tokens.expand(text_feature.shape[0], -1, -1)
        frame_atts_text = torch.ones(text_feature.size()[:-1], dtype=torch.long).to(text_feature.device)

        text_query_output=self.text_Qformer.bert(
            query_embeds=text_query_tokens, #[32,768]
            encoder_hidden_states=text_feature,# (B, 1, 768)
            encoder_attention_mask=frame_atts_text,
            return_dict=True,
            )
        text_hidden=text_query_output.last_hidden_state # (B, 32, 768)
        #拼接
        combined_feature_emo = torch.cat([audio_hidden, video_hidden, text_hidden], dim=1)# (B, 96, 768)

        #Qformer-congnition---------------------------------------------------
        audio_query_tokens_cong=self.audio_query_tokens_cong.expand(audio_feature.shape[0], -1, -1)
        audio_query_output=self.audio_Qformer_cong.bert(
            query_embeds=audio_query_tokens_cong, 
            encoder_hidden_states=audio_feature,
            encoder_attention_mask=frame_atts_audio,
            return_dict=True,
            )
        audio_hidden=audio_query_output.last_hidden_state

        video_query_tokens_cong=self.video_query_tokens_cong.expand(video_feature.shape[0], -1, -1)
        video_query_output=self.video_Qformer_cong.bert(
            query_embeds=video_query_tokens_cong,
            encoder_hidden_states=video_feature,
            encoder_attention_mask=frame_atts_video,
            return_dict=True,
            )
        video_hidden=video_query_output.last_hidden_state

        text_query_tokens_cong=self.text_query_tokens_cong.expand(text_feature.shape[0], -1, -1)
        text_query_output=self.text_Qformer_cong.bert(
            query_embeds=text_query_tokens_cong,
            encoder_hidden_states=text_feature,
            encoder_attention_mask=frame_atts_text,
            return_dict=True,
            )
        text_hidden=text_query_output.last_hidden_state
        combined_feature_cong = torch.cat([audio_hidden, video_hidden, text_hidden], dim=1)

        #Loss---------------------------------------------------
        emo_loss = self.contrastive_loss(
        features=combined_feature_emo,
        labels=emo_category, 
        temperature=0.1
        )
    
        cog_loss = combined_feature_cong.new_zeros(()) if self.cognition_loss_weight == 0 else self.multilabel_contrastive_loss(
        features=combined_feature_cong,
        labels=[row.nonzero(as_tuple=True)[0].tolist() for row in cognition_category],  # 转类别索引
        temperature=0.1
        )

        return emo_loss + self.cognition_loss_weight * cog_loss, combined_feature_emo, combined_feature_cong
    
    def contrastive_loss(self, features, labels, temperature=0.1):
        """
        情绪监督式对比损失。

        Args:
            features: 情绪流 BridgeNet 输出，形状通常为 (B, 96, 768)。
                B 是 batch size；96 = audio/video/text 三个模态各 32 个 query token。
            labels: 每个样本的情绪标签，来自 batch["emo_category"]，形状为 (B,)。
            temperature: 温度系数，用来放大或压缩样本相似度差异。

        目标：
            - 同一 batch 中情绪标签相同的样本表示更接近；
            - 情绪标签不同的样本表示更远。

        情绪对比学习里去掉对角线，是为了避免样本和自己比较这种没有训练意义的正样本参与 loss。认知多标签 loss 当前代码没有去掉对角线，因此样本自己会被当成正样本，这可以保证每个样本至少有正样本，计算上更稳定；但从严格对比学习角度看，这会引入 trivial positive，可能削弱模型学习“不同样本之间认知相似性”的信号。因此认知 loss 不是理论上不需要去掉，而是当前实现没有去掉。
        """
        # 将每个样本的 96 个 token 平均成一个整体情绪向量：(B, 96, 768) -> (B, 768)
        features = features.mean(dim=1)
        # L2 归一化，只比较向量方向，避免向量长度影响相似度
        features = F.normalize(features, p=2, dim=1)
        # 计算 batch 内样本两两相似度矩阵：(B, 768) @ (768, B) -> (B, B)，logits[i][j] 表示：第 i 个样本 和 第 j 个样本 的相似度
        logits = torch.matmul(features, features.T) / temperature
        labels = labels.view(-1)  # 将 labels 从 (32, 1) 变为 (32,)
        assert logits.shape[0] == labels.shape[0]

        # 根据情绪标签构造正负样本矩阵：标签相同为 True，标签不同为 False
        #labels.unsqueeze(1) — 添加列维度，将 labels 从形状 (B,) 变为 (B, 1)。
        #labels.unsqueeze(0) — 添加行维度,将 labels 从形状 (B,) 变为 (1, B)。
        label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)

        # 构造一个去掉对角线的 mask，避免“样本和自己比较”参与损失计算,torch.eye(len(labels)) 会生成单位矩阵,对角线 True，其他 False
        #dtype=torch.bool: 生成布尔 mask
        mask = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        #masked_select(mask): 只选择 mask 为 True 的元素
        #view(logits.size(0), -1): 重新整理成 B 行，每行 B-1 个元素
        logits_masked = logits.masked_select(mask).view(logits.size(0), -1)#从 logits 中取出非对角线元素。
        #对标签正负样本矩阵做同样的去对角线处理
        label_matrix_masked = label_matrix.masked_select(mask).view(logits.size(0), -1)

        # 对每个样本，把它与其他样本的相似度转成 log probability，等价于log_prob = log_softmax(logits_masked)，表示每个样本和其他样本相似的 log 概率
        #torch.logsumexp: 稳定计算 log(sum(exp(x)))
        #dim=1: 每一行内部归一化
        #keepdim=True: 保持维度为 (B, 1)，方便广播相减
        log_prob = logits_masked - torch.logsumexp(logits_masked, dim=1, keepdim=True)

        # 把布尔正样本矩阵转成浮点矩阵。原来True / False，现在 1.0 / 0.0，
        # 为什么这么写：后面要用它和 log_prob 相乘。正样本位置乘 1 保留，负样本位置乘 0 去掉。
        positives = label_matrix_masked.float()
        # 计算每个样本对应正样本的平均 log probability。
        #.sum(1)对每个样本的所有正样本 log probability 求和。
        #positives.sum(1) 对每个样本的正样本数量求和，clamp(min=1) 避免除以 0。
        #它衡量的是：对每个样本来说，模型有没有把情绪相同的样本排得更近。如果正样本相似度高，这个值会更大。
        mean_log_prob_pos = (positives * log_prob).sum(1) / positives.sum(1).clamp(min=1)

        # 类内（同一种情绪）损失：正样本 log probability 越大越好，但训练时优化器是最小化 loss，所以取负号。
        intra_loss = -mean_log_prob_pos.mean()

        # 负样本：情绪标签不同的样本对
        negatives = (~label_matrix_masked).float()
        # 类间（不同情绪）损失：如果负样本相似度过高，exp(logits) 会变大，从而增大惩罚
        inter_loss = torch.log1p(torch.exp(logits_masked) * negatives).mean()

        # 总情绪对比损失 = 同类拉近损失 + 异类推远损失
        return intra_loss + inter_loss


    def multilabel_contrastive_loss(self, features, labels, temperature=0.1):
        """
        认知多标签对比损失。

        Args:
            features: 认知流 BridgeNet 输出，形状通常为 (B, 96, 768)。
            labels: 每个样本的认知标签索引列表，例如 [[0, 2], [1], [], ...]。
                它由 cognition_category 的多热向量转换而来，例如 [1,0,1,0] -> [0,2]。
            temperature: 温度系数，用来控制相似度分布的尖锐程度。

        目标：
            - 认知标签集合有重叠的样本表示更接近；
            - 认知标签完全不重叠的样本表示更远。

        两个矩阵：
        1. 样本相似度矩阵：(B, B)：模型当前认为 batch 内样本两两认知表示有多像
        2. 标签相似度矩阵：(B, B)：根据认知标签集合计算的 Jaccard 相似度，表示样本间真实的认知相似度
        """
        # 将每个样本的 96 个 token 平均成一个整体认知向量：(B, 96, 768) -> (B, 768)
        features = features.mean(dim=1)
        # L2 归一化，方便用点积近似余弦相似度
        features = F.normalize(features, p=2, dim=1)
        # 计算 batch 内样本两两认知特征相似度矩阵：(B, B)
        sim_matrix = torch.matmul(features, features.T) / temperature

        # 根据认知多标签构造标签相似度矩阵，初始形状与 sim_matrix 相同：(B, B)
        weight_matrix = torch.zeros_like(sim_matrix)
        #enumerate(labels): 同时拿到索引 i 和对应元素 lbls_i
        for i, lbls_i in enumerate(labels):
            for j, lbls_j in enumerate(labels):
                #把标签列表转成集合，方便计算交集和并集：
                #lbls_i = [0, 2] → set_i = {0, 2}
                #lbls_j = [2, 3] → set_j = {2, 3}
                set_i = set(lbls_i)
                set_j = set(lbls_j)

                # 两个样本都没有认知标签时，认为它们标签完全相似
                if not set_i and not set_j:
                    sim = 1.0
                # 只有一个样本没有认知标签时，认为它们完全不相似
                elif not set_i or not set_j:
                    sim = 0.0
                else:
                    # Jaccard 相似度 = 标签集合交集大小 / 并集大小
                    intersection = len(set_i & set_j)
                    union = len(set_i | set_j)
                    sim = intersection / union if union > 0 else 0.0

                weight_matrix[i, j] = sim #把第 i 个样本和第 j 个样本的标签相似度写入矩阵。

        # 当前实现将 Jaccard 相似度二值化：只要有标签重叠就是正样本，否则是负样本
        pos_weight = (weight_matrix > 0).float() #正样本矩阵
        neg_weight = (weight_matrix == 0).float() #负样本矩阵

        # 将相似度指数化，用于构造类似 softmax 的对比学习比例项，sim_matrix 越大 → exp_sim 越大，sim_matrix 越小 → exp_sim 越小
        exp_sim = torch.exp(sim_matrix)
        # 正样本项：希望正样本相似度在所有样本相似度中占比更大
        #分子：每个样本和所有正样本的指数相似度之和。
        pos_term = -torch.log(
            (exp_sim * pos_weight).sum(dim=1) /
            (exp_sim.sum(dim=1) + 1e-8) #1e-8,防止除以 0。
        )
        # 负样本项：如果负样本相似度过高，会增加该惩罚项
        # .sum(dim=1) 对每个样本，把它和所有负样本的相似度惩罚加起来。
        neg_term = torch.log(1 + (exp_sim * neg_weight).sum(dim=1))

        # 对 batch 中所有样本取平均，得到最终认知多标签对比损失,mean(): 对 batch 里所有样本取平均，得到一个标量 loss
        return (pos_term + neg_term).mean()

    def training_step(self, batch, batch_idx):
        audio, video, text, emo_category, cognition_category = batch['audio'], batch['video'], batch['text'], batch['emo_category'], batch['cognition_category']
        loss, combined_feature_emo, combined_feature_cong = self.forward(audio, video, text, emo_category, cognition_category)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(audio), sync_dist=True)
        
        self.training_step_outputs.append({
            'emo_feature': combined_feature_emo.mean(dim=1).detach().cpu(),
            'cog_feature': combined_feature_cong.mean(dim=1).detach().cpu(),
            'emo_label': emo_category.detach().cpu(),
            'cog_label': cognition_category.detach().cpu()
        })
        return loss
    
    def validation_step(self, batch, batch_idx):
        audio, video, text, emo_category, cognition_category =batch['audio'],batch['video'],batch['text'],batch['emo_category'],batch['cognition_category']
        loss,combined_feature_emo, combined_feature_cong=self.forward(audio, video, text, emo_category, cognition_category)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=len(audio),sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=0.000013, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6)
        return optimizer
