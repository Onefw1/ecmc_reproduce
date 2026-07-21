import os

import lightning.pytorch as pl
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from MMDAmodel import ECMC
from module.modeling_llama import LlamaForCausalLM


class ECMCLLaMA(ECMC):
    """Stage 2 emotion-caption decoder for the non-clinical EgoCom setting."""

    def __init__(self, llama_ckpt, learning_rate=1e-4, decoder_dtype="float32"):
        super().__init__(
            cognition_loss_weight=0.0,
            train_qformers=False,
            train_query_tokens=False,
        )
        dtype_by_name = {"float16": torch.float16, "float32": torch.float32}
        if decoder_dtype not in dtype_by_name:
            raise ValueError("decoder_dtype must be 'float16' or 'float32'.")
        self.llama_model = LlamaForCausalLM.from_pretrained(
            llama_ckpt,
            torch_dtype=dtype_by_name[decoder_dtype],
        )
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_ckpt, use_fast=False)
        if self.llama_tokenizer.pad_token_id is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        self.emo_llama_project = nn.Linear(768, self.llama_model.config.hidden_size)
        self.learning_rate = learning_rate

        # Stage 2 learns only the bridge-to-decoder projection. The Stage 1
        # encoder and decoder weights remain fixed, which keeps Colab memory use manageable.
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name.startswith("emo_llama_project")
        self._set_frozen_modules_to_eval()

    def _set_frozen_modules_to_eval(self):
        for module in (
            self.text2vec_model,
            self.audio_Qformer,
            self.video_Qformer,
            self.text_Qformer,
            self.audio_Qformer_cong,
            self.video_Qformer_cong,
            self.text_Qformer_cong,
            self.llama_model,
        ):
            module.eval()

    def train(self, mode=True):
        super().train(mode)
        self._set_frozen_modules_to_eval()
        return self

    @property
    def trainable_parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _prompt_tokens(self, batch_size):
        prompt = (
            "The following soft tokens summarize a speaker turn from an everyday conversation. "
            "Write one brief, evidence-grounded emotion caption. Do not infer mental health, "
            "medical conditions, facial expressions, or information absent from the conversation.\n"
            "Emotion caption:\n"
        )
        tokens = self.llama_tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=96,
        ).to(self.device)
        return tokens.input_ids.expand(batch_size, -1), tokens.attention_mask.expand(batch_size, -1)

    @staticmethod
    def _require_finite(tensor, name):
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite values detected in Stage 2 {name}.")

    def forward(self, audio, video, text, emo_category, cognition_category, emotion_cap):
        # The frozen Stage 1 encoder supplies 96 emotion soft tokens.
        with torch.no_grad():
            _, emotion_features, _ = super().forward(
                audio, video, text, emo_category, cognition_category
            )
        self._require_finite(emotion_features, "emotion features")

        decoder_dtype = next(self.llama_model.parameters()).dtype
        emotion_input = self.emo_llama_project(emotion_features).to(decoder_dtype)
        self._require_finite(emotion_input, "projected emotion tokens")
        batch_size = emotion_input.size(0)

        prompt_ids, prompt_mask = self._prompt_tokens(batch_size)
        prompt_embeds = self.llama_model.model.embed_tokens(prompt_ids)

        eos = self.llama_tokenizer.eos_token or ""
        captions = [f"{str(caption).strip()}{eos}" for caption in emotion_cap]
        caption_tokens = self.llama_tokenizer(
            captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=96,
        ).to(self.device)
        targets = caption_tokens.input_ids.masked_fill(
            caption_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        bos = torch.full(
            (batch_size, 1),
            self.llama_tokenizer.bos_token_id,
            dtype=torch.long,
            device=self.device,
        )
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        caption_embeds = self.llama_model.model.embed_tokens(caption_tokens.input_ids)
        input_embeds = torch.cat(
            [bos_embeds, emotion_input, prompt_embeds, caption_embeds], dim=1
        )
        self._require_finite(input_embeds, "decoder input embeddings")
        attention_mask = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.long, device=self.device),
                torch.ones(emotion_input.shape[:-1], dtype=torch.long, device=self.device),
                prompt_mask,
                caption_tokens.attention_mask,
            ],
            dim=1,
        )
        outputs = self.llama_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=targets,
            use_cache=False,
            return_dict=True,
        )
        self._require_finite(outputs.logits, "decoder logits")
        self._require_finite(outputs.loss, "caption loss")
        return outputs.loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(
            batch["audio"], batch["video"], batch["text"], batch["emo_category"],
            batch["cognition_category"], batch["emotion_cap"],
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch["audio"]))
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(
            batch["audio"], batch["video"], batch["text"], batch["emo_category"],
            batch["cognition_category"], batch["emotion_cap"],
        )
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(batch["audio"]))
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.emo_llama_project.parameters(), lr=self.learning_rate)

    def on_save_checkpoint(self, checkpoint):
        """Persist only the Stage 2 projection; frozen base weights are external assets."""
        checkpoint["state_dict"] = {
            name: value
            for name, value in checkpoint["state_dict"].items()
            if name.startswith("emo_llama_project.")
        }
