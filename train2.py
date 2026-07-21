import os

import lightning.pytorch as pl
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from MMDAdataloader import MMDADataset, collate_multimodal_batch
from MMDAmodel2 import ECMCLLaMA


def required_env(name):
    value = os.getenv(name, "")
    if not value:
        raise ValueError(f"Set {name} before starting Stage 2.")
    return value


pl.seed_everything(666)
data_root = "my_text/egocom_ecmc_formal/MMDA"
label_root = "my_text/egocom_ecmc_labeled"
stage1_ckpt = required_env("ECMC_STAGE1_CKPT")
llama_ckpt = required_env("ECMC_LLAMA_CKPT")

batch_size = int(os.getenv("ECMC_BATCH_SIZE", "1"))
num_workers = int(os.getenv("ECMC_NUM_WORKERS", "0"))
max_steps = int(os.getenv("ECMC_MAX_STEPS", "20"))
max_epochs = int(os.getenv("ECMC_MAX_EPOCHS", "1"))
learning_rate = float(os.getenv("ECMC_LEARNING_RATE", "1e-4"))
checkpoint_dir = os.getenv("ECMC_CHECKPOINT_DIR", "./checkpoints/stage2_emotion")
checkpoint_every_n_steps = int(os.getenv("ECMC_CHECKPOINT_EVERY_N_STEPS", "3909"))
save_last = os.getenv("ECMC_SAVE_LAST", "0") == "1"

model = ECMCLLaMA(llama_ckpt=llama_ckpt, learning_rate=learning_rate)
checkpoint = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
missing_keys, unexpected_keys = model.load_state_dict(checkpoint["state_dict"], strict=False)
print(f"Initialized Stage 1 encoder from: {stage1_ckpt}")
print(f"Missing keys: {len(missing_keys)}; unexpected keys: {len(unexpected_keys)}")
print(f"Stage 2 trainable parameters: {model.trainable_parameter_count:,}")

dataset_kwargs = {
    "root_dir": data_root,
    "modalities": ("text", "audio", "video"),
    "max_seq_len": {"audio": 32, "video": 32},
}
train_set = MMDADataset(
    split_file=os.path.join(label_root, "train_full_v2_conservative.csv"),
    **dataset_kwargs,
)
val_set = MMDADataset(
    split_file=os.path.join(label_root, "val_full_v2_conservative.csv"),
    **dataset_kwargs,
)
if not len(train_set) or not len(val_set):
    raise RuntimeError("EgoCom features are missing from /content. Re-run the local rsync setup cells.")

loader_kwargs = {"batch_size": batch_size, "num_workers": num_workers, "collate_fn": collate_multimodal_batch}
if num_workers > 0:
    loader_kwargs.update(prefetch_factor=2, persistent_workers=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename="stage2-emotion-{epoch:02d}-{step:05d}",
    save_top_k=-1,
    every_n_train_steps=checkpoint_every_n_steps,
    save_last=save_last,
)
trainer = Trainer(
    accelerator="gpu",
    devices=1,
    precision="16-mixed",
    max_epochs=max_epochs,
    max_steps=max_steps,
    gradient_clip_val=1.0,
    callbacks=[checkpoint_callback],
    logger=TensorBoardLogger(save_dir="./logger", name="MMDA_stage2_emotion"),
    log_every_n_steps=10,
)
trainer.fit(model, train_loader, val_loader)
