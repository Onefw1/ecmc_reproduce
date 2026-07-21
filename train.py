from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from torch.nn import CrossEntropyLoss
import torch
from lightning.pytorch import Trainer, LightningDataModule, LightningModule, Callback, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl
import torch.optim as optim
import os
print("Current working directory:", os.getcwd())
print("Files in directory:", os.listdir())
from MMDAdataloader import *
from MMDAmodel import ECMC

pl.seed_everything(666)
data_root = "my_text/egocom_ecmc_formal/MMDA"
label_root = "my_text/egocom_ecmc_labeled"

# EgoCom cognition positives are too sparse for stable contrastive supervision.
# Adapter-only is the default resource-safe mode for Colab validation.
train_qformers = os.getenv("ECMC_TRAIN_QFORMERS", "0") == "1"
train_query_tokens = os.getenv("ECMC_TRAIN_QUERY_TOKENS", "0") == "1"
learning_rate = float(os.getenv("ECMC_LEARNING_RATE", "1e-6"))
model=ECMC(
    cognition_loss_weight=0.0,
    train_qformers=train_qformers,
    train_query_tokens=train_query_tokens,
    learning_rate=learning_rate,
)

batch_size = int(os.getenv("ECMC_BATCH_SIZE", "4"))
num_workers = int(os.getenv("ECMC_NUM_WORKERS", "0"))
max_steps = int(os.getenv("ECMC_MAX_STEPS", "-1"))
max_epochs = int(os.getenv("ECMC_MAX_EPOCHS", "100"))
checkpoint_every_n_steps = int(os.getenv("ECMC_CHECKPOINT_EVERY_N_STEPS", "100"))
save_last = os.getenv("ECMC_SAVE_LAST", "1") == "1"
checkpoint_dir = os.getenv("ECMC_CHECKPOINT_DIR", "./checkpoints")
precision = os.getenv("ECMC_PRECISION", "32-true")
gradient_clip_val = float(os.getenv("ECMC_GRADIENT_CLIP_VAL", "1.0"))
resume_ckpt = os.getenv("ECMC_RESUME_CKPT", "")
init_weights_ckpt = os.getenv("ECMC_INIT_WEIGHTS_CKPT", "")
if resume_ckpt and init_weights_ckpt:
    raise ValueError("Use only one of ECMC_RESUME_CKPT or ECMC_INIT_WEIGHTS_CKPT.")
if init_weights_ckpt:
    checkpoint = torch.load(init_weights_ckpt, map_location="cpu", weights_only=False)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["state_dict"], strict=False)
    print(f"Initialized model weights from: {init_weights_ckpt}")
    print(f"Missing keys: {len(missing_keys)}; unexpected keys: {len(unexpected_keys)}")
print(f"Resource-safe mode: {not train_qformers}; trainable parameters: "
      f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Train query tokens: {train_query_tokens}; trainer precision: {precision}; "
      f"learning rate: {learning_rate}; gradient clip: {gradient_clip_val}")
#create the train and val set
train_set = MMDADataset(
        root_dir=data_root,
        split_file=os.path.join(label_root, "train_full_v2_conservative.csv"),
        modalities=("text", "audio", "video"),
        max_seq_len={"audio": 32, "video": 32},
    )
val_set = MMDADataset(
        root_dir=data_root,
        split_file=os.path.join(label_root, "val_full_v2_conservative.csv"),
        modalities=("text", "audio", "video"),
        max_seq_len={"audio": 32, "video": 32},
    )

loader_kwargs = {
    "batch_size": batch_size,
    "collate_fn": collate_multimodal_batch,
    "num_workers": num_workers,
}
if num_workers > 0:
    loader_kwargs.update(prefetch_factor=2, persistent_workers=True)

# EgoCom is dominated by neutral turns. Balance emotion sampling so Stage 1
# contrastive batches contain useful positive and negative pairs more often.
emotion_counts = train_set.df["emotion_bin"].value_counts().to_dict()
sample_weights = train_set.df["emotion_bin"].map(
    lambda label: 1.0 / emotion_counts[label]
).to_numpy()
train_sampler = WeightedRandomSampler(
    weights=torch.as_tensor(sample_weights, dtype=torch.double),
    num_samples=len(train_set),
    replacement=True,
)

train_loader=DataLoader(train_set, sampler=train_sampler, shuffle=False, drop_last=True, **loader_kwargs)
val_loader=DataLoader(val_set, shuffle=False, **loader_kwargs)

#put your own checkpoint dir here
checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='stage1-{epoch:02d}-{step:05d}',
        save_top_k=-1,
        every_n_train_steps=checkpoint_every_n_steps,
        save_last=save_last,
    )

trainer = pl.Trainer(
    profiler="simple",
    logger=TensorBoardLogger(name='MMDA_model',save_dir='./logger'),
    accelerator='gpu',
    max_epochs=max_epochs,
    max_steps=max_steps,
    devices=1,
    log_every_n_steps=50,
    precision=precision,
    gradient_clip_val=gradient_clip_val,
    gradient_clip_algorithm="norm",
    callbacks=[checkpoint_callback],
    #accumulate_grad_batches=4,
    #strategy="ddp_find_unused_parameters_true"
    )

trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt or None)
