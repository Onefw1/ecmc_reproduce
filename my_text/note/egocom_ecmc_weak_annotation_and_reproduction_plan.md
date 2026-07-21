# EgoCom 上的 ECMC-inspired 复现记录与计划

最后更新：2026-07-21

## 1. 实验定位

本项目复现 ECMC 的方法框架：对话 turn 级多模态输入、emotion/cognition caption 弱监督，以及 Stage 1/Stage 2 训练流程。

EgoCom 是日常第一视角多人对话数据集，不是临床访谈或认知障碍评估数据。因此本实验应表述为：

> 基于 EgoCom 的 ECMC-inspired 多模态对话理解与弱监督实验。

不能将结果解释为临床情绪诊断或认知障碍筛查能力。

## 2. 已完成的数据构造

### 2.1 数据来源与切片

- 使用 EgoCom 官方预提取特征及转写文本。
- 以 transcript 的 speaker turn 作为最小样本单位：同一说话人的连续 utterance 合并为一个 turn。
- 按 turn 时间范围匹配官方 4 秒 history 特征，导出 audio/video 序列。
- 没有重新抽取 HuBERT、VideoMAE 或文本特征，直接使用官方预提取结果。

特征规格：

- audio：`[T, 512]`
- video：`[T, 2048]`
- 所有样本均已通过文件存在性、shape 和 NaN 检查。

### 2.2 正式预处理数据

目录：`D:/py_code/ECMC/my_text/egocom_ecmc_formal/`

- train：3909 条
- val：312 条
- test：760 条
- 合计：4981 条

其中的 `MMDA/split_audio_f/` 和 `MMDA/split_video_f/` 保存每条样本对应的 `.npy` 特征。正式预处理数据保留为不可变的原始实验输入，不应被弱标注结果覆盖。

预处理 Notebook：`D:/py_code/ECMC/my_text/EgoCom_experiment/egocom_ecmc_formal_preprocess.ipynb`

## 3. 已完成的弱标注

### 3.1 标注原则

弱标注仅基于 transcript 文本：

- 不推断语音语调、表情、画面或临床病史。
- 不做心理健康或医学认知障碍诊断。
- 普通口语停顿、语气词、重复、语法不完整、ASR 误差及技术词回忆困难不视为认知缺陷。
- 只有存在明确文本证据时，才将认知维度标为 1。

字段定义：

- `emotion_bin`：`-1` 负向、`0` 中性或不明确、`1` 正向。
- `cognition_bin`：`[orientation, attention, memory, language]`，每一维均为 0/1。
- `emotion`、`cognition`：相应的英文短 caption。

标注模型为 DeepSeek-v4-flash，使用 OpenAI 兼容的 JSON response 格式；标注版本为 `v2_conservative`。

### 3.2 最终产物与校验结果

目录：`D:/py_code/ECMC/my_text/egocom_ecmc_labeled/`

- `train_full_v2_conservative.csv`：3909 条
- `val_full_v2_conservative.csv`：312 条
- `test_full_v2_conservative.csv`：760 条
- `annotation_report_full_v2_conservative.json`：分布报告
- `raw_llm_outputs/`：原始 API 返回日志

最终 CSV 已校验：

- 行数与正式预处理数据一致。
- 不存在占位标签。
- 不存在重复 `id`。
- 所有 `cognition_bin` 均为合法四维列表。
- 原始 JSONL 是追加式日志，可能保留历史失败记录；最终 CSV 才是训练的权威结果。

### 3.3 标签分布

| Split | Emotion -1 | Emotion 0 | Emotion 1 | Cognition 正例       |
| ----- | ---------: | --------: | --------: | -------------------- |
| train |        198 |      3052 |       659 | memory 4，language 1 |
| val   |         11 |       237 |        64 | orientation 1        |
| test  |         42 |       609 |       109 | 无                   |

结论：EgoCom 的日常任务型对话以中性表达为主，且几乎不包含能从纯文本可靠识别的认知异常证据。这个分布是数据域与保守提示词共同导致的正常结果，不应为了平衡标签而人为扩张认知正例。

## 4. 能否开始复现

可以开始。当前阶段已具备运行 ECMC-inspired 实验所需的数据、特征和弱标签。

但实验设计必须反映标签边界：

- emotion 是主要可用的弱监督信号和主要实验任务。
- cognition 只能作为极稀疏的辅助弱监督或消融项；它不适合做可靠的分类指标比较。
- Stage 2 的 cognition caption 可保留用于流程复现，但要明确其为保守文本弱标注，而非临床认知结论。

## 5. 接下来的复现步骤

### 当前进度（2026-07-21）

- 已完成 DataLoader 适配：`MMDAdataloader.py` 直接使用 EgoCom CSV 的 `id` 匹配 `.npy` 特征文件。
- 已完成 DataLoader 验证：train 集保留 3909 条；batch 形状为 audio `[B, 32, 512]`、video `[B, 32, 2048]`、cognition `[B, 4]`。
- 已完成模型输入适配：`MMDAmodel.py` 新增 `Linear(512, 768)` audio adapter 与 `Linear(2048, 768)` video adapter。
- 已将 `train.py` 切换到最终 labeled CSV 与 EgoCom 的 MMDA 特征目录。
- Stage 1 默认使用 `cognition_loss_weight=0.0`。原因是 EgoCom 的认知正例过少，常见的全零 batch 会使原始认知对比损失不稳定；认知分支仍保留给后续 caption 辅助实验。
- 已在 Colab 挂载 Google Drive，用于保存权重、数据、日志和 checkpoint；Colab 本地磁盘只用于每次会话的训练副本。
- 已生成并持久化 `weights/`（`bert-base-uncased` 文本编码器）与根目录 `pytorch_model.bin`（由同一 BERT 构造的 Q-Former 兼容初始化）。原 ECMC 仓库未提供作者实际使用的 Q-Former 权重来源，因此这不是作者同款 checkpoint。
- 已适配 `module/Qformer.py` 到当前 Colab 的新版 Transformers：移除了对旧私有工具函数路径的依赖，并补充新版权重绑定所需元数据。
- 已在 Colab 成功构造 Q-Former 并成功初始化 Stage 1 模型。
- 已完成真实 EgoCom batch size 为 2 的 forward/backward 连通性检查：audio `[2, 32, 512]`、video `[2, 32, 2048]`，输出 emotion/cognition feature 均为 `[2, 96, 768]`。该 batch 的两个样本均为中性，且没有负样本，因此原始 emotion contrastive loss 为 `0.0`、adapter 梯度为 `0.0`；这证明链路连通，但不是有效训练验证。
- 已完成有效训练 smoke test：使用 `[0, 0, 1, 1]` 的真实 batch size 为 4，得到 `loss=7.7583`，audio adapter gradient norm 为 `2.51e-4`，video adapter gradient norm 为 `4.06e-3`。Stage 1 的 forward 与 backward 均有效。
- Stage 1 训练入口使用 emotion 反频率加权采样，缓解中性样本占比过高导致 batch 缺少对比学习正负样本的问题。
- 为适配 Colab RAM，`train.py` 默认启用资源安全模式：冻结六个 Q-Former 及 query token，仅训练 EgoCom audio/video adapter；默认 `num_workers=0`。通过 `ECMC_TRAIN_QFORMERS=1` 才启用完整 Q-Former 微调。
- 已完成 100 step Stage 1 验证训练，训练正常结束并成功向 Google Drive 写入 checkpoint。每个 checkpoint 约 2.7GB，因为 Lightning 保存完整模型状态；`last.ckpt` 与同一步 checkpoint 基本重复。
- 首次完整 1 epoch 资源安全训练在预期的第 978 个 batch 前出现 `NaN`，对应 checkpoint 仅保留作调试，不可用于后续 Stage 2。初步修复了 emotion contrastive loss 在 FP16 下的 `log1p(exp(logit))` 溢出问题；后续日志进一步确认最终根因是 train split 的 3909 条样本不能被 batch size 4 整除，最后一个单样本 batch 不能计算对比损失。
- `softplus` 稳定性修复后的 100 step 验证训练已通过，未再报告 `NaN`。下一次完整 epoch checkpoint 必须输出到独立目录，避免与首次失败训练混淆。
- 后续完整 epoch 在 `16-mixed` 下仍出现 `NaN`，不视为 Colab 硬件故障。资源安全模式现将 emotion contrastive loss 的相似度与 log-probability 计算固定为 FP32，并默认以 `32-true` precision 运行，以排除半精度溢出或精度损失。
- 若 FP32 下仍出现 `NaN`，训练会在首个非有限 loss 时 fail-fast，并输出该 batch 的 emotion labels。资源安全模式将 learning rate 设为 `1e-6`，并启用 global norm 为 `1.0` 的梯度裁剪，以避免极端 batch 的梯度更新破坏 adapter 参数。
- 最新的 FP32、learning rate `1e-6`、gradient clip `1.0` 的 100-step 验证训练已完成，无 `NaN` 或 `FloatingPointError`；模型摘要确认仅 1,967,616 个 adapter 参数参与训练，其余约 708M 参数冻结。
- 训练 DataLoader 现启用 `drop_last=True`，因此完整 epoch 使用 977 个完整 batch（3908 次采样），丢弃最后一个单样本 batch；`contrastive_loss` 同时对 batch size 小于 2 的输入返回零损失作为保护。
- 已完成稳定的资源安全 Stage 1 完整 epoch：FP32、learning rate `1e-6`、gradient clip `1.0` 下运行 977/977 step，无 `NaN` 或异常退出；最终 `train_loss_epoch=7.490`、`val_loss_epoch=4.052`，checkpoint 回调已触发。训练使用 emotion 加权采样而验证使用原始分布，因此两者 loss 不应直接比较高低。
- 稳定 checkpoint 位于 Google Drive 的 `ECMC/checkpoints/stable_fp32_epoch2/`；该 checkpoint 是当前可用于恢复资源安全 Stage 1 的有效起点。
- 已从稳定 checkpoint 恢复并完成总计 5 个 epoch 的资源安全 Stage 1 训练：恢复后运行 4 个 epoch、3908 个完整 batch，最终为 `Epoch 4/4, 977/977`，无 `NaN` 或异常退出；最终 `train_loss_epoch=7.419`、`val_loss_epoch=4.052`。最终 checkpoint 保存到 `stable_fp32_epoch5/`。
- 下一组 Stage 1 实验为 adapter + emotion query token 微调：冻结六个 Q-Former 主体与文本 BERT，训练两个 adapter 和 emotion 流的三组 query token。由于 cognition loss 为 0，cognition 流 query token 保持冻结。该实验从 `stable_fp32_epoch5/` 加载模型权重，但不恢复旧优化器状态。

下一步：执行 adapter + query token Stage 1 实验；随后比较其与 adapter-only 基线的稳定性和 loss，再准备 Stage 2 所需的 LLaMA 权重与适合 EgoCom 日常对话的 emotion caption prompt。

### Stage 1 训练产物（2026-07-21）

#### 最新更新：adapter + emotion query token 实验已完成

- 初始化：从 `ECMC/checkpoints/stable_fp32_epoch5/stage1-epoch=04-step=04885.ckpt` 仅加载模型权重；未恢复 adapter-only 实验的 optimizer 状态。
- 训练模块：audio/video adapter 加 emotion 流的三组 query token；六个 Q-Former 主体、文本 BERT 与 cognition 流 query token 保持冻结。
- 配置：`batch_size=4`、FP32（`32-true`）、learning rate `1e-6`、gradient clip `1.0`、`cognition_loss_weight=0.0`。
- 结果：完成 5 个 epoch、共 4885 个训练 step；末尾为 `Epoch 4/4, 977/977`，全程无 NaN 或异常退出；最终 `train_loss_epoch=7.517`、`val_loss_epoch=4.050`。
- 产物：最终 checkpoint 已保存至 Google Drive `ECMC/checkpoints/stable_querytokens_epoch5/`。

本节中“后续 query token 实验”及“下一步执行 adapter + query token”均已由本次结果替代。下一步是整理两组 Stage 1 的可比记录，再启动 Stage 2 emotion caption 实验。

#### Stage 2 emotion caption 准备状态

- 已将 `MMDAmodel2.py` 改为 EgoCom emotion-only decoder：输入为 Stage 1 的 emotion soft tokens，监督目标仅为 `emotion` caption；prompt 明确禁止推断医疗、心理健康、认知障碍或未在对话中出现的信息。
- 已将 `train2.py` 改为读取 EgoCom 的 `MMDA` 特征和 `*_full_v2_conservative.csv`，支持单 GPU、`batch_size=1`、FP16 与 Stage 1 checkpoint 初始化。
- Stage 2 冻结 Stage 1 encoder 和 LLaMA，只训练 `emo_llama_project`；默认 `ECMC_MAX_STEPS=20`，必须先完成 smoke test，尚未开始正式 caption 训练。
- 运行前需准备一个本地 LLaMA-compatible decoder checkpoint。轻量测试建议使用 `TinyLlama/TinyLlama-1.1B-Chat-v1.0`；其输出只能作为弱标注 caption 重写实验，不构成临床结论。

当前有效训练配置：

- 任务：emotion-focused contrastive learning；`cognition_loss_weight=0.0`。
- 可训练模块：`audio_adapter` 与 `video_adapter`，共 1,967,616 个参数。
- 冻结模块：文本 BERT、六个 Q-Former、六组 query token、未使用的 LLaMA 投影层，约 708M 参数不更新。
- 数据：train 3909 条，`batch_size=4`，emotion 反频率加权采样；`drop_last=True` 后每个 epoch 使用 977 个完整 batch。
- 数值配置：FP32（`32-true`）、AdamW learning rate `1e-6`、global gradient clip `1.0`、`num_workers=0`。
- 训练过程：先完成 1 个稳定 epoch；再从 `stable_fp32_epoch2/` 恢复，完成总计 5 个 epoch。

后续 query token 实验将解除 emotion 流三组 query token 的冻结。Q-Former 主体仍冻结，因此不会为约 595M 个 Q-Former 参数创建 AdamW 优化器状态；训练参数只比 adapter-only 基线增加约 73,728 个 query token 参数。

训练结果：

| 阶段 | 训练进度 | train_loss_epoch | val_loss_epoch | 状态 |
| --- | --- | ---: | ---: | --- |
| 稳定起点 | 1 epoch，977/977 step | 7.490 | 4.052 | 无 NaN，checkpoint 可恢复 |
| 当前有效结果 | 总计 5 epoch，Epoch 4/4、977/977 step | 7.419 | 4.052 | 无 NaN，Stage 1 基线完成 |

训练产物：

- 有效 Stage 1 checkpoint：Google Drive `ECMC/checkpoints/stable_fp32_epoch5/`。
- 中间可恢复 checkpoint：Google Drive `ECMC/checkpoints/stable_fp32_epoch2/`。
- 早期 `last.ckpt`、`step=00050`、`step=00100` 与出现 NaN 的 `step=00978` 仅为调试产物，不应用于 Stage 2。

解释边界：当前 loss 表明资源安全的训练流程稳定运行，但不是 emotion 分类准确率或 caption 质量指标。训练集采用加权采样、验证集保留原始分布，train/val loss 不应直接横向比较；Stage 2 生成与人工样例检查仍未开始。

### A. 适配数据加载器（已完成）

目标：让 `MMDAdataloader.py` 读取最终 labeled CSV，而不是 formal CSV。

使用：

- `my_text/egocom_ecmc_labeled/train_full_v2_conservative.csv`
- `my_text/egocom_ecmc_labeled/val_full_v2_conservative.csv`
- `my_text/egocom_ecmc_labeled/test_full_v2_conservative.csv`

验收：能取到一个 batch，且 audio 为 `[B, T, 512]`、video 为 `[B, T, 2048]`、cognition 为 `[B, 4]`。

### B. 适配模型输入维度（已完成）

原 ECMC 模型默认特征宽度接近 768，而 EgoCom 的特征维度不同。建议在 `MMDAmodel.py` 的输入端增加最小适配层：

```python
audio_adapter = nn.Linear(512, 768)
video_adapter = nn.Linear(2048, 768)
```

适配后保持后续 Q-Former/LLM 主结构不变。这样改动范围小，也便于与原 ECMC 架构对应。

### C. 本地与 Colab 调试（已完成）

已完成：

1. DataLoader 单 batch 检查。
2. 模型单 batch forward。
3. 小规模 loss 计算与反向传播验证。

完整训练继续在 Colab/GPU 执行；轻薄本不承担完整训练。

### D. Colab/GPU 训练（进行中）

1. Stage 1：当前使用资源安全 adapter-only 配置，以 emotion 为主监督；Q-Former、query token 和文本编码器冻结，cognition loss 权重为 0。
2. 已完成 100 step、1 个稳定完整 epoch及从 checkpoint 恢复后的总计 5 个 epoch；资源安全 Stage 1 基线完成。
3. Stage 2：先训练 emotion caption；在改写原始临床 prompt 后，再将 cognition caption 作为辅助实验。
4. 对 cognition 分支报告样例与定性结果，不以其稀疏正例计算具有统计意义的分类性能。

## 6. 报告与论文写法

建议明确写出：

- EgoCom 仅用于评估 ECMC 框架迁移到非临床日常对话的可行性。
- emotion/cognition 标注来自 LLM 弱监督，并经过保守提示词约束。
- cognition 正例稀疏源于数据集不包含临床认知评估证据，而非模型性能结论。
- 主要定量结果应围绕 emotion 任务、caption 质量及多模态消融展开。

## 7. 关键文件

- 预处理：`D:/py_code/ECMC/my_text/EgoCom_experiment/egocom_ecmc_formal_preprocess.ipynb`
- 弱标注：`D:/py_code/ECMC/my_text/EgoCom_experiment/egocom_ecmc_weak_annotation.ipynb`
- 数据加载器：`D:/py_code/ECMC/MMDAdataloader.py`
- Stage 1 模型：`D:/py_code/ECMC/MMDAmodel.py`
- Stage 2 模型与提示词：`D:/py_code/ECMC/MMDAmodel2.py`

## 8. 汇报进度

当前已完成 EgoCom 到 ECMC-inspired 实验数据的构造：以 speaker turn 为样本单位，完成 turn 文本、官方预提取 audio/video history 特征的对齐，以及 train/val/test 划分，共得到 4981 条样本。已使用保守的文本弱标注策略生成 emotion/cognition 标签和 caption；其中 emotion 可作为主要监督信号，cognition 因日常对话中缺少明确证据而仅作为稀疏辅助监督，不作临床认知结论。

模型侧已完成 EgoCom 特征维度适配，将 audio 的 512 维和 video 的 2048 维特征映射到 ECMC Q-Former 所需的 768 维。DataLoader、模型初始化、真实 batch 的 forward/backward 和 100-step Stage 1 训练均已跑通，且 checkpoint 可保存到 Google Drive。

当前已完成资源安全配置下无 NaN 的总计 5 个 epoch Stage 1 训练，最终 `train_loss_epoch=7.419`、`val_loss_epoch=4.052`，并保存了最终 checkpoint。下一步是准备 Stage 2 的 LLaMA 权重和适配后的 emotion caption prompt。整个实验应表述为 EgoCom 上的 ECMC-inspired 弱监督多模态对话实验，而非原论文临床数据集上的严格复现。
