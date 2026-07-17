# EgoCom 上的 ECMC-inspired 复现记录与计划

最后更新：2026-07-17

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

| Split | Emotion -1 | Emotion 0 | Emotion 1 | Cognition 正例 |
| --- | ---: | ---: | ---: | --- |
| train | 198 | 3052 | 659 | memory 4，language 1 |
| val | 11 | 237 | 64 | orientation 1 |
| test | 42 | 609 | 109 | 无 |

结论：EgoCom 的日常任务型对话以中性表达为主，且几乎不包含能从纯文本可靠识别的认知异常证据。这个分布是数据域与保守提示词共同导致的正常结果，不应为了平衡标签而人为扩张认知正例。

## 4. 能否开始复现

可以开始。当前阶段已具备运行 ECMC-inspired 实验所需的数据、特征和弱标签。

但实验设计必须反映标签边界：

- emotion 是主要可用的弱监督信号和主要实验任务。
- cognition 只能作为极稀疏的辅助弱监督或消融项；它不适合做可靠的分类指标比较。
- Stage 2 的 cognition caption 可保留用于流程复现，但要明确其为保守文本弱标注，而非临床认知结论。

## 5. 接下来的复现步骤

### 当前进度（2026-07-17）

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
- 首次完整 1 epoch 资源安全训练完成 978 step，但训练末期 `train_loss` 出现 `NaN`，对应 checkpoint 仅保留作调试，不可用于后续 Stage 2。根因是 emotion contrastive loss 的 `log1p(exp(logit))` 在 FP16 混合精度下可能溢出；已改为数学等价的稳定 `softplus(logit)` 实现，需重新运行验证训练。

下一步：执行完整 1 epoch（978 step）的资源安全 Stage 1 训练，使用 batch size 4，checkpoint 间隔设为 978 step，并关闭 `save_last` 以避免重复保存。`train.py` 已支持通过环境变量设置 batch size、step 数、worker 数、checkpoint 间隔和是否保存 `last.ckpt`。

### A. 适配数据加载器

目标：让 `MMDAdataloader.py` 读取最终 labeled CSV，而不是 formal CSV。

使用：

- `my_text/egocom_ecmc_labeled/train_full_v2_conservative.csv`
- `my_text/egocom_ecmc_labeled/val_full_v2_conservative.csv`
- `my_text/egocom_ecmc_labeled/test_full_v2_conservative.csv`

验收：能取到一个 batch，且 audio 为 `[B, T, 512]`、video 为 `[B, T, 2048]`、cognition 为 `[B, 4]`。

### B. 适配模型输入维度

原 ECMC 模型默认特征宽度接近 768，而 EgoCom 的特征维度不同。建议在 `MMDAmodel.py` 的输入端增加最小适配层：

```python
audio_adapter = nn.Linear(512, 768)
video_adapter = nn.Linear(2048, 768)
```

适配后保持后续 Q-Former/LLM 主结构不变。这样改动范围小，也便于与原 ECMC 架构对应。

### C. 本地调试

轻薄本只执行：

1. DataLoader 单 batch 检查。
2. 模型单 batch forward。
3. 小规模 loss 计算与反向传播验证。

不建议在本地进行完整训练，以免内存和显存压力影响环境稳定性。

### D. Colab/GPU 训练

1. Stage 1：先训练 adapter 与 ECMC 的对比学习相关模块；冻结大型语言模型；以 emotion 为主监督。
2. 先跑约 100 step，确认 loss、显存、梯度和 checkpoint 正常，再扩展为完整 epoch。
3. Stage 2：先训练 emotion caption；随后可运行包含 cognition caption 的辅助实验。
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
