# EAV 多模态情绪识别阶段汇报

## 1. 研究目标

当前工作不是完整复现 ECMC 原论文，而是在 EAV 数据集上做一个 ECMC-inspired 的多模态情绪识别实验。

更准确的定位是：

> 在 EAV 预提取 audio / video / EEG 特征上，验证 ECMC 中可迁移的多模态表示学习思想是否有效。

当前任务是五分类情绪识别：

```text
Neutral / Sadness / Anger / Happiness / Calmness
```

主要关注的问题是：

1. 简单特征平均是否足够？
2. 保留时序信息是否有效？
3. audio/video 多模态融合是否有效？
4. query-token 和跨模态对齐是否有帮助？
5. EEG 作为第三模态是否带来额外增益？

## 2. 数据与任务

当前使用的是 EAV 的预提取特征版本，不是原始 `.wav` / `.mp4`。

每个 subject 包含：

| 文件 | shape 示例 | 含义 |
| ---- | ---------: | ---- |
| `audio.npy` | `(400, 501, 256)` | 每个样本 501 个音频时间片，每个时间片 256 维 |
| `video.npy` | `(400, 25, 512)` | 每个样本 25 个视频时间片，每个时间片 512 维 |
| `eeg.npy` | `(400, 500, 30)` | 每个样本 500 个 EEG 时间点，每个时间点 30 维 |
| `labels.npy` | `(400,)` | 五分类情绪标签 |
| `split.npy` | `(400,)` | 训练 / 测试划分 |

数据规模：

| 项目 | 数值 |
| ---- | ---: |
| subjects | 42 |
| 每个 subject 样本数 | 400 |
| 总样本数 | 16800 |
| 训练集 | 11760 |
| 测试集 | 5040 |
| 测试集每类样本数 | 1008 |

由于测试集类别均衡，`Accuracy` 和 `Macro-F1` 都有参考价值；后续主要用 `Macro-F1` 比较模型。

## 3. 实验设计

实验按由浅到深的顺序展开。

### 3.1 Mean Pooling Baseline

先对时间维度直接求平均，再用 MLP 分类：

```text
audio: (501, 256) -> (256)
video: (25, 512) -> (512)
```

目的：建立最简单 baseline，判断 audio/video 是否有基本情绪信息。

### 3.2 Temporal Attention 模型

保留时间片序列，用 projection + attention pooling 聚合关键时间片：

```text
audio/video/eeg sequence -> temporal projection -> attention pooling -> classifier
```

目的：验证时序信息是否比简单平均更有效。

### 3.3 QueryToken 模型

引入可学习 query tokens，从不同模态序列中抽取任务相关表示：

```text
learnable query tokens
-> query self-attention
-> query-to-modality cross-attention
-> fusion
-> classifier
```

这是轻量 QueryToken 实现，借鉴 ECMC / Q-Former 思路，但没有直接复用 ECMC 原版 Q-Former。

### 3.4 对比学习实验

做了两类对比学习尝试：

| 方法 | 含义 | 当前结果 |
| ---- | ---- | -------- |
| `AVCL` | 同一样本 audio/video 表示靠近，不同样本远离 | 有效 |
| `label-SCL` | 同类别样本靠近，不同类别样本远离 | 当前无收益 |

### 3.5 EEG 接入实验

最后加入 EEG，做三组关键实验：

```text
Temporal EEG
Temporal AVE
QueryToken AVE
```

目的：判断 EEG 单独是否有信号，以及加入 audio/video 后是否带来增益。

## 4. 关键结果总表

| 模型 | 模态 | 说明 | Accuracy | Macro-F1 |
| ---- | ---- | ---- | -------: | -------: |
| MLP AV | A+V | mean pooling + MLP | 0.5726 | 0.5734 |
| Temporal Audio | A | audio 时序建模 | 0.5752 | 0.5777 |
| Temporal Video | V | video 时序建模 | 0.5905 | 0.5864 |
| Temporal AV | A+V | audio/video 时序融合 | 0.7355 | 0.7368 |
| QueryToken AV, q=16 | A+V | query-token + CE | 0.7308 | 0.7295 |
| QueryToken AV, q=32 | A+V | query-token + CE | 0.7361 | 0.7356 |
| QueryToken AV + AVCL, q=16 | A+V | query-token + audio-video 对齐 | 0.7486 | 0.7470 |
| QueryToken AV + AVCL, q=32 | A+V | 更大 query 数量 + AVCL | 0.7405 | 0.7417 |
| QueryToken AV + label-SCL, q=16 | A+V | query-token + 标签对比学习 | 0.7234 | 0.7234 |
| Temporal EEG | E | EEG 单模态时序建模 | 0.4038 | 0.4047 |
| Temporal AVE | A+V+E | audio/video/eeg 时序融合 | **0.7524** | **0.7533** |
| QueryToken AVE, q=16 | A+V+E | 三模态 query-token + CE | 0.7474 | 0.7463 |

当前最佳结果：

```text
Temporal AVE
Accuracy = 0.7524
Macro-F1 = 0.7533
```

## 5. 关键发现

### 5.1 时序建模非常有效

简单 AV mean pooling 到 Temporal AV：

```text
Macro-F1: 0.5734 -> 0.7368
提升: +0.1634
```

说明 EAV 的 audio/video 时间片中包含关键情绪信息，直接平均会损失大量时序细节。

### 5.2 Audio 和 video 存在明显互补

Temporal AV 明显强于任一单模态：

```text
Temporal Audio: 0.5777
Temporal Video: 0.5864
Temporal AV:    0.7368
```

说明 audio 和 video 在 EAV 情绪识别中提供互补信息。

### 5.3 QueryToken + AVCL 有效，但不是当前最强

QueryToken AV + AVCL 达到：

```text
Macro-F1 = 0.7470
```

相比 Temporal AV：

```text
0.7368 -> 0.7470
提升: +0.0102
```

这说明 query-token 表示抽取和音视频实例级对齐是有效的。

但后续结果显示：

```text
q=32 + AVCL: 0.7417 < q=16 + AVCL: 0.7470
label-SCL:   0.7234 < q=16 CE:     0.7295
```

因此不能简单认为“更复杂一定更好”。当前 EAV 上，AVCL 有效，label-SCL 暂时无收益。

### 5.4 EEG 单模态弱，但三模态融合有效

Temporal EEG 单独结果：

```text
Macro-F1 = 0.4047
```

它高于随机五分类的 0.20，说明 EEG 有情绪信息；但远弱于 audio/video。

加入 EEG 后：

```text
Temporal AV:  0.7368
Temporal AVE: 0.7533
提升: +0.0165
```

说明 EEG 作为第三模态对 audio/video 有补充价值。

### 5.5 当前最稳主结果是 Temporal AVE

虽然 QueryToken AV + AVCL 是更接近 ECMC 结构思想的一组实验，但加入 EEG 后，Temporal AVE 成为当前最佳：

```text
Temporal AVE:              0.7533
QueryToken AV + AVCL q=16: 0.7470
QueryToken AVE q=16 CE:    0.7463
```

因此当前阶段主结果应收敛到 Temporal AVE，而不是继续发散调 label-SCL。

## 6. 与 ECMC 的关系

当前实验不是 ECMC 完整复现。

ECMC 原论文面向 MMDA 临床访谈，包含：

```text
audio / video / text 三模态
访谈问答轮次切分
emotion caption
cognition caption
cognition label
LLaMA 生成阶段
```

当前 EAV 数据只有：

```text
audio.npy
video.npy
eeg.npy
五分类情绪标签
```

缺少：

```text
原始音频 / 视频
ASR transcript
text 分支
emotion caption
cognition caption
cognition label
临床访谈上下文
```

所以当前工作更准确地说是：

> ECMC-inspired EAV 多模态情绪识别实验。

已经迁移和验证的 ECMC 相关思想包括：

| ECMC 思想 | 当前 EAV 实验对应实现 | 结果 |
| --------- | --------------------- | ---- |
| 多模态表示学习 | audio/video/eeg 融合 | 有效 |
| 时序特征建模 | Temporal attention pooling | 有效 |
| query-token 表示抽取 | QueryToken AV / AVE | 部分有效 |
| 跨模态对齐 | AVCL | 有效 |
| 标签监督对比学习 | label-SCL | 当前无收益 |

不能声称的内容：

```text
不能说已经完整复现 ECMC
不能说完成 emotion-cognition captioning
不能说复现了 LLaMA 生成阶段
不能说有认知标签建模
```

## 7. 当前不足

1. 当前使用的是预提取特征，不是原始音频/视频，因此无法做 ASR 或重新抽取底层特征。
2. 当前没有 text 模态，不能复现 ECMC 的 audio-video-text 三模态结构。
3. 当前没有 emotion/cognition caption 标注，不能复现生成式 captioning 任务。
4. 当前 best epoch 是在测试集表现上记录的，正式实验最好增加 validation split。
5. 当前结果大多是单随机种子，需要多 seed 验证稳定性。
6. QueryToken AVE 还没有加入 AVCL，因此三模态 query-token 对齐是否有效尚未验证。

## 8. 下一步计划

建议停止继续发散实验，先冻结当前阶段结果。

当前主结果：

```text
Temporal AVE
Accuracy = 0.7524
Macro-F1 = 0.7533
```

下一步只做少量高价值补充：

1. 对 Temporal AVE 做多随机种子复跑，验证 `0.7533` 是否稳定。
2. 可选补充 QueryToken AVE + AVCL，验证三模态 query-token 加跨模态对齐后是否能超过 Temporal AVE。
3. 整理 notebook、结果表格、分类报告，形成可复现的阶段材料。
4. 如果后续拿到原始音频/视频，再考虑 ASR text 分支。
5. 如果要完整复现 ECMC，需要继续争取 MMDA 或类似心理健康访谈数据。

## 9. 给师兄汇报的话术

可以这样说：

> 我现在还不是完整复现 ECMC，而是在 EAV 上先做 ECMC-inspired 的多模态情绪识别验证。EAV 当前提供的是 audio、video、EEG 的预提取特征和五分类情绪标签，没有原始音频、文本转录、认知标签和 caption 标注，所以不能直接复现 ECMC 的 emotion-cognition captioning。

> 我先做了简单 mean pooling baseline，AV 的 Macro-F1 是 0.5734。然后保留时间片做 temporal attention，Temporal AV 提升到 0.7368，说明 audio/video 的时序信息很重要。之后我尝试了 query-token 结构和音视频对齐，QueryToken AV + AVCL 提升到 0.7470，说明 query-token 和跨模态对齐是有效的。但 label-SCL 没有带来收益。

> 最近我把 EEG 接入进来。EEG 单模态 Macro-F1 是 0.4047，单独看比较弱，但加入 audio/video 后，Temporal AVE 达到 0.7533，超过之前所有结果，说明 EEG 对音视频情绪识别有补充价值。当前阶段我会把 Temporal AVE 作为主结果，后续优先做多随机种子验证和可复现整理，而不是继续发散调模型。

一句话版本：

> 当前阶段我在 EAV 上验证了 ECMC-inspired 的多模态表示学习思路。结果显示，时序建模、音视频融合、query-token 音视频对齐和 EEG 三模态融合都对情绪识别有帮助；当前最佳是 Temporal AVE，Macro-F1 为 0.7533。但这还不是完整 ECMC 复现，因为缺少文本、认知标签和 caption 生成监督。
