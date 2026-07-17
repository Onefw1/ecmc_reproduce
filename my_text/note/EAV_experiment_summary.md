# EAV 多模态情绪识别实验阶段总结

## 1. 当前实验目标

当前目标不是完整复现 ECMC 原论文，而是在 EAV 数据集上验证 ECMC-inspired 思路是否有效：

- 多模态情绪识别：audio / video / audio+video
- 保留时序信息，而不是直接平均全部时间片
- 使用 attention pooling 学习关键时间片
- 尝试 supervised contrastive loss（SCL）约束情绪表示空间

更准确的任务名称可以写作：

> 基于 ECMC 思想的 EAV 多模态情绪识别实验。

## 2. 术语与英文简写说明

| 简写 / 术语           | 英文全称                                    | 中文名称 / 含义                                                                                                                               |
| --------------------- | ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `EAV`               | EEG-Audio-Visual dataset                    | 脑电-音频-视觉多模态情绪数据集，当前使用的是其预提取特征版本。                                                                                |
| `ECMC`              | Emotion-Cognition Multimodal Captioning     | 情绪-认知多模态描述生成方法 / 项目，原论文面向心理健康理解。                                                                                  |
| `MMDA`              | Multimodal Mental Disorder Analysis dataset | ECMC 原论文使用的多模态心理障碍分析数据集。                                                                                                   |
| `AV`                | Audio-Video                                 | 音频-视频双模态。本文中 `Temporal AV`、`QueryToken AV` 都表示使用 audio 和 video 两个模态。                                               |
| `EEG`               | Electroencephalography                      | 脑电信号。EAV 数据中包含 EEG 特征，当前已完成 EEG 单模态和 audio-video-EEG 三模态实验。                                                       |
| `ASR`               | Automatic Speech Recognition                | 自动语音识别，即把音频转成文本。当前只有 `.npy` 特征，暂时不能做 ASR。                                                                      |
| `MLP`               | Multilayer Perceptron                       | 多层感知机，用作简单分类器 baseline。                                                                                                         |
| `CE`                | Cross Entropy                               | 交叉熵分类损失，五分类情绪识别的基础监督损失。                                                                                                |
| `SCL`               | Supervised Contrastive Loss                 | 监督式对比损失，利用标签让同类样本表示更接近、异类样本表示更远。                                                                              |
| `label-SCL`         | Label-based Supervised Contrastive Loss     | 基于类别标签的监督式对比损失。本文中主要使用 `labels.npy` 的五类情绪标签构造正负样本。                                                      |
| `AVCL`              | Audio-Video Contrastive Loss                | 音频-视频对比损失。当前实现是 instance-level AVCL，即同一样本的 audio/video 表示靠近，不同样本的 audio/video 表示远离。                       |
| `QueryToken`        | Learnable Query Token                       | 可学习查询 token，用来从 audio/video 时序特征中抽取任务相关表示。                                                                             |
| `Q-Former`          | Querying Transformer                        | 查询式 Transformer，ECMC 原代码基于 BERT 改造 Q-Former；当前 EAV notebook 是轻量 QueryToken cross-attention 实现，不是直接复用原版 Q-Former。 |
| `Attention Pooling` | Attention Pooling                           | 注意力池化，用模型学习哪些时间片更重要，再聚合成整体表示。                                                                                    |
| `Temporal`          | Temporal Modeling                           | 时序建模，指保留 audio/video 的时间片维度，而不是直接对时间维度求平均。                                                                       |
| `Macro-F1`          | Macro-averaged F1-score                     | 宏平均 F1。先分别计算每个类别的 F1，再取平均，更能反映各类别是否均衡。                                                                        |
| `Accuracy`          | Accuracy                                    | 准确率，即整体预测正确的比例。                                                                                                                |
| `Precision`         | Precision                                   | 精确率，预测为某类的样本中有多少是真的该类。                                                                                                  |
| `Recall`            | Recall                                      | 召回率，某个真实类别的样本中有多少被正确找回。                                                                                                |
| `Best Epoch`        | Best Epoch                                  | 指验证 / 测试 Macro-F1 最高的训练轮次，不一定是最后一个 epoch。                                                                               |
| `q=16 / q=32`       | Number of query tokens                      | query token 数量。`q=16` 表示每个模态使用 16 个可学习 query token；`q=32` 表示每个模态使用 32 个。                                        |

## 3. 数据说明

当前使用的是 EAV 的预提取特征版本，不是原始 `.wav` / `.mp4`。

每个 subject 下包含：

| 文件           |          shape 示例 | 含义                                                                  |
| -------------- | ------------------: | --------------------------------------------------------------------- |
| `audio.npy`  | `(400, 501, 256)` | 每个 subject 400 条音频样本，每条样本 501 个时间片，每个时间片 256 维 |
| `video.npy`  |  `(400, 25, 512)` | 每个 subject 400 条视频样本，每条样本 25 个时间片，每个时间片 512 维  |
| `eeg.npy`    |  `(400, 500, 30)` | EEG 特征，每条样本 500 个时间点，每个时间点 30 维                     |
| `labels.npy` |          `(400,)` | 五分类情绪标签                                                        |
| `split.npy`  |          `(400,)` | 训练/测试划分                                                         |

数据规模：

- subjects：42
- 每个 subject：400 条样本
- 总样本数：16800
- 训练集：11760
- 测试集：5040
- 测试集每类 support：1008

五类情绪为：

```text
Neutral
Sadness
Anger
Happiness
Calmness
```

## 4. 指标说明

实验主要关注两个指标：

- `Accuracy`：整体预测正确比例。
- `Macro-F1`：分别计算五个类别的 F1-score 后取平均，更能反映每个类别是否都学得较均衡。

由于测试集每类样本数均为 1008，类别分布均衡，因此 Accuracy 和 Macro-F1 都有参考价值。但后续更建议以 Macro-F1 作为主要比较指标。

## 5. 已完成实验结果

| 实验                      | 模型简述                                                                      | Best Epoch | Accuracy | Macro-F1 |
| ------------------------- | ----------------------------------------------------------------------------- | ---------: | -------: | -------: |
| MLP Audio                 | audio mean pooling + MLP                                                      |         13 |   0.4165 |   0.4207 |
| MLP Video                 | video mean pooling + MLP                                                      |          9 |   0.4413 |   0.4359 |
| MLP AV                    | audio/video mean pooling + concat + MLP                                       |         17 |   0.5726 |   0.5734 |
| Temporal Audio            | audio temporal projection + attention pooling                                 |         29 |   0.5752 |   0.5777 |
| Temporal Video            | video temporal projection + attention pooling                                 |         29 |   0.5905 |   0.5864 |
| Temporal AV               | audio/video temporal projection + attention pooling + fusion                  |         28 |   0.7355 |   0.7368 |
| Temporal AV + SCL         | Temporal AV + supervised contrastive loss, weight=0.1                         |         28 |   0.7288 |   0.7309 |
| Temporal AV + SCL         | Temporal AV + supervised contrastive loss, weight=0.01                        |         28 |   0.7268 |   0.7253 |
| QueryToken AV             | query-token cross-attention + CE, q=16                                        |         14 |   0.7308 |   0.7295 |
| QueryToken AV             | query-token cross-attention + CE, q=32                                        |         26 |   0.7361 |   0.7356 |
| QueryToken AV + AVCL      | query-token cross-attention + audio-video contrastive loss, q=16, weight=0.05 |         23 |   0.7486 |   0.7470 |
| QueryToken AV + AVCL      | query-token cross-attention + audio-video contrastive loss, q=32, weight=0.05 |         19 |   0.7405 |   0.7417 |
| QueryToken AV + label-SCL | query-token cross-attention + CE + label-SCL, q=16, weight=0.05               |         14 |   0.7234 |   0.7234 |
| Temporal EEG              | eeg temporal projection + attention pooling                                  |         27 |   0.4038 |   0.4047 |
| Temporal AVE              | audio/video/eeg temporal projection + attention pooling + fusion             |         23 |   0.7524 |   0.7533 |
| QueryToken AVE            | audio/video/eeg query-token cross-attention + CE, q=16                       |         29 |   0.7474 |   0.7463 |

## 6. 分类表现摘要

### 6.1 MLP Audio

```text
Accuracy  = 0.4165
Macro-F1  = 0.4207
```

单独使用音频均值特征时，效果明显高于随机猜测，但整体较弱。Anger 表现最好，F1 为 0.59；Neutral 和 Calmness 较弱。

### 6.2 MLP Video

```text
Accuracy  = 0.4413
Macro-F1  = 0.4359
```

单独使用视频均值特征略强于音频均值特征。Happiness 表现最好，F1 为 0.55；Sadness 较弱。

### 6.3 MLP AV

```text
Accuracy  = 0.5726
Macro-F1  = 0.5734
```

简单 audio+video 融合显著优于任一单模态，说明 EAV 上 audio 和 video 存在互补信息。

### 6.4 Temporal Audio

```text
Accuracy  = 0.5752
Macro-F1  = 0.5777
```

相比 MLP Audio：

```text
Macro-F1: 0.4207 -> 0.5777
提升: +0.1570
```

说明音频的时序维度非常有用，直接 mean pooling 会损失较多信息。Temporal Audio 中 Anger 表现最好，F1 为 0.74。

### 6.5 Temporal Video

```text
Accuracy  = 0.5905
Macro-F1  = 0.5864
```

相比 MLP Video：

```text
Macro-F1: 0.4359 -> 0.5864
提升: +0.1505
```

说明视频的时序信息同样重要。Temporal Video 中 Happiness 表现最好，F1 为 0.70。

### 6.6 Temporal AV

```text
Accuracy  = 0.7355
Macro-F1  = 0.7368
```

相比 MLP AV：

```text
Macro-F1: 0.5734 -> 0.7368
提升: +0.1634
```

相比 Temporal Audio：

```text
Macro-F1: 0.5777 -> 0.7368
提升: +0.1591
```

相比 Temporal Video：

```text
Macro-F1: 0.5864 -> 0.7368
提升: +0.1504
```

这是此前最强的 temporal attention baseline。结果说明：

- 保留时序信息有效。
- attention pooling 比简单 mean pooling 更适合当前特征。
- audio 和 video 的融合带来明显增益，不只是单模态增强。

各类别表现：

| 类别      | Precision | Recall |   F1 |
| --------- | --------: | -----: | ---: |
| Neutral   |      0.64 |   0.75 | 0.69 |
| Sadness   |      0.78 |   0.65 | 0.71 |
| Anger     |      0.89 |   0.80 | 0.84 |
| Happiness |      0.76 |   0.81 | 0.79 |
| Calmness  |      0.64 |   0.66 | 0.65 |

Anger 和 Happiness 识别较好，Calmness 相对较弱。

### 6.7 Temporal AV + SCL

```text
contrastive_weight = 0.1
Accuracy  = 0.7288
Macro-F1  = 0.7309
```

相比 Temporal AV：

```text
Macro-F1: 0.7368 -> 0.7309
变化: -0.0059
```

当前 SCL 配置没有带来进一步提升。当前设置为：

```text
contrastive_weight = 0.1
temperature = 0.07
batch_size = 64
```

这不能说明 SCL 一定无效，只能说明当前超参数下，SCL 没有超过纯 CE 分类训练。可能原因：

- `contrastive_weight=0.1` 对当前任务偏强。
- batch size 较小，batch 内正负样本对数量有限。
- 当前 attention-fusion 表示已经足够适合分类，额外对比约束未带来增益。

继续将权重调小到 `0.01` 后，结果为：

```text
contrastive_weight = 0.01
Accuracy  = 0.7268
Macro-F1  = 0.7253
```

相比 Temporal AV：

```text
Macro-F1: 0.7368 -> 0.7253
变化: -0.0115
```

相比 `contrastive_weight=0.1`：

```text
Macro-F1: 0.7309 -> 0.7253
变化: -0.0056
```

`contrastive_weight=0.01` 的各类别表现：

| 类别      | Precision | Recall |   F1 |
| --------- | --------: | -----: | ---: |
| Neutral   |      0.67 |   0.65 | 0.66 |
| Sadness   |      0.72 |   0.73 | 0.72 |
| Anger     |      0.87 |   0.79 | 0.83 |
| Happiness |      0.71 |   0.86 | 0.78 |
| Calmness  |      0.68 |   0.59 | 0.63 |

与纯 Temporal AV 相比，`weight=0.01` 只有 Sadness 的 F1 略高，其余类别均持平或下降。因此，当前两个 label-SCL 权重实验都没有超过纯 CE 训练。

### 6.8 QueryToken AV

QueryToken AV 使用可学习 query tokens 从 audio/video 时序特征中抽取模态表示。当前实验设置为：

```text
num_query = 16
num_layers = 2
num_heads = 4
hidden_dim = 256
dropout = 0.1
```

不加 audio-video contrastive loss，仅使用 CE 分类损失时，`q=16` 的结果为：

```text
QueryToken AV + CE
Best Epoch = 14
Accuracy   = 0.7308
Macro-F1   = 0.7295
```

相比 Temporal AV：

```text
Macro-F1: 0.7368 -> 0.7295
变化: -0.0073
```

说明 query-token 结构本身是可行的，但在当前 `q=16`、纯 CE 训练设置下，尚未超过更简单的 Temporal AV baseline。

将 query token 数量增加到 `q=32` 后，纯 CE 结果为：

```text
QueryToken AV + CE, q=32
Best Epoch = 26
Accuracy   = 0.7361
Macro-F1   = 0.7356
```

相比 `q=16`：

```text
Macro-F1: 0.7295 -> 0.7356
提升: +0.0060
```

相比 Temporal AV：

```text
Macro-F1: 0.7368 -> 0.7356
变化: -0.0012
```

这说明 query token 数量对当前模型有明显影响。增加到 ECMC 原代码使用的 `32` 个 query token 后，QueryToken AV + CE 基本追平 Temporal AV baseline，但仍未超过当时最强的 QueryToken AV + AVCL。

`q=32` 纯 CE 的各类别表现：

| 类别      | Precision | Recall |   F1 |
| --------- | --------: | -----: | ---: |
| Neutral   |      0.70 |   0.65 | 0.67 |
| Sadness   |      0.76 |   0.72 | 0.74 |
| Anger     |      0.80 |   0.85 | 0.82 |
| Happiness |      0.79 |   0.79 | 0.79 |
| Calmness  |      0.64 |   0.67 | 0.65 |

加入 audio-video contrastive loss 后，结果为：

```text
QueryToken AV + AVCL
contrastive_weight = 0.05
temperature = 0.07
Best Epoch = 23
Accuracy   = 0.7486
Macro-F1   = 0.7470
```

相比 Temporal AV：

```text
Macro-F1: 0.7368 -> 0.7470
提升: +0.0102
```

相比 QueryToken AV + CE：

```text
Macro-F1: 0.7295 -> 0.7470
提升: +0.0175
```

QueryToken AV + AVCL 的各类别表现：

| 类别      | Precision | Recall |   F1 |
| --------- | --------: | -----: | ---: |
| Neutral   |      0.67 |   0.76 | 0.71 |
| Sadness   |      0.77 |   0.71 | 0.74 |
| Anger     |      0.81 |   0.86 | 0.84 |
| Happiness |      0.78 |   0.80 | 0.79 |
| Calmness  |      0.71 |   0.61 | 0.65 |

相比 QueryToken AV + CE，AVCL 主要提升了 Neutral、Sadness 和 Calmness 等相对更容易混淆的类别。该结果说明，在 query-token 结构下，audio-video 跨模态对齐比简单 label-SCL 更有效。

继续将 query token 数量增加到 `q=32` 并加入 AVCL 后，结果为：

```text
QueryToken AV + AVCL, q=32
contrastive_weight = 0.05
temperature = 0.07
Best Epoch = 19
Accuracy   = 0.7405
Macro-F1   = 0.7417
```

相比 `q=32 + CE`：

```text
Macro-F1: 0.7356 -> 0.7417
提升: +0.0061
```

相比 `q=16 + AVCL`：

```text
Macro-F1: 0.7470 -> 0.7417
变化: -0.0053
```

`q=32 + AVCL` 的各类别表现：

| 类别      | Precision | Recall |   F1 |
| --------- | --------: | -----: | ---: |
| Neutral   |      0.68 |   0.69 | 0.68 |
| Sadness   |      0.80 |   0.72 | 0.76 |
| Anger     |      0.84 |   0.83 | 0.83 |
| Happiness |      0.79 |   0.77 | 0.78 |
| Calmness  |      0.62 |   0.68 | 0.65 |

该结果说明：`q=32` 在纯 CE 下强于 `q=16`，但加入 AVCL 后没有超过 `q=16 + AVCL`。因此，在 EAV 预提取特征和五分类任务上，更多 query token 不一定带来更好的跨模态对齐收益。该阶段最强模型为 `QueryToken AV + AVCL, q=16, weight=0.05`。

进一步测试 `QueryToken AV + CE + label-SCL, q=16` 后，结果为：

```text
QueryToken AV + CE + label-SCL, q=16
scl_weight = 0.05
scl_temperature = 0.1
Best Epoch = 14
Accuracy   = 0.7234
Macro-F1   = 0.7234
```

相比 `q=16 + CE`：

```text
Macro-F1: 0.7295 -> 0.7234
变化: -0.0061
```

相比 `q=16 + AVCL`：

```text
Macro-F1: 0.7470 -> 0.7234
变化: -0.0236
```

`q=16 + label-SCL` 的各类别表现：

| 类别      | Precision | Recall |   F1 |
| --------- | --------: | -----: | ---: |
| Neutral   |      0.67 |   0.66 | 0.66 |
| Sadness   |      0.69 |   0.77 | 0.72 |
| Anger     |      0.86 |   0.80 | 0.83 |
| Happiness |      0.76 |   0.78 | 0.77 |
| Calmness  |      0.64 |   0.61 | 0.63 |

该结果说明，在当前实现和超参数下，query-token 后加入 label-SCL 没有提升 EAV 五分类情绪识别效果，反而低于纯 CE 和 AVCL。当前最有效的对比学习形式仍是 `QueryToken AV + AVCL, q=16, weight=0.05`。

### 6.9 Temporal EEG / Temporal AVE / QueryToken AVE

单独使用 EEG 时，结果为：

```text
Temporal EEG
Best Epoch = 27
Accuracy   = 0.4038
Macro-F1   = 0.4047
```

Temporal EEG 的各类别表现：

| 类别      | Precision | Recall |   F1 |
| --------- | --------: | -----: | ---: |
| Neutral   |      0.41 |   0.43 | 0.42 |
| Sadness   |      0.31 |   0.33 | 0.32 |
| Anger     |      0.46 |   0.39 | 0.42 |
| Happiness |      0.55 |   0.55 | 0.55 |
| Calmness  |      0.32 |   0.32 | 0.32 |

EEG 单模态明显高于随机五分类的 `0.20`，说明 EEG 中包含一定情绪信息；但它显著弱于 audio/video，因此不适合作为当前主模型。

加入 EEG 后的 Temporal AVE 结果为：

```text
Temporal AVE
Best Epoch = 23
Accuracy   = 0.7524
Macro-F1   = 0.7533
```

相比 Temporal AV：

```text
Macro-F1: 0.7368 -> 0.7533
提升: +0.0165
```

相比此前最佳 QueryToken AV + AVCL：

```text
Macro-F1: 0.7470 -> 0.7533
提升: +0.0063
```

Temporal AVE 的各类别表现：

| 类别      | Precision | Recall |   F1 |
| --------- | --------: | -----: | ---: |
| Neutral   |      0.66 |   0.76 | 0.71 |
| Sadness   |      0.85 |   0.62 | 0.72 |
| Anger     |      0.85 |   0.87 | 0.86 |
| Happiness |      0.83 |   0.80 | 0.82 |
| Calmness  |      0.62 |   0.71 | 0.67 |

该结果说明：虽然 EEG 单模态较弱，但作为第三模态加入 audio/video 后带来了明确增益。当前最佳模型更新为 `Temporal AVE`。

QueryToken AVE 的结果为：

```text
QueryToken AVE, q=16, CE
Best Epoch = 29
Accuracy   = 0.7474
Macro-F1   = 0.7463
```

相比 Temporal AVE：

```text
Macro-F1: 0.7533 -> 0.7463
变化: -0.0070
```

QueryToken AVE 的各类别表现：

| 类别      | Precision | Recall |   F1 |
| --------- | --------: | -----: | ---: |
| Neutral   |      0.71 |   0.69 | 0.70 |
| Sadness   |      0.74 |   0.72 | 0.73 |
| Anger     |      0.85 |   0.83 | 0.84 |
| Happiness |      0.75 |   0.85 | 0.80 |
| Calmness  |      0.68 |   0.64 | 0.66 |

QueryToken AVE 接近 QueryToken AV + AVCL，但没有超过 Temporal AVE。说明当前三模态设置下，简单的 temporal attention fusion 比 query-token 三模态 CE 更稳。

## 7. 当前核心结论

### 7.1 时序建模是有效的

MLP 版本直接对时间维度求平均：

```text
audio: (501,256) -> (256)
video: (25,512) -> (512)
```

Temporal 版本保留时间维度：

```text
audio: (501,256) -> projection -> attention pooling
video: (25,512) -> projection -> attention pooling
```

结果显示：

| 对比                        | Macro-F1 提升 |
| --------------------------- | ------------: |
| MLP Audio -> Temporal Audio |       +0.1570 |
| MLP Video -> Temporal Video |       +0.1505 |
| MLP AV -> Temporal AV       |       +0.1634 |

因此可以确认：EAV 的预提取 audio/video 特征中，时间片顺序和局部片段信息对情绪识别有明显价值。

### 7.2 多模态融合是有效的

Temporal AV 明显强于 Temporal Audio 和 Temporal Video：

| 对比                          | Macro-F1 提升 |
| ----------------------------- | ------------: |
| Temporal Audio -> Temporal AV |       +0.1591 |
| Temporal Video -> Temporal AV |       +0.1504 |

因此可以确认：audio 与 video 在 EAV 情绪识别中存在互补信息。

### 7.3 当前 SCL 没有进一步提升

Temporal AV + SCL 在已测试的两个权重下均低于 Temporal AV：

```text
weight=0.1:  0.7309 < 0.7368
weight=0.01: 0.7253 < 0.7368
weight=0.05: 0.7220
```

阶段性结论应写为：

> 在当前超参数设置下，supervised contrastive loss 未进一步提升 Temporal AV 表现，需要继续调参验证。

不要写成：

> 对比学习无效。

因为 label-SCL 只是一种较简单的类别级对比学习，不能代表所有对比学习形式。后续 query-token + audio-video contrastive loss 的结果已经显示，跨模态对齐形式的对比学习可以带来提升；而 query-token + label-SCL 的结果仍未带来收益。

### 7.4 EEG 作为第三模态带来了额外增益

当前最佳结果为：

```text
Temporal AVE
Best Epoch = 23
Accuracy   = 0.7524
Macro-F1   = 0.7533
```

该结果超过 Temporal AV 的 `0.7368 Macro-F1`，也超过此前最佳 QueryToken AV + AVCL 的 `0.7470 Macro-F1`。因此可以确认：虽然 EEG 单模态较弱，但加入 audio/video 后对 EAV 情绪识别有补充价值。

当前结果排序为：

| 模型 | Accuracy | Macro-F1 |
| ---- | -------: | -------: |
| Temporal AVE |   0.7524 |   0.7533 |
| QueryToken AV + AVCL, q=16 |   0.7486 |   0.7470 |
| QueryToken AVE, q=16, CE |   0.7474 |   0.7463 |
| Temporal AV |   0.7355 |   0.7368 |
| Temporal EEG |   0.4038 |   0.4047 |

## 8. 和 ECMC 思路的关系

当前实验不是 ECMC 完整复现，而是 ECMC-inspired adaptation。

ECMC 原论文面向 MMDA 临床访谈，包含：

- audio / video / text 三模态
- 问答轮次切分
- emotion caption
- cognition caption
- cognition label
- LLaMA 生成阶段

当前 EAV 数据只有预提取特征和五类情绪标签，没有：

- 原始访谈文本
- ASR transcript
- cognition label
- cognition caption
- emotion caption
- 原始 `.wav` / `.mp4`

因此当前合理迁移的是 ECMC 第一阶段思想：

```text
多模态编码 + 时序表示学习 + 情绪监督 + 对比学习尝试
```

当前结果支持：

> ECMC-inspired 的时序多模态表示学习在 EAV 情绪识别任务上有效。

进一步地，QueryToken AV + AVCL 的结果支持：

> 更接近 ECMC 的 query-token 跨模态对齐结构在 EAV 上优于普通 temporal attention fusion。

加入 EEG 后的结果进一步支持：

> EEG 虽然单模态较弱，但作为第三模态加入 audio/video 后可以带来额外增益，说明 EAV 的脑电特征对情绪识别具有补充信息。

但暂不支持：

> 完整 ECMC emotion-cognition captioning 已复现。

## 9. 下一步建议

### 9.1 当前主结果应收敛到 Temporal AVE

加入 EEG 后，Temporal AVE 已经超过此前所有结果：

```text
Temporal AVE: Macro-F1 = 0.7533
QueryToken AV + AVCL, q=16: Macro-F1 = 0.7470
Temporal AV: Macro-F1 = 0.7368
```

因此当前阶段主结果应更新为：

```text
Temporal AVE
Accuracy = 0.7524
Macro-F1 = 0.7533
```

这说明 EEG 虽然单模态较弱，但作为第三模态加入 audio/video 后有补充价值。后续不建议继续围绕 label-SCL 大范围调参，也不建议继续盲目增加 query token 数量。

### 9.2 后续最有价值的补充实验

如果继续扩展，建议只补少量有解释价值的实验：

| 优先级 | 实验 | 目的 |
| -----: | ---- | ---- |
| 1 | QueryToken AVE + AVCL | 验证 query-token 三模态模型加入跨模态对齐后，是否能超过 Temporal AVE |
| 2 | Temporal AVE 多随机种子复跑 | 验证当前 `0.7533` 是否稳定 |
| 3 | EEG 标准化方式消融 | 比较 per-sample z-score 与 train-set mean/std 标准化 |
| 4 | QueryToken AVE, q=32 | 验证三模态 query token 数量是否仍存在容量影响 |

其中最自然的下一组是：

```text
QueryToken AVE + AVCL
```

因为当前已知：

```text
Temporal AVE > Temporal AV
QueryToken AV + AVCL > QueryToken AV
```

所以需要验证：

```text
QueryToken AVE + AVCL 是否能超过 Temporal AVE？
```

### 9.3 Text / ASR 暂时不能做

当前数据是 `.npy` 特征，不能还原语音文本。ASR 需要原始音频文件：

```text
.wav / .mp3 / .flac / .mp4
```

因此除非后续拿到 EAV 原始音频或视频，否则不能做：

```text
audio -> text
text-only
audio+video+text
```

### 9.4 建议下一阶段实验顺序

1. 保留 `Temporal AVE` 作为当前主结果
2. 可选补充 `QueryToken AVE + AVCL`
3. 对 `Temporal AVE` 做多随机种子复跑，验证稳定性
4. 整理当前 notebook、结果表格和分类报告
5. 若拿到原始音频，再做 ASR text 分支
6. 将 notebook 整理成项目脚本，便于复现实验

## 10. 当前可写进报告的简短结论

> 在 EAV 数据集上，简单的 audio-video mean pooling baseline 取得 0.5734 Macro-F1。引入时序建模和 attention pooling 后，Temporal AV 的 Macro-F1 提升至 0.7368，说明 EAV 的 audio/video 时间片中包含关键情绪信息，多模态时序融合显著优于简单特征平均。QueryToken AV + AVCL 一度将 Macro-F1 提升至 0.7470，说明 query-token 表示抽取和音视频实例级对齐有效。进一步加入 EEG 后，Temporal AVE 的 Macro-F1 达到 0.7533，超过 Temporal AV 和 QueryToken AV + AVCL，说明 EEG 虽然单模态较弱，但作为第三模态对 EAV 情绪识别具有补充价值。当前最佳模型为 Temporal AVE。

## 11. 结论：沟通时的介绍思路

> 我目前是在 EAV 数据集上做一个 ECMC-inspired 的多模态情绪识别实验，目标是先验证 ECMC 中比较核心的多模态时序表示学习思想是否有效。因为当前 EAV 数据只有预提取好的 audio、video、EEG 特征和五分类情绪标签，没有原始音频、视频、文本转录、认知标签和 caption 标注，所以现阶段还不能完整复现 ECMC 的 emotion-cognition captioning 任务。

目前已经完成的工作可以这样介绍：

1. 我先整理并确认了 EAV 数据结构。当前每个 subject 有 400 条样本，包含 `audio.npy`、`video.npy`、`eeg.npy`、`labels.npy` 和 `split.npy`。目前已经完成 audio、video、EEG 相关的单模态、双模态和三模态实验。
2. 我先做了简单 baseline：对 audio/video 的时间维度直接 mean pooling，再用 MLP 分类。结果显示，单模态 audio 和 video 效果都比较一般，audio+video 融合后 Macro-F1 从单模态的 0.42 左右提升到 0.5734，说明两个模态之间确实有互补信息。
3. 然后我做了保留时序信息的 Temporal 模型：先对 audio/video 的时间片特征做投影，再用 attention pooling 聚合关键时间片，最后做分类。这个方法明显优于 mean pooling，Temporal AV 的 Accuracy 为 0.7355，Macro-F1 为 0.7368。
4. 我还尝试加入 supervised contrastive loss，也就是用标签约束同类样本表示更接近、不同类样本表示更远。目前测试了 `contrastive_weight=0.1`、`0.05`、`0.01`，都没有超过纯 CE 训练的 Temporal AV，说明当前这种简单 label-SCL 设置暂时没有带来收益。
5. 在此基础上，我进一步做了更接近 ECMC 的 QueryToken AV。纯 CE 训练时，`q=16` 的 Macro-F1 为 0.7295，`q=32` 的 Macro-F1 提升到 0.7356，基本追平 Temporal AV；在 `q=16` 上加入 audio-video contrastive loss 后，Macro-F1 提升到 0.7470。继续测试 `q=32 + AVCL` 后，Macro-F1 为 0.7417，说明更大的 query 数量没有进一步超过 `q=16 + AVCL`。测试 `q=16 + label-SCL` 后，Macro-F1 为 0.7234，说明当前简单类别级 label-SCL 暂时没有收益。
6. 最近我接入了 EEG。Temporal EEG 单模态 Macro-F1 为 0.4047，说明 EEG 单独较弱但包含一定情绪信息；Temporal AVE 的 Macro-F1 达到 0.7533，超过之前所有 audio/video 实验，成为当前最佳结果；QueryToken AVE 的 Macro-F1 为 0.7463，接近但没有超过 Temporal AVE。

目前阶段性的实验结论可以这样说：

> 当前结果说明，在 EAV 数据集上，保留 audio/video 的时序信息非常重要，attention pooling 比简单 mean pooling 更有效；同时 audio-video 融合显著优于单模态，说明多模态信息确实对情绪识别有帮助。QueryToken AV + AVCL 证明了 query-token 表示抽取和音视频实例级对齐有效；进一步加入 EEG 后，Temporal AVE 取得当前最好结果，Macro-F1 为 0.7533，说明 EEG 对 audio/video 有补充价值。当前主结果应从 QueryToken AV + AVCL 更新为 Temporal AVE。

下一步计划建议这样汇报：

1. 短期先把当前 EAV 实验整理稳定，包括 notebook、训练代码、结果表格和分类报告，保证已有结果可以复现。
2. 当前先保留 Temporal AVE 作为主结果，不再继续发散调 label-SCL。
3. 如果继续补实验，优先尝试 QueryToken AVE + AVCL，验证三模态 query-token 模型加入跨模态对齐后能否超过 Temporal AVE。
4. 同时可以对 Temporal AVE 做多随机种子复跑，验证 `0.7533` 是否稳定。
5. ASR/text 分支暂时不能做，因为当前 EAV 文件是 `.npy` 特征，不包含原始 `.wav` 或 `.mp4`。如果后续拿到原始音频或视频，再考虑做 `audio -> text`，并进一步尝试 text-only 或 audio-video-text 三模态实验。

可以给师兄的简短版本：

> 我现在还不是完整复现 ECMC，而是在 EAV 上先做 ECMC-inspired 的多模态情绪识别验证。已经完成了 audio、video、EEG 的单模态、双模态和三模态实验。结果上，Temporal AV 的 Macro-F1 为 0.7368，明显超过简单 AV mean pooling 的 0.5734；QueryToken AV + AVCL 将结果提升到 0.7470；进一步加入 EEG 后，Temporal AVE 达到 0.7533，成为当前最好结果。这说明时序建模、音视频融合、query-token 音视频对齐，以及 EEG 三模态补充信息都对 EAV 情绪识别有帮助。下一步我会保留 Temporal AVE 作为主结果，并优先考虑 QueryToken AVE + AVCL 或多随机种子复跑；ASR/text 分支需要先拿到原始音频或视频数据。
