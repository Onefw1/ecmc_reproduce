# module 模块上下文

导航：← [项目根上下文](../CLAUDE.md) / [module](.)

更新时间：2026-06-28 10:32:19

## 模块职责

[module](.) 目录存放 ECMC 模型依赖的本地 Transformer 实现，主要为上层 [MMDAmodel.py](../MMDAmodel.py) 和 [MMDAmodel2.py](../MMDAmodel2.py) 服务。

## 文件说明

| 文件 | 作用 | 被谁使用 |
|---|---|---|
| [Qformer.py](Qformer.py) | 基于 BERT 结构改造的 Q-Former 实现，支持 `query_embeds` 和 cross-attention | [MMDAmodel.py](../MMDAmodel.py) 中 `BertConfig`、`BertLMHeadModel` |
| [modeling_llama.py](modeling_llama.py) | 本地 LLaMA causal language model 实现 | [MMDAmodel.py](../MMDAmodel.py)、[MMDAmodel2.py](../MMDAmodel2.py) 中 `LlamaForCausalLM` |

## 与论文对应关系

- 论文 Method 中的 dual-stream BridgeNet 基于 Q-Former；本模块的 [Qformer.py](Qformer.py) 提供底层 Q-Former/BERT 组件。
- 论文公式 (6)(7) 描述 query token 自注意力与跨模态 cross-attention；代码中对应 `query_embeds` 和 `encoder_hidden_states` 的传入机制。
- 论文使用 LLaMA decoder 生成 emotion–cognition captions；本模块 [modeling_llama.py](modeling_llama.py) 为 [MMDAmodel2.py](../MMDAmodel2.py) 提供 LLaMA decoder。

## 面向小白的理解

可以把本目录理解为“模型零件库”：

- Q-Former 像一组“会提问的小探针”，去音频、视频、文本特征里问：“哪些信息和情绪有关？哪些信息和认知障碍有关？”
- LLaMA 像“写报告的人”，拿到前面提炼出的关键信息后，按 prompt 写出情绪和认知描述。

## 注意事项

- 这些文件大多来自或改造自 Transformer 模型实现，体量较大，通常不优先修改。
- 如果用户问训练逻辑或业务流程，优先看上层 [MMDAmodel.py](../MMDAmodel.py)、[MMDAmodel2.py](../MMDAmodel2.py)、[train.py](../train.py)、[train2.py](../train2.py)。
- 如果用户问 Q-Former 或 LLaMA 内部细节，再进入本目录逐函数解释。
