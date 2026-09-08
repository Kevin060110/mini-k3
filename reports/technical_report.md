# mini-k3 技术报告

## 摘要

mini-k3 在 MiniMind 的小型 decoder-only LLM 基础上，引入 Kimi K3 的三类核心思想：Kimi
Delta Attention（KDA）、Attention Residuals（AttnRes）和 Stable LatentMoE。目标是在有限参数、
单卡可训练的条件下探索更长上下文、更好的深层信息流和更高的参数/计算解耦。默认模型总参数
14,785,408，每 token 估算激活参数 7,707,520（52.13%）。

需要强调：本报告的“实现”是论文思想的缩放适配，不是 K3 的 2.8T 官方架构逐位复刻。当前已完成
单元测试、端到端训练冒烟和 CPU 消融性能测试；尚未进行大语料预训练，因此不能声称语言能力已经
超过训练完毕的 MiniMind。本文严格区分实测结果与预期收益。

## 1. 背景与设计依据

MiniMind 提供了易读的 decoder-only Transformer、GQA、RoPE、SwiGLU、Top-k MoE，以及从预训练
到偏好优化的训练代码，是良好的小模型基线。上游源码：
<https://github.com/jingyaogong/minimind>。

Kimi K3 技术报告描述了一个 2.8T 总参数、104B 激活参数、93 层的 MoE；其注意力由 69 层 KDA 和
24 层 Gated MLA 组成，并使用 AttnRes、Stable LatentMoE 与 1M 上下文。官方摘要称这些架构与训练
改进相对 K2 带来约 2.5 倍整体 scaling efficiency。来源：
<https://arxiv.org/abs/2607.24653>、<https://github.com/MoonshotAI/Kimi-K3>。

K3 的规模、视觉编码器、MXFP4/8 量化、分布式专家并行和百万 token 内核不适合直接移植到 mini
环境。本项目只选择可以独立验证且不会依赖专用集群的部分。

## 2. 模型架构

### 2.1 混合 KDA / 全注意力

默认 `attention_pattern="KKKF"`：每四层中三层为 KDA，一层为 PyTorch SDPA 全注意力。全注意力
用于精确保留与任意历史 token 的内容寻址；KDA 用固定大小状态承载历史信息。

对每个 head，参考实现按时间更新状态矩阵：

```text
e_t = v_t - S_(t-1) k_t
S_t = decay_t * S_(t-1) + beta_t * k_t e_t^T
o_t = S_t q_t
```

`q/k` 经 RMSNorm、RoPE、正值映射和归一化；`beta/decay` 从 token 表征学习。delta 误差项会先减去
当前记忆对 key 的预测，再写入新值，降低重复写入。状态大小为 `heads × head_dim²`，推理缓存不随
序列长度增长。当前代码为清晰、可测试的逐 token PyTorch 参考路径；要获得墙钟时间优势，需要替换为
chunkwise/fused KDA kernel。

### 2.2 Attention Residuals 缩放版

普通残差只连接相邻层。`AttentionResidualMixer` 保存最近 4 个层状态，并以当前 token 生成 query，
对可学习的层 key 做 softmax，逐 token 混合残差源。窗口上限避免深度增大时二次复杂度，同时让后层
可以绕过不合适的中间变换。关闭 `attn_residual` 即恢复局部残差，便于消融。

### 2.3 Stable LatentMoE 缩放版

MoE 首先将 256 维 hidden state 投影到 128 维 latent channel，再执行 8 专家 Top-2 路由，之后投影
回 hidden channel。每个 token 还经过 1 个共享专家，避免稀疏路由丢失通用能力。路由采用归一化输入、
softmax 权重归一、router z-loss 和负载均衡项。`routing_bias` 被注册为 buffer，预留给大规模训练时的
auxiliary-loss-free 在线负载校正；本参考训练器默认仍采用可微均衡项，优先保证小 batch 稳定性。

这与原 MiniMind 的 hidden-space MoE 相比，把专家的大部分计算放在 latent channel，总参数可增大而
每 token 只激活部分 routed experts。`num_parameters(active_only=True)` 明确报告估算激活参数。

### 2.4 工程改进

- 所有改进均有配置开关，可组合为 dense/full、KDA、KDA+AttnRes、完整 mini-k3 四种消融。
- 全注意力使用 SDPA；padding mask 与 causal mask 正确合并。
- loss 自动包含 MoE 路由正则；支持 ignore index `-100`。
- 提供无第三方 tokenizer 的 byte 冒烟模式，以及兼容 MiniMind tokenizer/JSONL 的真实训练模式。
- 检查配置维度、路由 Top-k 和 attention pattern，错误在初始化阶段暴露。

## 3. 默认训练设置

### 3.1 已提供的 14.8M 配置

| 项目 | 设置 |
|---|---:|
| hidden / layers | 256 / 8 |
| attention heads / KV heads | 8 / 4 |
| attention pattern | KKKF |
| dense FFN | 704 |
| latent dim | 128 |
| routed experts / active | 8 / 2 |
| shared experts | 1 |
| expert FFN | 384 |
| max context | 4096 |
| vocabulary | 6400 |
| total / active params | 14.79M / 7.71M |

### 3.2 建议正式预训练方案（尚未执行）

| 项目 | 建议值 |
|---|---|
| 数据 | MiniMind 清洗预训练 JSONL；去重并单独留出验证集 |
| token budget | 至少 0.5B，推荐 2–5B tokens |
| sequence curriculum | 512 → 2048 → 4096 |
| optimizer | AdamW, betas=(0.9, 0.95), weight decay=0.1 |
| peak LR | 3e-4，100–1000 steps warmup，cosine decay 到 0.1× |
| precision | BF16；不支持 BF16 时 FP32/FP16 + scaler |
| effective batch | 约 0.25–1M tokens/update，按显存梯度累积 |
| gradient clipping | global norm 1.0 |
| checkpoint/eval | 每 1k–5k updates 保存并计算 validation PPL |

公平比较必须让 baseline 与 mini-k3 使用相同 tokenizer、训练 tokens、batch tokens、数据顺序和优化器。
除最终 loss/PPL 外，应报告峰值显存、训练 tokens/s 和固定 prompt 的生成 tokens/s。

## 4. 实验与结果

### 4.1 正确性测试（实测）

环境：Windows，Python 3.10，PyTorch 2.6.0+cu124，CPU。命令：

```bash
python -m unittest discover -s tests -v
```

结果：5/5 通过，用时 0.258 秒。覆盖：前向/反向有限值、四种消融组合、因果前缀不变性、MoE router
梯度和激活参数计数。

### 4.2 端到端训练冒烟（实测）

tiny 2 层模型、4 条 byte-tokenized 样本、batch=2、2 个 optimizer updates、CPU。交叉熵含路由正则：

| update | total loss | auxiliary loss |
|---:|---:|---:|
| 1 | 8.7656 | 0.02387 |
| 2 | 8.7561 | 0.02389 |

checkpoint 成功写入 `out/smoke_test.pt`。该实验只证明数据加载、反向、优化和保存链路可用；样本太少，
不能解释为语言质量提升。

### 4.3 架构消融性能（实测）

CPU，batch=1，sequence=64，1 次 warmup + 3 次计时；统一 hidden=256、6 层。完整原始 JSON 位于
`reports/benchmark_results.json`。

| 变体 | 总参数 | 激活参数 | latency (ms) | tokens/s |
|---|---:|---:|---:|---:|
| dense-full baseline | 6,065,792 | 6,065,792 | 7.61 | 8,415 |
| KDA + dense | 6,414,032 | 6,414,032 | 44.40 | 1,442 |
| KDA + AttnRes + dense | 6,415,592 | 6,415,592 | 47.58 | 1,345 |
| full mini-k3 | 11,533,544 | 6,225,128 | 58.97 | 1,085 |

结论：LatentMoE 让总参数相对 dense baseline 增加 90.1%，但估算激活参数只增加 2.6%，达到了参数/
激活计算解耦。当前 mini-k3 **没有在 CPU 墙钟时间上更快**：逐 token KDA 参考循环使其慢约 7.8 倍。
因此“更快”目前只成立于长序列状态复杂度和具备 fused kernel 后的设计潜力，不能从本次结果宣称已实现。

### 4.4 尚缺的质量评测

没有下载大语料或消耗数小时/数天 GPU 预算，故没有可信的 validation perplexity、C-Eval、CMMLU、
GSM8K 或 HumanEval 分数。完成正式训练后，建议至少比较：

1. validation PPL 与达到同一 PPL 所需 tokens/FLOPs；
2. C-Eval/CMMLU（中文知识）、GSM8K（推理）、HumanEval（代码）；
3. 512/2K/4K 长度的 passkey retrieval；
4. 全注意力 baseline、+KDA、+AttnRes、+LatentMoE 的逐项消融。

## 5. 局限与下一步

1. KDA 是正确性优先的 recurrent reference kernel；首要工作是接入 Triton chunkwise kernel，并在 GPU
   上以 2K–32K 序列测量吞吐和峰值显存。
2. 训练阶段尚未实现 K3 的动态 auxiliary-loss-free expert bias 更新；可按每步专家负载的符号误差更新
   `routing_bias`，并对比当前可微均衡项。
3. 生成目前会重算上下文。KDA state cache 与 full-attention KV cache 可进一步降低逐 token 延迟。
4. 没有实现 Gated MLA、SiTU-GLU、原生视觉、量化感知训练和 K3 后训练/RL 系统；这些超出本次 mini
   文本模型范围。
5. “更强”最终必须由等数据、等预算训练后的下游评测证明。当前交付的是可训练研究原型和严谨基准框架。

## 6. 复现命令

```bash
# 单元测试
python -m unittest discover -s tests -v

# 消融基准（结果覆盖写入 reports/benchmark_results.json）
python benchmarks/benchmark_model.py --device cpu --batch-size 1 --seq-len 64 --warmup 1 --steps 3

# 无依赖训练冒烟
python trainer/train_minik3.py --config tests/tiny_config.json \
  --data tests/tiny_data.jsonl --tokenizer byte --seq-len 32 \
  --batch-size 2 --grad-accum 1 --epochs 1 --device cpu \
  --output out/smoke_test.pt
```

## 7. 文件清单

- `model/model_minik3.py`：架构实现。
- `trainer/train_minik3.py`：训练器。
- `configs/mini_k3_15m.json`：默认实验配置。
- `tests/test_model.py`：正确性测试。
- `benchmarks/benchmark_model.py`：参数/吞吐消融。
- `reports/benchmark_results.json`：本机原始测量。
- `reports/k3_tech_report.pdf`：论文归档。
