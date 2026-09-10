# mini-k3 技术报告

## 摘要

mini-k3 在 MiniMind 的小型 decoder-only LLM 基础上，引入 Kimi K3 的三类核心思想：Kimi
Delta Attention（KDA）、Attention Residuals（AttnRes）和 Stable LatentMoE。目标是在有限参数、
单卡可训练的条件下探索更长上下文、更好的深层信息流和更高的参数/计算解耦。默认模型总参数
14,785,408，每 token 估算激活参数 7,707,520（52.13%）。

需要强调：本报告的“实现”是论文思想的缩放适配，不是 K3 的 2.8T 官方架构逐位复刻。当前已完成
单元测试、端到端训练冒烟、CPU 消融性能测试，以及在 MiniMind 官方 1.24GB 数据上的第一阶段
累计 10,000-step 预训练。该训练预算仍明显小于完整 MiniMind，不能声称语言能力已经超过训练完毕的 MiniMind。
本文严格区分实测结果与预期收益。

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

### 3.2 已执行的第一阶段预训练

| 项目 | 实际设置 |
|---|---|
| GPU | NVIDIA RTX 4070 Laptop 8GB |
| 数据 | `pretrain_t2t_mini.jsonl`，1,270,238 条，1.24GB |
| optimizer steps | 1,000 |
| batch / gradient accumulation | 2 / 2 |
| sequence length | 128 |
| token positions | 512,000 |
| optimizer | AdamW, betas=(0.9, 0.95), weight decay=0.1 |
| LR | peak 3e-4，100-step warmup，cosine decay |
| precision | BF16 autocast |
| checkpoint interval | 100 steps |
| wall time | 约 46 分 29 秒（含首次数据索引和保存） |

训练数据采用文件偏移流式索引，而不是一次性载入 1.24GB 文本。最终 checkpoint 的
`updates` 字段已校验为 1000，文件约 178MB，保存在本地 `out/`，不提交 GitHub。

### 3.3 建议完整预训练方案（尚未执行）

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

### 3.4 后续训练与暂停恢复（已完成）

配置 `configs/continued_pretrain.json` 将累计目标设为 10,000 optimizer steps（5.12M token
positions），从 1,000-step checkpoint 延续并已完成。续训峰值学习率为 1e-4，保持 batch=2、
梯度累积=2、sequence=128，并每 100 steps 保存。续训主体耗时约 6 小时 30 分。

训练 checkpoint 新增 `epoch`、`micro_step`、Python/PyTorch RNG state、完整训练参数和 `save_reason`。
保存先写入同目录 `.tmp` 文件，再使用原子替换，避免中断时留下半写入权重。每个 optimizer step 后检查
pause marker；发现后保存全部状态并正常退出。恢复时重建同一 epoch 的确定性样本排列，跳过已经消费的
样本，从下一个 micro-batch 继续。旧版 1,000-step checkpoint 没有数据位置字段，因此第一次延续从新
排列开头开始；后续所有暂停均能记录精确位置。

控制器 `scripts/train_control.py` 提供 `start`、`pause`、`resume` 和 `status`，并通过 PID 文件防止重复
启动。暂停和 checkpoint 文件都在 Git 忽略的 `out/` 目录。

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

### 4.3 第一阶段预训练结果（实测）

训练日志共记录 201 个采样点（step 1，之后每 5 steps）：

| 指标 | 结果 |
|---|---:|
| step 1 loss | 8.8946 |
| 前 20 个日志点平均 loss | 8.3225 |
| 最后 20 个日志点平均 loss | 6.5616 |
| 最低单批 loss | 5.7140 |
| step 1000 loss | 6.4493 |
| 平均 loss 降幅（首尾窗口） | 21.16% |

另在数据文件尾部固定 16 条样本、2,032 个非 padding target tokens 上做 probe：

| 模型 | cross-entropy | perplexity |
|---|---:|---:|
| 同架构随机初始化 | 8.8121 | 6715.3 |
| 1,000-step checkpoint | 6.9195 | 1011.8 |

perplexity 相对随机初始化下降约 84.9%，说明训练确实学到了 token 分布。需要注意，初次训练对完整文件
进行了随机采样，没有事先排除这 16 条数据；虽然 4,000 个已见样本只占 127 万条数据约 0.315%，该
probe 仍不能称为严格无污染验证集。原始结果保存于 `reports/pretrain_eval.json`。

### 4.4 10,000-step 续训结果（实测）

续训日志记录 1,800 个 loss 采样点，最终以 `event=complete, updates=10000` 正常退出，错误日志为空。

| 指标 | 结果 |
|---|---:|
| 累计 optimizer steps | 10,000 |
| 累计 token positions | 5,120,000 |
| step 1005 起始 loss | 6.0480 |
| 续训前 20 日志点平均 loss | 6.5420 |
| 续训后 20 日志点平均 loss | 5.3797 |
| 续训最低单批 loss | 4.0753 |
| step 10000 loss | 5.4108 |
| 首尾窗口平均 loss 降幅 | 17.77% |

相同固定 16 条、2,032 target-token probe 的阶段对比：

| 模型阶段 | cross-entropy | perplexity |
|---|---:|---:|
| 随机初始化 | 8.8121 | 6715.3 |
| step 1000 | 6.9195 | 1011.8 |
| step 10000 | 6.1233 | 456.4 |

step 10000 相对 step 1000 的 probe perplexity 再下降约 54.9%，相对随机初始化下降约 93.2%。同样，
该 probe 并非严格预留验证集，不能替代公平的 MiniMind 基线和下游任务评测。机器可读结果位于
`reports/continued_pretrain_eval.json` 和 `reports/continued_pretrain_summary.json`。

### 4.5 架构消融性能（实测）

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

### 4.6 尚缺的质量评测

当前累计训练 5.12M token positions，仍远低于建议的 0.5B–5B token budget，故没有可信的严格
validation perplexity、C-Eval、CMMLU、GSM8K 或 HumanEval 分数。继续完整训练后，建议至少比较：

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

# 第一阶段 checkpoint probe
python benchmarks/evaluate_checkpoint.py \
  --checkpoint out/mini_k3_pretrain_1000.pt \
  --data dataset/pretrain_t2t_mini.jsonl --samples 16 --device cuda
```

## 7. 文件清单

- `model/model_minik3.py`：架构实现。
- `trainer/train_minik3.py`：训练器。
- `configs/mini_k3_15m.json`：默认实验配置。
- `tests/test_model.py`：正确性测试。
- `benchmarks/benchmark_model.py`：参数/吞吐消融。
- `benchmarks/evaluate_checkpoint.py`：checkpoint loss/perplexity probe。
- `reports/benchmark_results.json`：本机原始测量。
- `reports/pretrain_eval.json`：第一阶段预训练 probe 结果。
- `reports/continued_pretrain_eval.json`：10,000-step checkpoint probe 结果。
- `reports/continued_pretrain_summary.json`：续训设置和曲线摘要。
- `reports/k3_tech_report.pdf`：论文归档。
