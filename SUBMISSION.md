# mini-k3 项目提交说明

## 项目信息

- 项目名称：mini-k3
- GitHub：<https://github.com/Kevin060110/mini-k3>
- 技术基础：[MiniMind](https://github.com/jingyaogong/minimind)、[Kimi K3](https://arxiv.org/abs/2607.24653)
- 最终代码提交：以 GitHub `main` 分支为准

## 完成内容

mini-k3 是一个基于 MiniMind、受 Kimi K3 架构启发的轻量级 decoder-only 语言模型。项目实现了：

1. Kimi Delta Attention 与全注意力按 `KKKF` 组合的混合注意力；
2. 逐 token 跨层 Attention Residual；
3. 低维 Top-2 Stable LatentMoE 与共享专家；
4. 流式 JSONL 数据读取、BF16 训练、周期 checkpoint；
5. 可安全暂停、恢复和查询状态的训练控制器；
6. 单元测试、架构消融、训练评估及完整技术报告。

## 最终训练结果

| 项目 | 结果 |
|---|---:|
| 模型总参数 | 14.79M |
| 每 token 估算激活参数 | 7.71M |
| optimizer steps | 10,000 |
| token positions | 5,120,000 |
| 数据集 | MiniMind `pretrain_t2t_mini.jsonl`，1,270,238 条 |
| 续训首/尾窗口平均 loss | 6.542 / 5.380 |
| 最终 loss | 5.4108 |
| 最低记录 loss | 4.0753 |
| step 1000 probe perplexity | 1011.8 |
| step 10000 probe perplexity | 456.4 |
| 单元测试 | 5/5 通过 |

固定 probe 未在首次训练前严格隔离，因此只用于确认训练确实改善了 token 分布，不能作为无污染测试集，
也不能据此宣称模型已经超过完整训练的 MiniMind。当前纯 PyTorch KDA 是正确性优先的参考实现，在 CPU
墙钟速度上慢于优化后的全注意力；真正的长序列速度收益需要 fused/chunkwise GPU kernel。

## 主要文件

- `README.md`：项目概览与复现命令；
- `reports/technical_report.md`：完整技术报告；
- `reports/continued_pretrain_summary.json`：最终训练摘要；
- `reports/continued_pretrain_eval.json`：最终 probe 结果；
- `model/model_minik3.py`：模型实现；
- `trainer/train_minik3.py`：可恢复训练器；
- `scripts/train_control.py`：训练状态与暂停/恢复控制。

最终权重约 178MB，位于本地 `out/mini_k3_continued.pt`。为避免 Git 仓库膨胀，权重和 1.24GB 训练
数据未提交 GitHub；仓库包含全部代码、配置和结果摘要，可按文档复现。

## 发给老师的参考措辞

老师您好，我已完成 mini-k3 项目并整理好代码、实验结果和技术报告。项目基于 MiniMind，引入了受
Kimi K3 启发的混合 Delta Attention、跨层 Attention Residual 和低维稀疏 MoE，并实现了可暂停/
恢复的训练流程。

模型在 RTX 4070 Laptop 8GB 上累计完成 10,000 个 optimizer steps，处理约 512 万 token positions。
续训阶段首尾窗口平均 loss 从 6.542 降至 5.380，固定 probe perplexity 从 step 1000 的 1011.8
进一步降至 step 10000 的 456.4，5 项单元测试全部通过。报告中也如实说明了训练预算、probe 未严格
隔离以及当前参考 KDA 内核的速度局限，没有将阶段性结果夸大为超过完整 MiniMind。

项目地址：<https://github.com/Kevin060110/mini-k3>

烦请老师查收并批评指正，谢谢老师！
