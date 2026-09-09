# mini-k3

一个基于 [MiniMind](https://github.com/jingyaogong/minimind)、受
[Kimi K3](https://arxiv.org/abs/2607.24653) 启发的可训练 mini LLM 实验项目。

本项目将 K3 的三个架构思想缩放到单卡/教学场景：

- `KimiDeltaAttention`：3:1 的线性 delta attention / 全注意力混合；
- `AttentionResidualMixer`：逐 token 跨层残差选择；
- `StableLatentMoE`：低维潜空间 Top-2 专家、共享专家及稳定路由正则。

默认配置共有 **14.79M** 参数、每 token 激活约 **7.71M** 参数。实现是研究型缩放版本，
不是 Moonshot 2.8T 模型的逐位复现，也不包含其闭源训练数据和集群内核。

## 训练状态

已在 RTX 4070 Laptop 8GB 上完成第一阶段 **1,000 optimizer steps** 预训练，处理 512,000 个
token positions。训练 loss 的前 20 个日志点均值由 `8.322` 降至最后 20 个日志点的 `6.562`。
固定 probe 的 perplexity 从随机初始化的 `6715.3` 降至 `1011.8`。由于该 probe 未在训练前严格
隔离，结果只用于确认学习有效，不能作为与已完整训练 MiniMind 的公平质量对比。

本地 checkpoint：`out/mini_k3_pretrain_1000.pt`（约 178MB，已由 `.gitignore` 排除）。可复现
评估结果见 [`reports/pretrain_eval.json`](reports/pretrain_eval.json)。

## 可暂停的后续训练

后续阶段目标为累计 10,000 optimizer steps，配置位于
[`configs/continued_pretrain.json`](configs/continued_pretrain.json)。控制命令：

```bash
# 查看状态
python scripts/train_control.py status

# 安全暂停：当前 optimizer step 完成后原子保存再退出
python scripts/train_control.py pause

# 从最新 out/mini_k3_continued.pt 恢复；首次运行则从 1000-step checkpoint 开始
python scripts/train_control.py resume
```

也可用 `python scripts/train_control.py start` 启动。重复执行不会产生两个训练进程。暂停延迟通常是
完成一个 optimizer step 所需的时间。不要用任务管理器“结束任务”，除非进程失去响应；`Ctrl+C` /
`SIGTERM` 也会请求保存后退出。状态文件、日志、PID 和 checkpoint 全部位于 `out/`，不会提交 Git。

## 快速验证

```bash
python -m unittest discover -s tests -v
python benchmarks/benchmark_model.py --device cpu --batch-size 1 --seq-len 64 --steps 3
python trainer/train_minik3.py --config tests/tiny_config.json \
  --data tests/tiny_data.jsonl --tokenizer byte --seq-len 32 \
  --batch-size 2 --grad-accum 1 --epochs 1 --device cpu \
  --output out/smoke_test.pt
```

真实预训练沿用 MiniMind 的 `{"text": "..."}` JSONL 数据和本仓库 tokenizer：

```bash
pip install -r requirements.txt
python trainer/train_minik3.py \
  --config configs/mini_k3_15m.json \
  --data dataset/pretrain_t2t_mini.jsonl --tokenizer model \
  --seq-len 512 --batch-size 8 --grad-accum 4 --epochs 1
```

详细设计、训练建议、实验结果和局限见 [技术报告](reports/technical_report.md)。原始 MiniMind
说明保存在 `reports/MINIMIND_UPSTREAM_README.md`。

## 目录

```text
model/model_minik3.py          Mini-K3 模型
trainer/train_minik3.py        预训练入口
scripts/train_control.py       启动/暂停/恢复/状态控制
configs/mini_k3_15m.json       默认配置
configs/continued_pretrain.json 后续训练计划
benchmarks/benchmark_model.py  消融性能基准
tests/test_model.py            正确性测试
reports/technical_report.md    技术报告
```

MiniMind 原项目采用 Apache-2.0 许可证；本仓库保留其 `LICENSE` 和来源说明。
