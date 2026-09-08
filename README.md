# mini-k3

一个基于 [MiniMind](https://github.com/jingyaogong/minimind)、受
[Kimi K3](https://arxiv.org/abs/2607.24653) 启发的可训练 mini LLM 实验项目。

本项目将 K3 的三个架构思想缩放到单卡/教学场景：

- `KimiDeltaAttention`：3:1 的线性 delta attention / 全注意力混合；
- `AttentionResidualMixer`：逐 token 跨层残差选择；
- `StableLatentMoE`：低维潜空间 Top-2 专家、共享专家及稳定路由正则。

默认配置共有 **14.79M** 参数、每 token 激活约 **7.71M** 参数。实现是研究型缩放版本，
不是 Moonshot 2.8T 模型的逐位复现，也不包含其闭源训练数据和集群内核。

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
  --data dataset/pretrain_hq.jsonl --tokenizer model \
  --seq-len 512 --batch-size 8 --grad-accum 4 --epochs 1
```

详细设计、训练建议、实验结果和局限见 [技术报告](reports/technical_report.md)。原始 MiniMind
说明保存在 `reports/MINIMIND_UPSTREAM_README.md`。

## 目录

```text
model/model_minik3.py          Mini-K3 模型
trainer/train_minik3.py        预训练入口
configs/mini_k3_15m.json       默认配置
benchmarks/benchmark_model.py  消融性能基准
tests/test_model.py            正确性测试
reports/technical_report.md    技术报告
```

MiniMind 原项目采用 Apache-2.0 许可证；本仓库保留其 `LICENSE` 和来源说明。
