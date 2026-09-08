"""Reproducible forward-pass and parameter benchmark for Mini-K3 ablations."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import MiniK3Config, MiniK3ForCausalLM


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(name, cfg, batch, seq_len, warmup, steps, device):
    torch.manual_seed(42)
    model = MiniK3ForCausalLM(cfg).to(device).eval()
    ids = torch.randint(0, cfg.vocab_size, (batch, seq_len), device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(ids)
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(steps):
            model(ids)
        synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "name": name,
        "total_parameters": model.num_parameters(),
        "active_parameters": model.num_parameters(active_only=True),
        "latency_ms": elapsed * 1000 / steps,
        "tokens_per_second": batch * seq_len * steps / elapsed,
        "peak_memory_mb": (torch.cuda.max_memory_allocated(device) / 2**20
                           if device.type == "cuda" else None),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--output", default="reports/benchmark_results.json")
    args = p.parse_args()
    common = dict(vocab_size=6400, hidden_size=256, num_hidden_layers=6,
                  num_attention_heads=8, num_key_value_heads=4,
                  intermediate_size=704, latent_size=128,
                  moe_intermediate_size=384, num_experts=8,
                  num_experts_per_tok=2)
    variants = {
        "dense-full-baseline": MiniK3Config(**common, attention_pattern="F", attn_residual=False, use_moe=False),
        "kda+dense": MiniK3Config(**common, attention_pattern="KKKF", attn_residual=False, use_moe=False),
        "kda+attnres+dense": MiniK3Config(**common, attention_pattern="KKKF", attn_residual=True, use_moe=False),
        "mini-k3": MiniK3Config(**common, attention_pattern="KKKF", attn_residual=True, use_moe=True),
    }
    device = torch.device(args.device)
    metadata = {"torch": torch.__version__, "device": str(device), "batch_size": args.batch_size,
                "seq_len": args.seq_len, "steps": args.steps}
    results = [measure(n, c, args.batch_size, args.seq_len, args.warmup, args.steps, device)
               for n, c in variants.items()]
    payload = {"metadata": metadata, "results": results}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
