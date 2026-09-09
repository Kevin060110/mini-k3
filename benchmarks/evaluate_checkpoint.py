"""Evaluate a Mini-K3 checkpoint against its random-initialized architecture."""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import MiniK3Config, MiniK3ForCausalLM  # noqa: E402
from trainer.train_minik3 import JsonlTokenDataset  # noqa: E402


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, token_count = 0.0, 0
    for ids, labels, mask in loader:
        ids, labels, mask = ids.to(device), labels.to(device), mask.to(device)
        logits = model(ids, attention_mask=mask)["logits"][:, :-1]
        targets = labels[:, 1:]
        loss_sum += F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                    ignore_index=-100, reduction="sum").item()
        token_count += targets.ne(-100).sum().item()
    loss = loss_sum / token_count
    return {"cross_entropy": loss, "perplexity": math.exp(loss), "tokens": token_count}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", default="model")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="reports/pretrain_eval.json")
    args = p.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    dataset = JsonlTokenDataset(args.data, tokenizer, args.seq_len)
    # Fixed for reproducibility. Since training sampled the full file, this is a
    # probe set rather than a strict uncontaminated validation set.
    indices = range(len(dataset) - args.samples, len(dataset))
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = MiniK3Config(**checkpoint["config"])

    torch.manual_seed(42)
    random_model = MiniK3ForCausalLM(cfg).to(device)
    random_result = evaluate(random_model, loader, device)
    del random_model
    trained_model = MiniK3ForCausalLM(cfg)
    trained_model.load_state_dict(checkpoint["model"])
    trained_result = evaluate(trained_model.to(device), loader, device)
    payload = {
        "checkpoint_updates": checkpoint["updates"],
        "probe_samples": args.samples,
        "sequence_length": args.seq_len,
        "random_initialization": random_result,
        "trained_checkpoint": trained_result,
        "note": "Fixed tail probe; not a strict held-out split because training sampled the full file."
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
