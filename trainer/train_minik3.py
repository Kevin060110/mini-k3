"""Single-GPU Mini-K3 pretraining entry point using MiniMind JSONL/tokenizer."""
import argparse
import json
import math
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import MiniK3Config, MiniK3ForCausalLM  # noqa: E402


class ByteTokenizer:
    """Dependency-free smoke-test tokenizer; use MiniMind's tokenizer for real runs."""
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2

    def __call__(self, text, add_special_tokens=False, truncation=True, max_length=None):
        ids = [byte + 3 for byte in text.encode("utf-8")]
        return {"input_ids": ids[:max_length] if max_length is not None else ids}


class JsonlTokenDataset(Dataset):
    def __init__(self, path, tokenizer, seq_len):
        self.texts = []
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("text"):
                    self.texts.append(str(row["text"]))
        if not self.texts:
            raise ValueError(f"no non-empty 'text' records in {path}")
        self.tokenizer, self.seq_len = tokenizer, seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        tok = self.tokenizer(self.texts[index], add_special_tokens=False,
                             truncation=True, max_length=self.seq_len - 2)["input_ids"]
        ids = [self.tokenizer.bos_token_id] + tok + [self.tokenizer.eos_token_id]
        ids += [self.tokenizer.pad_token_id] * (self.seq_len - len(ids))
        ids = torch.tensor(ids, dtype=torch.long)
        labels = ids.clone()
        labels[ids == self.tokenizer.pad_token_id] = -100
        return ids, labels, ids.ne(self.tokenizer.pad_token_id)


def cosine_lr(step, total, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/mini_k3_15m.json")
    p.add_argument("--data", required=True, help="MiniMind pretrain JSONL with a text field")
    p.add_argument("--tokenizer", default="model", help="MiniMind tokenizer path, or 'byte' for smoke tests")
    p.add_argument("--output", default="out/mini_k3_last.pt")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = MiniK3Config(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    if args.tokenizer == "byte":
        tokenizer = ByteTokenizer()
    else:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install requirements.txt or pass --tokenizer byte for a smoke test") from exc
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    dataset = JsonlTokenDataset(args.data, tokenizer, args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    device = torch.device(args.device)
    model = MiniK3ForCausalLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  betas=(0.9, 0.95), weight_decay=args.weight_decay)
    total_updates = math.ceil(len(loader) / args.grad_accum) * args.epochs
    use_amp = device.type == "cuda"
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    update = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for epoch in range(args.epochs):
        for micro_step, (ids, labels, mask) in enumerate(loader, 1):
            ids, labels, mask = ids.to(device), labels.to(device), mask.to(device)
            with amp:
                out = model(ids, attention_mask=mask, labels=labels)
                loss = out["loss"] / args.grad_accum
            loss.backward()
            if micro_step % args.grad_accum == 0 or micro_step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                lr = cosine_lr(update, total_updates, args.learning_rate, args.warmup_steps)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                if update % args.log_every == 0 or update == 1:
                    print(json.dumps({"epoch": epoch + 1, "update": update,
                                      "loss": float(out["loss"].detach()),
                                      "aux_loss": float(out["aux_loss"].detach()), "lr": lr}))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg.__dict__,
                "optimizer": optimizer.state_dict(), "updates": update}, output)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
