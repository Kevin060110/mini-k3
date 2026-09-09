"""Single-GPU Mini-K3 pretraining entry point using MiniMind JSONL/tokenizer."""
import argparse
import json
import math
import random
import signal
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
        self.path = str(Path(path).resolve())
        self.offsets = []
        with open(self.path, "rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        if not self.offsets:
            raise ValueError(f"no records in {path}")
        self.tokenizer, self.seq_len = tokenizer, seq_len
        self._handle = None

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        if self._handle is None:
            self._handle = open(self.path, "rb")
        self._handle.seek(self.offsets[index])
        row = json.loads(self._handle.readline().decode("utf-8"))
        tok = self.tokenizer(str(row["text"]), add_special_tokens=False,
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


def save_checkpoint(path, model, optimizer, cfg, update, epoch, micro_step, args, reason):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model": model.state_dict(), "config": cfg.__dict__,
        "optimizer": optimizer.state_dict(), "updates": update,
        "epoch": epoch, "micro_step": micro_step,
        "python_rng_state": random.getstate(), "torch_rng_state": torch.get_rng_state(),
        "train_args": vars(args), "save_reason": reason,
    }, tmp)
    tmp.replace(path)
    print(json.dumps({"event": "checkpoint", "reason": reason, "updates": update,
                      "epoch": epoch + 1, "micro_step": micro_step, "path": str(path)}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/mini_k3_15m.json")
    p.add_argument("--data", required=True, help="MiniMind pretrain JSONL with a text field")
    p.add_argument("--tokenizer", default="model", help="MiniMind tokenizer path, or 'byte' for smoke tests")
    p.add_argument("--output", default="out/mini_k3_last.pt")
    p.add_argument("--resume", default="", help="checkpoint to resume model/optimizer/progress from")
    p.add_argument("--pause-file", default="", help="save and exit safely when this file exists")
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
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=0, help="0 means a full epoch schedule")
    p.add_argument("--num-workers", type=int, default=0)
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
    device = torch.device(args.device)
    model = MiniK3ForCausalLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  betas=(0.9, 0.95), weight_decay=args.weight_decay)
    batches_per_epoch = math.ceil(len(dataset) / args.batch_size)
    total_updates = args.max_steps or math.ceil(batches_per_epoch / args.grad_accum) * args.epochs
    use_amp = device.type == "cuda"
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
    update, start_epoch, start_micro_step = 0, 0, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if checkpoint.get("optimizer"):
            optimizer.load_state_dict(checkpoint["optimizer"])
        update = int(checkpoint.get("updates", 0))
        start_epoch = int(checkpoint.get("epoch", 0))
        start_micro_step = int(checkpoint.get("micro_step", 0))
        if checkpoint.get("python_rng_state"):
            random.setstate(checkpoint["python_rng_state"])
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        print(json.dumps({"event": "resumed", "checkpoint": args.resume, "updates": update,
                          "epoch": start_epoch + 1, "micro_step": start_micro_step}), flush=True)
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print(json.dumps({"event": "stop_requested", "signal": signum}), flush=True)

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    last_epoch, last_micro_step = start_epoch, start_micro_step
    finished = False
    for epoch in range(start_epoch, args.epochs):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        indices = torch.randperm(len(dataset), generator=generator).tolist()
        skip_samples = start_micro_step * args.batch_size if epoch == start_epoch else 0
        epoch_dataset = torch.utils.data.Subset(dataset, indices[skip_samples:])
        loader = DataLoader(epoch_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                            num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"))
        for micro_step, (ids, labels, mask) in enumerate(loader, 1):
            absolute_micro_step = micro_step + (start_micro_step if epoch == start_epoch else 0)
            ids, labels, mask = ids.to(device), labels.to(device), mask.to(device)
            with amp:
                out = model(ids, attention_mask=mask, labels=labels)
                loss = out["loss"] / args.grad_accum
            loss.backward()
            if absolute_micro_step % args.grad_accum == 0 or micro_step == len(loader):
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
                                      "aux_loss": float(out["aux_loss"].detach()), "lr": lr}), flush=True)
                if args.save_every and update % args.save_every == 0:
                    save_checkpoint(args.output, model, optimizer, cfg, update, epoch,
                                    absolute_micro_step, args, "interval")
                last_epoch, last_micro_step = epoch, absolute_micro_step
                pause_requested = bool(args.pause_file and Path(args.pause_file).exists())
                if stop_requested or pause_requested:
                    reason = "pause_file" if pause_requested else "signal"
                    save_checkpoint(args.output, model, optimizer, cfg, update, epoch,
                                    absolute_micro_step, args, reason)
                    print(json.dumps({"event": "paused", "updates": update}), flush=True)
                    return
                if args.max_steps and update >= args.max_steps:
                    finished = True
                    break
        start_micro_step = 0
        if finished:
            break
        last_epoch, last_micro_step = epoch + 1, 0
    save_checkpoint(args.output, model, optimizer, cfg, update, last_epoch,
                    last_micro_step, args, "complete")
    print(json.dumps({"event": "complete", "updates": update}), flush=True)


if __name__ == "__main__":
    main()
