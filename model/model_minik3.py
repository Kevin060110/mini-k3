"""Mini-K3: a small, educational adaptation of Kimi K3 for MiniMind.

This is not a bit-for-bit reproduction of the 2.8T Kimi K3 model.  It keeps the
ideas that remain useful at <1B scale: hybrid delta/full attention, attentive
residual routing and a stable latent-space MoE.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class MiniK3Config:
    vocab_size: int = 6400
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    intermediate_size: int = 1408
    max_position_embeddings: int = 32768
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    # K3-inspired architecture
    attention_pattern: str = "KKKF"  # K=KDA, F=full SDPA
    attn_residual: bool = True
    attn_residual_window: int = 4
    use_moe: bool = True
    latent_size: int = 256
    num_experts: int = 8
    num_experts_per_tok: int = 2
    num_shared_experts: int = 1
    moe_intermediate_size: int = 768
    router_temperature: float = 1.0
    router_z_loss_coef: float = 1e-3
    router_aux_loss_coef: float = 1e-2
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if not 0 < self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok must be in [1, num_experts]")
        if not self.attention_pattern or set(self.attention_pattern.upper()) - {"K", "F"}:
            raise ValueError("attention_pattern must contain only K and F")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


def _rope_cache(length: int, dim: int, theta: float, device: torch.device) -> Tuple[Tensor, Tensor]:
    inv = 1.0 / theta ** (torch.arange(0, dim, 2, device=device).float() / dim)
    phase = torch.outer(torch.arange(length, device=device).float(), inv)
    return phase.cos(), phase.sin()


def _rope(x: Tensor, cos: Tensor, sin: Tensor, offset: int = 0) -> Tensor:
    # x: [B, H, T, D]
    t = x.shape[-2]
    c, s = cos[offset:offset + t][None, None], sin[offset:offset + t][None, None]
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).to(x.dtype)


class FullAttention(nn.Module):
    def __init__(self, cfg: MiniK3Config):
        super().__init__()
        self.h, self.kv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.d = cfg.hidden_size // cfg.num_attention_heads
        self.q = nn.Linear(cfg.hidden_size, self.h * self.d, bias=False)
        self.k = nn.Linear(cfg.hidden_size, self.kv * self.d, bias=False)
        self.v = nn.Linear(cfg.hidden_size, self.kv * self.d, bias=False)
        self.o = nn.Linear(self.h * self.d, cfg.hidden_size, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.d, cfg.rms_norm_eps), RMSNorm(self.d, cfg.rms_norm_eps)
        self.dropout = cfg.dropout

    def forward(self, x: Tensor, rope: Tuple[Tensor, Tensor], attention_mask: Optional[Tensor] = None) -> Tensor:
        b, t, _ = x.shape
        q = self.q_norm(self.q(x).view(b, t, self.h, self.d)).transpose(1, 2)
        k = self.k_norm(self.k(x).view(b, t, self.kv, self.d)).transpose(1, 2)
        v = self.v(x).view(b, t, self.kv, self.d).transpose(1, 2)
        q, k = _rope(q, *rope), _rope(k, *rope)
        repeat = self.h // self.kv
        k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
        mask = None
        if attention_mask is not None and not bool(attention_mask.all()):
            causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
            mask = causal[None, None] & attention_mask[:, None, None, :].bool()
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           dropout_p=self.dropout if self.training else 0.0,
                                           is_causal=mask is None)
        return self.o(y.transpose(1, 2).reshape(b, t, -1))


class KimiDeltaAttention(nn.Module):
    """Causal gated delta-rule linear attention.

    The recurrent reference path is intentionally plain PyTorch and numerically
    stable. Its state is O(H*D^2), independent of sequence length; production
    training can replace this module with a fused/chunkwise KDA kernel.
    """
    def __init__(self, cfg: MiniK3Config):
        super().__init__()
        self.h = cfg.num_attention_heads
        self.d = cfg.hidden_size // self.h
        width = self.h * self.d
        self.qkv = nn.Linear(cfg.hidden_size, width * 3, bias=False)
        self.beta = nn.Linear(cfg.hidden_size, self.h, bias=True)
        self.decay = nn.Linear(cfg.hidden_size, self.h, bias=True)
        self.o = nn.Linear(width, cfg.hidden_size, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.d, cfg.rms_norm_eps), RMSNorm(self.d, cfg.rms_norm_eps)
        nn.init.constant_(self.decay.bias, 2.0)

    def forward(self, x: Tensor, rope: Tuple[Tensor, Tensor], attention_mask: Optional[Tensor] = None) -> Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self.q_norm(q.view(b, t, self.h, self.d)).transpose(1, 2)
        k = self.k_norm(k.view(b, t, self.h, self.d)).transpose(1, 2)
        v = v.view(b, t, self.h, self.d).transpose(1, 2)
        q, k = _rope(q, *rope), _rope(k, *rope)
        q, k = F.elu(q.float()) + 1.0, F.elu(k.float()) + 1.0
        q = q / (q.sum(-1, keepdim=True) + 1e-6)
        k = k / (k.sum(-1, keepdim=True) + 1e-6)
        beta = torch.sigmoid(self.beta(x)).transpose(1, 2).float()
        decay = torch.sigmoid(self.decay(x)).transpose(1, 2).float()
        state = x.new_zeros((b, self.h, self.d, self.d), dtype=torch.float32)
        outputs = []
        for i in range(t):
            ki, vi, qi = k[:, :, i], v[:, :, i].float(), q[:, :, i]
            prediction = torch.einsum("bhde,bhd->bhe", state, ki)
            error = vi - prediction
            update = torch.einsum("bhd,bhe->bhde", ki, error)
            valid = (torch.ones(b, 1, 1, 1, device=x.device) if attention_mask is None
                     else attention_mask[:, i].float()[:, None, None, None])
            decay_i = decay[:, :, i, None, None]
            beta_i = beta[:, :, i, None, None]
            state = state * (1.0 - valid * (1.0 - decay_i))
            state = state + valid * beta_i * update
            outputs.append(torch.einsum("bhde,bhd->bhe", state, qi))
        y = torch.stack(outputs, dim=2).to(x.dtype)
        return self.o(y.transpose(1, 2).reshape(b, t, -1))


class SwiGLU(nn.Module):
    def __init__(self, in_dim: int, mid_dim: int, out_dim: int):
        super().__init__()
        self.gate_up = nn.Linear(in_dim, 2 * mid_dim, bias=False)
        self.down = nn.Linear(mid_dim, out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_up(x).chunk(2, -1)
        return self.down(F.silu(gate) * up)


class StableLatentMoE(nn.Module):
    """Sparse experts in a compact latent channel with shared expert capacity."""
    def __init__(self, cfg: MiniK3Config):
        super().__init__()
        self.cfg = cfg
        self.down = nn.Linear(cfg.hidden_size, cfg.latent_size, bias=False)
        self.up = nn.Linear(cfg.latent_size, cfg.hidden_size, bias=False)
        self.router = nn.Linear(cfg.latent_size, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLU(cfg.latent_size, cfg.moe_intermediate_size, cfg.latent_size)
            for _ in range(cfg.num_experts)
        ])
        self.shared = nn.ModuleList([
            SwiGLU(cfg.latent_size, cfg.moe_intermediate_size, cfg.latent_size)
            for _ in range(cfg.num_shared_experts)
        ])
        self.register_buffer("routing_bias", torch.zeros(cfg.num_experts))
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        z = self.down(x).reshape(-1, self.cfg.latent_size)
        logits = self.router(F.normalize(z.float(), dim=-1)) / self.cfg.router_temperature
        probs = logits.softmax(-1)
        route_scores = probs + self.routing_bias
        _, ids = route_scores.topk(self.cfg.num_experts_per_tok, dim=-1)
        weights = probs.gather(-1, ids)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        out = torch.zeros_like(z)
        for expert_id, expert in enumerate(self.experts):
            token, slot = torch.where(ids == expert_id)
            if token.numel():
                out.index_add_(0, token, expert(z[token]) * weights[token, slot, None].to(z.dtype))
            elif self.training:
                out = out + 0.0 * sum(p.sum() for p in expert.parameters())
        if self.shared:
            out = out + sum(expert(z) for expert in self.shared) / len(self.shared)
        importance = probs.mean(0)
        load = F.one_hot(ids, self.cfg.num_experts).float().mean((0, 1))
        balance = self.cfg.num_experts * (importance * load.detach()).sum()
        z_loss = logits.logsumexp(-1).square().mean()
        self.aux_loss = (self.cfg.router_aux_loss_coef * balance
                         + self.cfg.router_z_loss_coef * z_loss)
        return self.up(out).view(shape)


class AttentionResidualMixer(nn.Module):
    """Token-wise attentive selection over a bounded bank of prior layer states."""
    def __init__(self, cfg: MiniK3Config):
        super().__init__()
        self.query = nn.Linear(cfg.hidden_size, 1, bias=False)
        self.keys = nn.Parameter(torch.zeros(cfg.attn_residual_window))
        self.window = cfg.attn_residual_window

    def forward(self, current: Tensor, residuals: List[Tensor]) -> Tensor:
        bank = residuals[-self.window:]
        if len(bank) == 1:
            return bank[0]
        scores = self.query(current.float()).squeeze(-1)[..., None]
        scores = scores * self.keys[:len(bank)].float()
        weights = scores.softmax(-1).to(current.dtype)
        return sum(weights[..., i, None] * value for i, value in enumerate(bank))


class MiniK3Block(nn.Module):
    def __init__(self, cfg: MiniK3Config, layer_id: int):
        super().__init__()
        kind = cfg.attention_pattern[layer_id % len(cfg.attention_pattern)].upper()
        self.attn = KimiDeltaAttention(cfg) if kind == "K" else FullAttention(cfg)
        self.attn_kind = kind
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mixer = AttentionResidualMixer(cfg) if cfg.attn_residual else None
        self.ffn = StableLatentMoE(cfg) if cfg.use_moe else SwiGLU(cfg.hidden_size, cfg.intermediate_size, cfg.hidden_size)

    def forward(self, x: Tensor, residuals: List[Tensor], rope: Tuple[Tensor, Tensor], mask: Optional[Tensor]) -> Tensor:
        residual = self.mixer(x, residuals) if self.mixer is not None else x
        x = residual + self.attn(self.attn_norm(x), rope, mask)
        return x + self.ffn(self.ffn_norm(x))


class MiniK3ForCausalLM(nn.Module):
    def __init__(self, config: Optional[MiniK3Config] = None):
        super().__init__()
        self.config = config or MiniK3Config()
        c = self.config
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList([MiniK3Block(c, i) for i in range(c.num_hidden_layers)])
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        if c.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None,
                labels: Optional[Tensor] = None) -> dict:
        _, t = input_ids.shape
        if t > self.config.max_position_embeddings:
            raise ValueError(f"sequence length {t} exceeds max_position_embeddings")
        x = self.embed_tokens(input_ids)
        rope = _rope_cache(t, self.config.hidden_size // self.config.num_attention_heads,
                           self.config.rope_theta, x.device)
        residuals = [x]
        for layer in self.layers:
            x = layer(x, residuals, rope, attention_mask)
            residuals.append(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, self.config.vocab_size),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
            loss = loss + self.aux_loss
        return {"loss": loss, "logits": logits, "aux_loss": self.aux_loss}

    @property
    def aux_loss(self) -> Tensor:
        losses = [layer.ffn.aux_loss for layer in self.layers if isinstance(layer.ffn, StableLatentMoE)]
        return sum(losses, self.embed_tokens.weight.new_zeros(()))

    @torch.inference_mode()
    def generate(self, input_ids: Tensor, max_new_tokens: int = 64, temperature: float = 0.8,
                 top_k: int = 50, eos_token_id: Optional[int] = 2) -> Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(input_ids[:, -self.config.max_position_embeddings:])["logits"][:, -1]
            if temperature <= 0:
                token = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    cutoff = logits.topk(min(top_k, logits.shape[-1])).values[:, -1, None]
                    logits = logits.masked_fill(logits < cutoff, -torch.inf)
                token = torch.multinomial(logits.softmax(-1), 1)
            input_ids = torch.cat((input_ids, token), dim=1)
            if eos_token_id is not None and bool((token == eos_token_id).all()):
                break
        return input_ids

    def num_parameters(self, active_only: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if not active_only or not self.config.use_moe:
            return total
        inactive_fraction = 1.0 - self.config.num_experts_per_tok / self.config.num_experts
        expert_params = sum(sum(p.numel() for p in layer.ffn.experts.parameters())
                            for layer in self.layers if isinstance(layer.ffn, StableLatentMoE))
        return int(total - inactive_fraction * expert_params)
