import unittest

import torch

from model import MiniK3Config, MiniK3ForCausalLM
from model.model_minik3 import StableLatentMoE


def tiny_config(**kwargs):
    values = dict(vocab_size=97, hidden_size=32, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2,
                  intermediate_size=64, latent_size=16,
                  moe_intermediate_size=24, num_experts=4,
                  num_experts_per_tok=2, dropout=0.0)
    values.update(kwargs)
    return MiniK3Config(**values)


class MiniK3Tests(unittest.TestCase):
    def test_forward_backward_is_finite(self):
        torch.manual_seed(0)
        model = MiniK3ForCausalLM(tiny_config())
        ids = torch.randint(0, 97, (2, 11))
        out = model(ids, labels=ids)
        self.assertEqual(out["logits"].shape, (2, 11, 97))
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_all_architecture_ablations(self):
        ids = torch.randint(0, 97, (1, 7))
        for pattern, attn_res, moe in (("F", False, False), ("K", False, False),
                                       ("KF", True, False), ("KKKF", True, True)):
            model = MiniK3ForCausalLM(tiny_config(attention_pattern=pattern,
                                                  attn_residual=attn_res, use_moe=moe))
            self.assertEqual(model(ids)["logits"].shape, (1, 7, 97))

    def test_causal_prefix_invariance(self):
        torch.manual_seed(1)
        model = MiniK3ForCausalLM(tiny_config(attention_pattern="KF")).eval()
        prefix = torch.randint(0, 97, (1, 6))
        a = torch.cat((prefix, torch.tensor([[3, 4]])), 1)
        b = torch.cat((prefix, torch.tensor([[8, 9]])), 1)
        with torch.no_grad():
            la, lb = model(a)["logits"], model(b)["logits"]
        torch.testing.assert_close(la[:, :6], lb[:, :6], rtol=1e-5, atol=1e-5)

    def test_sparse_experts_receive_gradients(self):
        model = MiniK3ForCausalLM(tiny_config())
        ids = torch.randint(0, 97, (4, 16))
        model(ids, labels=ids)["loss"].backward()
        moe = next(layer.ffn for layer in model.layers if isinstance(layer.ffn, StableLatentMoE))
        self.assertIsNotNone(moe.router.weight.grad)
        self.assertTrue(torch.isfinite(moe.router.weight.grad).all())

    def test_active_parameter_count_is_smaller(self):
        model = MiniK3ForCausalLM(tiny_config())
        self.assertLess(model.num_parameters(active_only=True), model.num_parameters())


if __name__ == "__main__":
    unittest.main()
