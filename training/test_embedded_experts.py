import json
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from embedded_experts import EmbeddedExperts, ExpertConfig, ExpertQwen, tensor_hashes
from train_experts import row_loss


def tiny_qwen():
    config = Qwen3_5Config(
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=8, linear_value_head_dim=8,
            linear_conv_kernel_dim=4, layer_types=['linear_attention', 'full_attention'],
            tie_word_embeddings=True, pad_token_id=0, eos_token_id=2,
            rope_parameters=dict(rope_type='default', rope_theta=10000.0,
                                 partial_rotary_factor=0.5, mrope_section=[1, 1, 2])),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32,
            num_heads=2, patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
            out_hidden_size=32, num_position_embeddings=16),
        tie_word_embeddings=True,
        image_token_id=60, video_token_id=61, vision_start_token_id=62,
        vision_end_token_id=63)
    return Qwen3_5ForConditionalGeneration(config).eval()


class ExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(240)
        self.block = EmbeddedExperts(ExpertConfig(32, rank=4))
        self.x = torch.randn(1, 12, 32)
        self.mask = torch.tensor([[False, True, True, False, False, False,
                                  True, True, True, False, False, False]])

    def activate(self):
        with torch.no_grad():
            self.block.output.normal_(0, 0.1)

    def test_zero_output_is_exact(self):
        y, _, _ = self.block(self.x, self.mask)
        self.assertTrue(torch.equal(y, self.x))

    def test_chunked_state_matches_full_with_nonzero_experts(self):
        self.activate()
        full, state, votes = self.block(self.x, self.mask)
        a, first, va = self.block(self.x[:, :5], self.mask[:, :5])
        b, last, vb = self.block(self.x[:, 5:], self.mask[:, 5:], first)
        torch.testing.assert_close(torch.cat((a, b), 1), full, rtol=0, atol=0)
        torch.testing.assert_close(last.user, state.user, rtol=0, atol=0)
        torch.testing.assert_close(torch.cat((va, vb), 1), votes, rtol=0, atol=0)

    def test_future_tokens_do_not_change_earlier_outputs(self):
        self.activate()
        short, _, _ = self.block(self.x[:, :6], self.mask[:, :6])
        changed = self.x.clone(); changed[:, 6:] = 123
        long, _, _ = self.block(changed, self.mask)
        self.assertTrue(torch.equal(short, long[:, :6]))

    def test_assistant_does_not_rewrite_user_bank(self):
        _, start, _ = self.block(self.x[:, :4], self.mask[:, :4])
        saved = start.user.clone()
        _, end, _ = self.block(self.x[:, 4:], torch.zeros_like(self.mask[:, 4:]), start)
        self.assertTrue(torch.equal(start.user, saved))
        self.assertTrue(torch.equal(end.user, saved))
        self.assertFalse(torch.equal(end.working, start.working))

    def test_preview_is_pure_and_reset_is_exact(self):
        self.activate()
        _, start, _ = self.block(self.x[:, :6], self.mask[:, :6])
        a, sa, _ = self.block(self.x[:, 6:], self.mask[:, 6:], start)
        b, sb, _ = self.block(self.x[:, 6:], self.mask[:, 6:], start)
        self.assertTrue(torch.equal(a, b)); self.assertTrue(torch.equal(sa.user, sb.user))
        a, _, _ = self.block(self.x, self.mask)
        b, _, _ = self.block(self.x, self.mask, self.block.initial_state(1))
        self.assertTrue(torch.equal(a, b))

    def test_memory_off_and_experts_off_are_separate(self):
        self.activate()
        on, state, _ = self.block(self.x, self.mask)
        off, no_memory, _ = self.block(self.x, self.mask, memory_enabled=False)
        disabled, _, _ = self.block(self.x, self.mask, enabled=False)
        self.assertGreater(state.user.abs().max().item(), 0)
        self.assertEqual(no_memory.user.abs().max().item(), 0)
        self.assertFalse(torch.equal(on, off))
        self.assertTrue(torch.equal(disabled, self.x))

    def test_output_bound_and_live_gradients(self):
        self.activate()
        y, _, _ = self.block(self.x, self.mask)
        self.assertLessEqual((y-self.x).square().mean(-1).sqrt().max().item(), 0.100001)
        y.square().mean().backward()
        for name in ['source', 'retain', 'write', 'output', 'input', 'memory_read']:
            grad = getattr(self.block, name).grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(grad.abs().sum().item(), 0, name)

    def test_bad_masks_and_config_fail(self):
        with self.assertRaises(ValueError): self.block(self.x, self.mask.float())
        with self.assertRaises(ValueError): ExpertConfig(32, rank=0)
        with self.assertRaises(ValueError): ExpertConfig(32, max_delta_rms=float('nan'))


class QwenIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(24)
        self.model = ExpertQwen(tiny_qwen(), ExpertConfig(32, rank=4))
        self.ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
        self.mask = torch.tensor([[False, True, True, False, True, True, False, False]])

    def test_zero_parity_on_real_qwen_architecture(self):
        with torch.no_grad():
            expected = self.model.base(self.ids, use_cache=False).logits
            actual = self.model(self.ids, self.mask).logits
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(self.model.base.get_input_embeddings().weight.data_ptr(),
                         self.model.base.get_output_embeddings().weight.data_ptr())

    def test_full_and_cached_nonzero_logits(self):
        with torch.no_grad():
            self.model.experts.output.normal_(0, 0.1)
            full = self.model(self.ids, self.mask)
            a = self.model(self.ids[:, :5], self.mask[:, :5], use_cache=True)
            pieces = [a.logits]
            for i in range(5, self.ids.shape[1]):
                a = self.model(self.ids[:, i:i+1], self.mask[:, i:i+1],
                    past_key_values=a.past_key_values, state=a.state, use_cache=True)
                pieces.append(a.logits)
        torch.testing.assert_close(torch.cat(pieces, 1), full.logits, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(a.state.user, full.state.user, atol=1e-6, rtol=1e-5)

    def test_optimizer_changes_only_experts_and_reload_retains_them(self):
        before = tensor_hashes(self.model.base)
        optimizer = torch.optim.AdamW(self.model.experts.parameters(), lr=0.01)
        self.model.train()
        self.assertFalse(self.model.base.training)
        for _ in range(2):
            optimizer.zero_grad()
            result = self.model(self.ids, self.mask)
            torch.nn.functional.cross_entropy(result.logits[0, :-1], self.ids[0, 1:]).backward()
            optimizer.step()
        self.assertEqual(before, tensor_hashes(self.model.base))
        self.assertTrue(all(p.grad is None for p in self.model.base.parameters()))
        self.assertGreater(self.model.experts.source.grad.abs().sum().item(), 0)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'model'
            self.model.save_pretrained(path)
            loaded = ExpertQwen.from_pretrained(path)
            torch.testing.assert_close(loaded(self.ids, self.mask).logits,
                                       self.model(self.ids, self.mask).logits, rtol=0, atol=0)
            self.assertEqual(before, tensor_hashes(loaded.base))
            self.assertNotIn('user', json.loads((path/'embedded_experts.json').read_text()))
            with self.assertRaises(ValueError): self.model.save_pretrained(path)

    def test_mismatched_cache_and_memory_rejected(self):
        with torch.no_grad():
            out = self.model(self.ids[:, :3], self.mask[:, :3], use_cache=True)
        with self.assertRaises(ValueError):
            self.model(self.ids[:, 3:4], self.mask[:, 3:4], state=out.state)
        with self.assertRaises(ValueError):
            self.model(self.ids[:, 3:4], self.mask[:, 3:4], past_key_values=out.past_key_values)

    def test_answer_only_loss_matches_full_logits_and_zero_kl(self):
        row = dict(input_ids=self.ids[0].tolist(), user_mask=self.mask[0].tolist(),
                   labels=[-100]*4+self.ids[0, 4:].tolist())
        actual, _, count = row_loss(self.model, row, kl_weight=0.1)
        with torch.no_grad():
            reference = self.model.base(self.ids, use_cache=False).logits
            expected = torch.nn.functional.cross_entropy(reference[0, 3:-1], self.ids[0, 4:])
        self.assertEqual(count, 4)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_bf16_body_keeps_fp32_experts_and_exact_zero_output(self):
        self.model.base.to(dtype=torch.bfloat16)
        self.assertEqual(self.model.experts.source.dtype, torch.float32)
        with torch.no_grad():
            ref = self.model.base(self.ids, use_cache=False).logits
            out = self.model(self.ids, self.mask).logits
        self.assertTrue(torch.equal(ref, out))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'bf16'
            self.model.save_pretrained(path)
            loaded = ExpertQwen.from_pretrained(path)
            self.assertEqual(next(loaded.base.parameters()).dtype, torch.bfloat16)
            self.assertEqual(loaded.experts.source.dtype, torch.float32)
            self.assertEqual(tensor_hashes(loaded.base), tensor_hashes(self.model.base))

    def test_modified_expert_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'model'
            self.model.save_pretrained(path)
            with (path/'experts.safetensors').open('ab') as stream:
                stream.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'Expert tensors differ'):
                ExpertQwen.from_pretrained(path)


if __name__ == '__main__':
    unittest.main()
