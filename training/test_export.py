import importlib.util
import unittest

from sft import export_bf16_state


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires PyTorch')
class ExportTest(unittest.TestCase):
    def test_mid_training_export_preserves_optimizer_trajectory(self):
        import copy
        import torch
        torch.manual_seed(17)
        uninterrupted = torch.nn.Linear(4, 3)
        checkpointed = copy.deepcopy(uninterrupted)
        optimizers = [torch.optim.AdamW(model.parameters(), lr=.01)
                      for model in (uninterrupted, checkpointed)]
        x, target = torch.randn(5, 4), torch.randn(5, 3)
        for step in range(5):
            for model, optimizer in zip((uninterrupted, checkpointed), optimizers):
                optimizer.zero_grad()
                (model(x) - target).square().mean().backward()
                optimizer.step()
            if step == 1:
                export_bf16_state(checkpointed)
        for left, right in zip(uninterrupted.parameters(), checkpointed.parameters()):
            self.assertTrue(torch.equal(left, right))
        left, right = [optimizer.state_dict()['state'] for optimizer in optimizers]
        for parameter_id in left:
            for name in left[parameter_id]:
                self.assertTrue(torch.equal(left[parameter_id][name], right[parameter_id][name]))

    def test_tied_weights_and_live_fp32_parameters(self):
        import torch
        model = torch.nn.Module()
        model.embedding = torch.nn.Embedding(7, 3)
        model.head = torch.nn.Linear(3, 7, bias=False)
        model.head.weight = model.embedding.weight
        before = model.embedding.weight.detach().clone()
        pointer = model.embedding.weight.data_ptr()
        state = export_bf16_state(model)
        self.assertEqual(state['embedding.weight'].data_ptr(), state['head.weight'].data_ptr())
        self.assertEqual(state['embedding.weight'].dtype, torch.bfloat16)
        self.assertEqual(state['embedding.weight'].device.type, 'cpu')
        self.assertEqual(model.embedding.weight.dtype, torch.float32)
        self.assertEqual(model.embedding.weight.data_ptr(), pointer)
        self.assertTrue(torch.equal(before, model.embedding.weight))

    def test_equal_independent_weights_are_not_aliased(self):
        import torch
        model = torch.nn.Module()
        model.register_buffer('a', torch.ones(2, 3))
        model.register_buffer('b', model.a.clone())
        state = export_bf16_state(model)
        self.assertNotEqual(state['a'].data_ptr(), state['b'].data_ptr())

    def test_views_offsets_and_integer_buffers(self):
        import torch
        model = torch.nn.Module()
        whole = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        for name, value in [('whole', whole), ('transposed', whole.T),
                            ('first', whole[0]), ('second', whole[1]),
                            ('counter', torch.tensor(9, dtype=torch.int64))]:
            model.register_buffer(name, value)
        state = export_bf16_state(model)
        for name, value in model.state_dict().items():
            expected = value.bfloat16() if value.is_floating_point() else value
            self.assertTrue(torch.equal(state[name], expected))
        self.assertNotEqual(state['first'].data_ptr(), state['second'].data_ptr())
        self.assertEqual(state['counter'].dtype, torch.int64)


if __name__ == '__main__':
    unittest.main()
