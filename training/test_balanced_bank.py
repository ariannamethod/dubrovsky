import copy
import unittest

import torch

from train_balanced_bank import actual_step, assign_gradient, restore


class AdamAuditTests(unittest.TestCase):
    def test_inherited_momentum_can_reverse_raw_gradient_direction(self):
        p = torch.nn.Parameter(torch.tensor([1., 2.]))
        optimizer = torch.optim.AdamW([p], lr=.01, weight_decay=0)
        for _ in range(4):
            p.grad = torch.tensor([-10., -10.]); optimizer.step()
        result = actual_step(optimizer, [p], torch.ones(2), torch.ones(2))
        self.assertGreater(result['first_order_contrast_change'], 0)
        self.assertLess(result['cosine_descent_contrast'], 0)
        self.assertLess(result['clip_scale'], 1)

    def test_restore_replays_same_real_clipped_adam_step(self):
        block = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(block.parameters(), lr=.003, weight_decay=0)
        parameters = list(block.parameters())
        vector = torch.arange(1., 9.)
        actual_step(optimizer, parameters, -vector, vector)
        state = {k: v.clone() for k,v in block.state_dict().items()}
        moments = copy.deepcopy(optimizer.state_dict())
        results = []
        for _ in range(2):
            restore(block, optimizer, state, moments)
            result = actual_step(optimizer, parameters, vector, vector)
            results.append((result,{k:v.clone() for k,v in block.state_dict().items()}))
        self.assertEqual(results[0][0],results[1][0])
        for k in state: self.assertTrue(torch.equal(results[0][1][k],results[1][1][k]))
        self.assertEqual(moments['state'][0]['step'].item(), 1)
        restore(block, optimizer, state, moments)
        for k in state: self.assertTrue(torch.equal(block.state_dict()[k],state[k]))
        for p in parameters: self.assertIsNone(p.grad)

    def test_reject_gradient_layout_mismatch(self):
        with self.assertRaises(ValueError):
            assign_gradient([torch.nn.Parameter(torch.zeros(3))],torch.ones(4))


if __name__=='__main__':unittest.main()
