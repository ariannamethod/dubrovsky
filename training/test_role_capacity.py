import unittest
import torch
from probe_role_capacity import sphere_capacity


class CapacityTests(unittest.TestCase):
    def test_maximizing_vector_attains_linear_bound(self):
        raw=torch.tensor([-2.,1.,.3,4.]);rows=torch.tensor([[1.,0.,-3.,2.],[-1.,2.,0.,0.]])
        result,delta=sphere_capacity(raw,rows,.1)
        actual=torch.dot(raw+delta,rows[0]-rows[1]).item()
        self.assertAlmostEqual(actual,result['max_margin'],places=5)
        self.assertAlmostEqual(delta.square().mean().sqrt().item(),.1,places=6)

    def test_required_gain_hits_margin(self):
        raw=torch.tensor([-4.,0.]);rows=torch.tensor([[1.,0.],[0.,0.]])
        result,delta=sphere_capacity(raw,rows,.1)
        value=torch.dot(raw+result['gain_for_target']*delta,rows[0]-rows[1]).item()
        self.assertAlmostEqual(value,1.,places=5)
        self.assertGreater(result['gain_for_target'],1.)

    def test_identical_head_rows_have_no_direction(self):
        result,delta=sphere_capacity(torch.ones(3),torch.ones(2,3),.1)
        self.assertEqual(result['bound'],0.)
        self.assertIsNone(result['gain_for_target'])
        self.assertTrue(torch.equal(delta,torch.zeros(3)))


if __name__=='__main__':unittest.main()
