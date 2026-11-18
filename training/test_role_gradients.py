import unittest
import torch
from probe_role_gradients import gradient_vector,gradient_summary

class GradientDiagnosticsTests(unittest.TestCase):
    def test_opposed_objectives_and_unused_parameter(self):
        a=torch.tensor([2.,3.],requires_grad=True);b=torch.tensor([7.],requires_grad=True)
        v=gradient_vector(a.sum(),[a,b]);w=gradient_vector(-a.sum(),[a,b])
        self.assertTrue(torch.equal(v,torch.tensor([1.,1.,0.])))
        result=gradient_summary(dict(ce=v,contrast=w,replay=torch.zeros_like(v)),[('reader',2),('output',1)])
        self.assertAlmostEqual(result['cosine_ce_contrast'],-1,places=6)
        self.assertIsNone(result['cosine_combined_contrast'])
        self.assertEqual(result['combined']['norm'],0)
        self.assertIsNone(a.grad);self.assertIsNone(b.grad)
        self.assertTrue(torch.equal(a,torch.tensor([2.,3.])))

    def test_replay_weight_and_group_boundaries(self):
        v=torch.tensor([3.,4.,0.]);r=gradient_summary(dict(ce=v,contrast=v,replay=v),[('reader',2),('output',1)])
        self.assertAlmostEqual(r['weighted']['replay']['norm'],1,places=6)
        self.assertEqual(r['weighted']['replay']['groups']['output'],0)
        with self.assertRaises(ValueError):gradient_summary(dict(ce=v,contrast=v,replay=v),[('reader',1)])
