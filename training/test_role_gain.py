import math
from pathlib import Path
import tempfile
import unittest

import torch

from embedded_experts import EmbeddedExperts, ExpertConfig, ExpertQwen
from test_embedded_experts import tiny_qwen
from train_role_gain import with_gain, per_source_loss


class GainTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(32)
        torch.set_num_threads(1)
        self.parent=EmbeddedExperts(ExpertConfig(32,rank=4))
        with torch.no_grad():self.parent.output.normal_(0,.3)
        self.block=with_gain(self.parent)
        self.raw=torch.randn(1,9,32)
        self.source=torch.randn_like(self.raw)
        self.mask=torch.tensor([[True,True,False,False,True,False,False,False,False]])

    def test_gain_one_preserves_all_outputs_and_states_in_both_dtypes(self):
        for dtype in (torch.float32,torch.bfloat16):
            raw=self.raw.to(dtype);source=self.source.to(dtype)
            a=self.parent(raw,self.mask,memory_hidden=source)
            b=self.block(raw,self.mask,memory_hidden=source)
            for x,y in zip((a[0],a[1].user,a[1].working,a[2]),(b[0],b[1].user,b[1].working,b[2])):
                self.assertTrue(torch.equal(x,y))

    def test_gain_scales_delta_and_leaves_memory_and_routing_unchanged(self):
        a=self.block(self.raw,self.mask,memory_hidden=self.source)
        with torch.no_grad():self.block.log_output_gain.fill_(math.log(4))
        b=self.block(self.raw,self.mask,memory_hidden=self.source)
        torch.testing.assert_close(b[0]-self.raw,4*(a[0]-self.raw),atol=1e-6,rtol=1e-5)
        self.assertTrue(torch.equal(a[1].user,b[1].user))
        self.assertTrue(torch.equal(a[1].working,b[1].working))
        self.assertTrue(torch.equal(a[2],b[2]))
        self.assertLessEqual((b[0]-self.raw).square().mean(-1).sqrt().max().item(),.400001)
        for value,expected in ((100,8),(-100,.125)):
            with torch.no_grad():self.block.log_output_gain.fill_(value)
            self.assertAlmostEqual(self.block.output_gain().item(),expected,places=6)

    def test_wrong_both_sides_still_receive_corrective_gradient(self):
        margins=torch.tensor([-2.87,-20.867],requires_grad=True)
        loss=per_source_loss(margins).mean();loss.backward()
        self.assertTrue((margins.grad<-.4).all())

    def test_gain_receives_gradient_and_roundtrips_with_metadata(self):
        out=self.block(self.raw,self.mask,memory_hidden=self.source)[0]
        (out-self.raw).square().mean().backward()
        self.assertGreater(abs(self.block.log_output_gain.grad.item()),0)
        model=ExpertQwen(tiny_qwen(),self.block.config)
        model.experts.load_state_dict(self.block.state_dict())
        with torch.no_grad():model.experts.log_output_gain.fill_(math.log(2))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model';model.save_pretrained(path)
            restored=ExpertQwen.from_pretrained(path)
            self.assertEqual(restored.expert_config,model.expert_config)
            for n,v in model.experts.state_dict().items():
                self.assertTrue(torch.equal(v,restored.experts.state_dict()[n]))

    def test_legacy_cannot_silently_enable_gain(self):
        with self.assertRaises(ValueError):ExpertConfig(32,format_version=2,max_output_gain=8)
        with self.assertRaises(ValueError):ExpertConfig(32,format_version=3,max_output_gain=float('nan'))


if __name__=='__main__':unittest.main()
