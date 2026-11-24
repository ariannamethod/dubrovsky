import unittest
import torch
from embedded_experts import EmbeddedExperts,ExpertConfig
from probe_bank_component import bounded_delta,mix_delta,positive_gain_for_margin


class BankComponentTests(unittest.TestCase):
    def test_exact_readout_reconstruction(self):
        torch.manual_seed(33);torch.set_num_threads(1)
        block=EmbeddedExperts(ExpertConfig(16,rank=4,format_version=3,max_output_gain=8))
        with torch.no_grad():block.output.normal_(0,.2);block.log_output_gain.fill_(1.)
        raw=torch.randn(1,7,16).bfloat16();source=torch.randn_like(raw)
        mask=torch.tensor([[True,True,False,True,False,False,False]])
        for enabled in (True,False):
            h,s,v=block(raw,mask,memory_hidden=source,memory_enabled=enabled)
            delta=bounded_delta(block,s,v[:,-1])*block.output_gain()
            self.assertTrue(torch.equal(h[:,-1],raw[:,-1]+delta.to(raw.dtype)))

    def test_unit_gain_and_zero_bank_effect_are_exact(self):
        on=torch.randn(2,31);zero=torch.randn_like(on)
        self.assertTrue(torch.equal(mix_delta(on,zero,1.),on))
        self.assertTrue(torch.equal(mix_delta(on,on,64.),on))

    def test_positive_gain_cannot_reverse_an_opposing_direction(self):
        self.assertIsNone(positive_gain_for_margin(-3.,-.1))
        self.assertIsNone(positive_gain_for_margin(-3.,0.))
        gain=positive_gain_for_margin(-3.,.5)
        self.assertEqual(-3.+(gain-1)*.5,1.)


if __name__=='__main__':unittest.main()
