import unittest
import torch
from embedded_experts import EmbeddedExperts, ExpertConfig
from isolated_writer import replay_bank, configure_writer, WRITER, weak_side_changes


class WriterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        torch.set_num_threads(1)
        self.block = EmbeddedExperts(ExpertConfig(12, experts=2, rank=3, format_version=3, max_output_gain=8))
        with torch.no_grad():
            self.block.output.normal_(0, .1)
            self.block.retain.normal_(0, .1)
            self.block.write.normal_(0, .1)
        self.raw = torch.randn(1, 9, 12)
        self.source = torch.randn_like(self.raw)
        self.mask = torch.tensor([[False, True, True, False, True, False, True, True, False]])

    def test_native_values_and_writer_gradients_match(self):
        native = self.block(self.raw, self.mask, memory_hidden=self.source)[1].user
        replay = replay_bank(self.block, self.source, self.mask)
        self.assertTrue(torch.equal(native, replay))
        parameters = [p for n,p in self.block.named_parameters() if n in WRITER]
        weight = torch.randn_like(native)
        a = torch.autograd.grad((native*weight).sum(), parameters)
        b = torch.autograd.grad((replay*weight).sum(), parameters)
        for x,y in zip(a,b):
            torch.testing.assert_close(x,y,rtol=1e-6,atol=1e-7)
            self.assertGreater(x.norm().item(),0)

    def test_assistant_positions_and_length_do_not_enter_bank(self):
        bank = replay_bank(self.block,self.source,self.mask)
        source = torch.cat((self.source[:,:3],torch.randn(1,40,12),self.source[:,3:]),1)
        mask = torch.cat((self.mask[:,:3],torch.zeros(1,40,dtype=torch.bool),self.mask[:,3:]),1)
        self.assertTrue(torch.equal(bank,replay_bank(self.block,source,mask)))
        changed = self.source.clone(); changed[:,4] *= -1
        self.assertFalse(torch.equal(bank,replay_bank(self.block,changed,self.mask)))

    def test_bank_gradient_survives_state_replacement(self):
        from dataclasses import replace
        state = self.block(self.raw,self.mask,memory_hidden=self.source)[1].detached()
        bank = replay_bank(self.block,self.source,self.mask)
        state = replace(state,user=bank)
        output = self.block(self.raw[:,:1],torch.zeros(1,1,dtype=torch.bool),state)[0]
        output.sum().backward()
        self.assertGreater(self.block.source.grad.norm().item(),0)
        self.assertFalse(state.working.requires_grad)

    def test_freeze_groups(self):
        from train_role_readout import TRAINABLE
        for enabled in (False,True):
            configure_writer(self.block,enabled)
            names={n for n,p in self.block.named_parameters() if p.requires_grad}
            self.assertEqual(names,TRAINABLE | WRITER if enabled else TRAINABLE)

    def test_contrast_can_improve_while_wrong_side_worsens(self):
        def result(m,s):return {'rows':[dict(id='a',pair='p',native=dict(margin=m,swapped_margin=s))]}
        r=weak_side_changes(result(4,2),result(6,3))
        self.assertEqual(r['contrast_improved'],1)
        self.assertEqual(r['weak_improved'],0)
        with self.assertRaises(ValueError):weak_side_changes(result(4,2),result(float('nan'),-1))


if __name__ == '__main__':unittest.main()
