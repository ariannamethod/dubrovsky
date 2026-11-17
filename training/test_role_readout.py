import unittest
from types import SimpleNamespace
import torch
from torch import nn
from embedded_experts import EmbeddedExperts, ExpertConfig, ExpertState
from train_role_readout import configure, TRAINABLE, swapped_state, causal_loss, value_logp


class RoleReadoutTests(unittest.TestCase):
    def test_update_preserves_writer_and_source_bank(self):
        torch.manual_seed(91)
        block=EmbeddedExperts(ExpertConfig(12,experts=2,rank=3))
        with torch.no_grad(): block.output.normal_(0,.03)
        optimizer=configure(block)
        raw=torch.randn(1,7,12);mask=torch.tensor([[True,True,False,True,False,False,False]])
        source=torch.randn_like(raw)
        old={n:p.clone() for n,p in block.state_dict().items()}
        output,before,_=block(raw,mask,memory_hidden=source)
        output.square().mean().backward();optimizer.step()
        _,after,_=block(raw,mask,memory_hidden=source)
        self.assertTrue(torch.equal(before.user,after.user))
        for n,p in block.state_dict().items():
            if n not in TRAINABLE:self.assertTrue(torch.equal(old[n],p),n)
        self.assertFalse(torch.equal(old['output'],block.output))
        self.assertIsNotNone(block.memory_read.grad)
        self.assertIsNotNone(block.router_memory.grad)
        self.assertIsNone(block.source.grad)

    def test_donor_keeps_working_prefix_without_aliasing_bank(self):
        state=ExpertState(torch.zeros(1,2,3),torch.ones(1,2,3,requires_grad=True),17)
        donor=torch.full_like(state.user,2)
        changed=swapped_state(state,donor)
        self.assertTrue(torch.equal(changed.working,state.working))
        self.assertFalse(changed.working.requires_grad)
        self.assertEqual(changed.positions,17)
        changed.user.zero_()
        self.assertTrue(torch.all(donor==2))
        self.assertTrue(torch.all(state.user==0))

    def test_objective_rewards_source_direction_without_requiring_name_prior(self):
        normal=torch.tensor(-2.,requires_grad=True);other=torch.tensor(-2.,requires_grad=True)
        loss=causal_loss(normal,other);loss.backward()
        self.assertLess(normal.grad.item(),0)
        self.assertGreater(other.grad.item(),0)
        self.assertLess(causal_loss(torch.tensor(-1.),torch.tensor(-3.)).item(),loss.item())

    def test_fp32_diagnostic_preserves_native_weights_and_dtype(self):
        block=EmbeddedExperts(ExpertConfig(4,experts=2,rank=2))
        with torch.no_grad():block.output.fill_(.0003)
        head=nn.Linear(4,8,bias=False).to(torch.bfloat16).requires_grad_(False)
        model=SimpleNamespace(experts=block,base=SimpleNamespace(lm_head=head))
        raw=torch.ones(1,4,4,dtype=torch.bfloat16)
        prepared=dict(raw=raw,mask=torch.zeros(1,4,dtype=torch.bool),source=torch.zeros_like(raw),cut=1,positions=[3],item={'input_ids':[0,1,2,3]})
        before=head.weight.clone();state=block.initial_state(1)
        for precision in ('native','head32','add32'):
            score=value_logp(model,prepared,state,precision=precision,head32=head.weight.detach().float())
            self.assertTrue(torch.isfinite(score).all())
            self.assertEqual(score.numel(),1)
        self.assertEqual(head.weight.dtype,torch.bfloat16)
        self.assertTrue(torch.equal(before,head.weight))
        self.assertEqual(prepared['raw'].dtype,torch.bfloat16)


if __name__=='__main__':unittest.main()
