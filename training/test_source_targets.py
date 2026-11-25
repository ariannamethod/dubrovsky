from types import SimpleNamespace
import unittest
import torch
from embedded_experts import ExpertState
from train_source_targets import source_target_loss, decision_summary


class BankOnlyReadout(torch.nn.Module):
    def __init__(self):
        super().__init__();self.strength=torch.nn.Parameter(torch.tensor(-.5))
        self.seen=[]
    def forward(self,raw,mask,state=None,**kwargs):
        if state is None:
            state=ExpertState(torch.ones(1,1,2),torch.full((1,1,2),7.),0)
        self.seen.append(state.detached())
        score=self.strength*state.user[0,0,0]
        delta=torch.stack((score,-score)).view(1,1,2)
        return raw+delta,state,torch.ones(1,raw.shape[1],1)


class SourceTargetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        block=BankOnlyReadout();head=torch.nn.Linear(2,2,bias=False)
        with torch.no_grad():head.weight.copy_(torch.eye(2))
        head.requires_grad_(False)
        self.model=SimpleNamespace(experts=block,base=SimpleNamespace(lm_head=head))
        common=dict(raw=torch.zeros(1,2,2),mask=torch.zeros(1,2,dtype=torch.bool),
            source=torch.zeros(1,2,2),cut=1,positions=[2])
        self.item=dict(alternatives=[{**common,'item':{'input_ids':[0,0,0]}},
            {**common,'item':{'input_ids':[0,0,1]}}],normal_bank=torch.ones(1,1,2),bank=-torch.ones(1,1,2))

    def test_donor_target_learns_counterpart_and_preserves_working_prefix(self):
        loss,stats=source_target_loss(self.model,self.item)
        # Donor has logits [.5,-.5]: its correct counterpart is token 1.
        self.assertAlmostEqual(stats['donor_ce'],torch.logsumexp(torch.tensor([.5,-.5]),0).item()+.5,places=6)
        self.assertEqual(stats['donor_target_margin'],-1.)
        loss.backward();self.assertLess(self.model.experts.strength.grad.item(),0)
        with torch.no_grad():self.model.experts.strength.add_(-.1*self.model.experts.strength.grad)
        after,_=source_target_loss(self.model,self.item);self.assertLess(after.item(),loss.item())
        for state in self.model.experts.seen:self.assertTrue(torch.equal(state.working,torch.full((1,1,2),7.)))
        self.assertTrue(torch.equal(self.item['bank'],-torch.ones(1,1,2)))

    def test_stale_normal_bank_is_rejected(self):
        self.item['normal_bank'].zero_()
        with self.assertRaisesRegex(RuntimeError,'Frozen writer'):source_target_loss(self.model,self.item)

    def test_positive_source_difference_is_not_counted_as_correct_switch(self):
        result=dict(rows=[dict(pair='p',native=dict(margin=2.,swapped_margin=1.)),
            dict(pair='p',native=dict(margin=3.,swapped_margin=-2.)),
            dict(pair='q',native=dict(margin=-1.,swapped_margin=-2.)),
            dict(pair='q',native=dict(margin=0.,swapped_margin=0.))])
        self.assertEqual(decision_summary(result),dict(normal_correct=2,donor_correct=2,
            source_switch_correct=1,normal_complete_pairs=1,four_decision_pairs=0))


if __name__=='__main__':unittest.main()
