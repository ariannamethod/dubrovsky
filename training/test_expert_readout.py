import unittest
import torch

from embedded_experts import EmbeddedExperts, ExpertConfig
from probe_expert_readout import BankIntervention, question_bounds


class ReadoutInterventionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(27)
        self.block = EmbeddedExperts(ExpertConfig(16,rank=4))
        with torch.no_grad():
            self.block.output.normal_(0,.02)
        self.raw = torch.randn(1,16,16)
        self.source = torch.randn_like(self.raw)
        self.mask = torch.tensor([[False,True,True,False,False,False,False,False,
                                   True,True,True,False,False,False,False,False]])

    def run_block(self, block, start=0, stop=16, state=None):
        return block(self.raw[:,start:stop], self.mask[:,start:stop], state,
                     memory_hidden=self.source[:,start:stop])

    def test_native_question_bounds(self):
        self.assertEqual(question_bounds(self.mask),(8,11))
        with self.assertRaises(ValueError):
            question_bounds(torch.zeros_like(self.mask))

    def test_same_bank_replacement_reproduces_every_token(self):
        baseline, state, votes = self.run_block(self.block)
        for cut in (8,11):
            _,before,_ = self.run_block(self.block,stop=cut)
            wrapper = BankIntervention(self.block,cut,bank=before.user.clone())
            actual,last,actual_votes = self.run_block(wrapper)
            self.assertTrue(torch.equal(baseline,actual))
            self.assertTrue(torch.equal(state.user,last.user))
            self.assertTrue(torch.equal(state.working,last.working))
            self.assertTrue(torch.equal(votes,actual_votes))

    def test_changed_bank_preserves_prefix_and_working_at_boundary(self):
        baseline,_,_ = self.run_block(self.block)
        _,before,_ = self.run_block(self.block,stop=8)
        bank = torch.ones_like(before.user)*.6
        wrapper = BankIntervention(self.block,8,bank=bank)
        actual,_,_ = self.run_block(wrapper)
        self.assertTrue(torch.equal(actual[:,:8],baseline[:,:8]))
        self.assertFalse(torch.equal(actual[:,8:],baseline[:,8:]))
        self.assertEqual(wrapper.capture['working_before'],before.working.tolist())
        self.assertEqual(wrapper.capture['incoming'],bank.tolist())

    def test_hold_question_changes_writes_but_not_working_updates(self):
        _,before,_ = self.run_block(self.block,stop=8)
        wrapper=BankIntervention(self.block,8,hold=True)
        _,after,_=self.run_block(wrapper)
        self.assertTrue(torch.equal(before.user,after.user))
        self.assertFalse(torch.equal(before.working,after.working))
        self.assertEqual(after.positions,16)

    def test_cached_assistant_continuation_does_not_reinject(self):
        _,before,_ = self.run_block(self.block,stop=8)
        wrapper=BankIntervention(self.block,8,bank=torch.ones_like(before.user)*.6)
        first,state,_=self.run_block(wrapper,stop=13)
        rest,last,_=self.run_block(wrapper,start=13,state=state)
        full,full_state,_=self.run_block(wrapper)
        self.assertTrue(torch.equal(torch.cat((first,rest),1),full))
        self.assertTrue(torch.equal(last.working,full_state.working))

    def test_replacement_is_read_only(self):
        params={k:v.clone() for k,v in self.block.state_dict().items()}
        raw,source,mask=self.raw.clone(),self.source.clone(),self.mask.clone()
        bank=torch.ones(1,4,4)
        self.run_block(BankIntervention(self.block,8,bank=bank))
        self.assertTrue(torch.equal(raw,self.raw))
        self.assertTrue(torch.equal(source,self.source))
        self.assertTrue(torch.equal(mask,self.mask))
        self.assertTrue(torch.equal(bank,torch.ones_like(bank)))
        self.assertTrue(all(torch.equal(v,params[k]) for k,v in self.block.state_dict().items()))

    def test_invalid_intervention_fails(self):
        with self.assertRaises(ValueError):BankIntervention(self.block,8)
        with self.assertRaises(ValueError):BankIntervention(self.block,8,hold=True,bank=torch.zeros(1,4,4))
        with self.assertRaises(ValueError):self.run_block(BankIntervention(self.block,16,hold=True))
        with self.assertRaises(ValueError):self.run_block(BankIntervention(self.block,8,bank=torch.zeros(1,4,3)))


if __name__=='__main__':
    unittest.main()
