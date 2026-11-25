import unittest
import torch
from embedded_experts import EmbeddedExperts, ExpertConfig
from probe_bank_trace import checked_trace, trace_writer, user_segments


class BankTraceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(35)
        self.block = EmbeddedExperts(ExpertConfig(12,rank=3,format_version=3,max_output_gain=8))
        with torch.no_grad():
            self.block.retain.normal_(0,.2); self.block.write.normal_(0,.2)
        self.raw = torch.randn(1,15,12); self.source = torch.randn_like(self.raw)
        self.mask = torch.tensor([[False,True,True,False,False,True,True,True,False,
                                  False,True,True,True,False,False]])

    def test_trace_matches_native_at_every_message_without_mutation(self):
        before = {k:v.clone() for k,v in self.block.state_dict().items()}
        trace = checked_trace(self.block,self.raw,self.mask,self.source)
        self.assertEqual(trace['spans'],[[1,3],[5,8],[10,13]])
        self.assertEqual(len(trace['tokens']),8)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in self.block.state_dict().items()))

    def test_final_question_continuation_equals_full_trace(self):
        whole = trace_writer(self.block,self.source,self.mask)
        bank = whole['banks'][-2].clone()
        tail = trace_writer(self.block,self.source[:,10:13],self.mask[:,10:13],bank)
        self.assertTrue(torch.equal(tail['banks'][-1],whole['banks'][-1]))
        self.assertTrue(torch.equal(bank,whole['banks'][-2]))

    def test_user_bank_ignores_unmasked_features_and_working_input(self):
        expected = checked_trace(self.block,self.raw,self.mask,self.source)
        source = self.source.clone(); source[~self.mask] += 200
        actual = checked_trace(self.block,self.raw*100,self.mask,source)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(expected['banks'],actual['banks'])))

    def test_invalid_masks_fail(self):
        for mask in (torch.zeros(1,3,dtype=torch.bool),torch.ones(2,3,dtype=torch.bool),torch.ones(1,3)):
            with self.assertRaises(ValueError):user_segments(mask)


if __name__=='__main__':unittest.main()
