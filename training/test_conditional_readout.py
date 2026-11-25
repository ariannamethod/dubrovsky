import unittest
import torch
from conditional_readout import source_pair_loss, conditional_gate
from output_geometry import correction_bound, output_basis, bf16_rounding_envelope


class ConditionalTests(unittest.TestCase):
    def test_task_rejects_common_shift_and_learns_source_direction(self):
        own=torch.tensor(3.,requires_grad=True); donor=torch.tensor(2.,requires_grad=True)
        task,_=source_pair_loss(own,donor,own.detach(),donor.detach(),centered=True)
        a,b=torch.autograd.grad(task,(own,donor))
        self.assertAlmostEqual(float(a+b),0.,places=7); self.assertLess(float(a-b),0.)
        shifted,_=source_pair_loss(own+7,donor+7,own.detach(),donor.detach(),centered=True)
        self.assertEqual(float(task),float(shifted))

    def test_gate_requires_conditional_repair_and_beats_gain_only(self):
        def result(a,b):
            return {'rows':[dict(id=str(i),pair='p',native=dict(margin=a,swapped_margin=b)) for i in range(2)]}
        parent=result(3,2); good=result(5,-1)
        self.assertFalse(conditional_gate(parent,result(.5,-.5),[])['pass_gate'])
        self.assertTrue(conditional_gate(parent,good,[])['pass_gate'])
        self.assertFalse(conditional_gate(parent,good,[good])['pass_gate'])

    def test_mixed_subspace_includes_cancellation_between_experts(self):
        o=torch.tensor([[[1.,10.]],[[1.,-10.]]]);w=torch.tensor([1.,0.])
        bound=correction_bound(o,w,.1,1)
        self.assertEqual(bound['rank'],2)
        self.assertAlmostEqual(bound['bound'],.1*2**.5)
        individual=correction_bound(o[:1],w,.1,1)['bound']
        self.assertGreater(bound['bound'],individual*10)

    def test_actual_and_conditional_repairs_must_be_the_same_pair(self):
        def result(p,q):
            return {'rows':[dict(id=pair+str(i),pair=pair,native=dict(margin=v[0],swapped_margin=v[1]))
                            for pair,v in [('p',p),('q',q)] for i in range(2)]}
        gate=conditional_gate(result((3,2),(-1,1)),result((1,-1),(7,3)),[])
        self.assertEqual(gate['actual_new_pairs'],['p'])
        self.assertEqual(gate['conditional_new_pairs'],['q'])
        self.assertFalse(gate['pass_gate'])

    def test_bound_covers_random_working_states_and_native_rounding(self):
        torch.manual_seed(36); torch.set_num_threads(1)
        o=torch.randn(4,4,32)*.1; rows=torch.randn(2,32).bfloat16().float();w=rows[0]-rows[1]
        b=correction_bound(o,w,.1,4);raw=torch.randn(32).bfloat16().float()
        for _ in range(40):
            z=torch.rand(4,4)*2-1;votes=torch.randn(4).softmax(0)
            d=torch.einsum('e,er,erd->d',votes,z,o);d*=min(1.,.1/float((d.square().mean()+1e-12).sqrt()));d*=4
            self.assertLessEqual(abs(float(d@w)),b['bound']+1e-6)
            h=(raw.bfloat16()+d.bfloat16()).bfloat16()
            native=(rows.bfloat16()@h).float(); margin=native[0]-native[1]
            exact=(raw.double()+d.double())@w.double()
            self.assertLessEqual(abs(float(margin-exact)),bf16_rounding_envelope(raw,rows,4*.1*32**.5))


if __name__=='__main__':unittest.main()
