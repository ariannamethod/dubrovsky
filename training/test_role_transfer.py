import copy
import unittest

import torch

from probe_role_transfer import base_readout, transfer_pair, causal_alternatives
from test_embedded_experts import tiny_qwen


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.pair = dict(id='pair', role='owner', rows=[dict(owner='Ada',keeper='Ben',
            history=[dict(role='user',content='Ada owns it; Ada lends it to Ben.'),
                     dict(role='assistant',content='The bench interviewed the cloud.')],
            question='Recall the owner of the object.')])

    def test_factors_commute_and_do_not_mutate_source(self):
        source = copy.deepcopy(self.pair)
        a = transfer_pair(self.pair,names={'Ada':'Mira','Ben':'Nico'})
        a = transfer_pair(a,assistant_history=['The staircase employs the river.'],object_name='ticket punch')
        b = transfer_pair(self.pair,assistant_history=['The staircase employs the river.'],object_name='ticket punch')
        b = transfer_pair(b,names={'Ada':'Mira','Ben':'Nico'})
        self.assertEqual(a,b); self.assertEqual(self.pair,source)
        self.assertEqual(a['rows'][0]['owner'],'Mira')
        self.assertEqual(a['rows'][0]['keeper'],'Nico')

    def test_scene_keeps_user_facts_and_labels_exact(self):
        result=transfer_pair(self.pair,assistant_history=['A much longer history has no renamed people.'])
        for key in ('owner','keeper','question'):
            self.assertEqual(result['rows'][0][key],self.pair['rows'][0][key])
        self.assertEqual(result['rows'][0]['history'][0],self.pair['rows'][0]['history'][0])

    def test_name_replacement_is_simultaneous_and_word_bounded(self):
        self.pair['rows'][0]['history'][0]['content']='Ada Ben Adaptation'
        r=transfer_pair(self.pair,names={'Ada':'Ben','Ben':'Ada'})
        self.assertEqual(r['rows'][0]['history'][0]['content'],'Ben Ada Adaptation')

    def test_reject_collapsed_names_and_ambiguous_query(self):
        with self.assertRaises(ValueError): transfer_pair(self.pair,names={'Ada':'X','Ben':'X'})
        self.pair['rows'][0]['question']='Which item?'
        with self.assertRaises(ValueError): transfer_pair(self.pair,object_name='ticket')

    def test_base_wrapper_restores_after_exception(self):
        class Block(torch.nn.Module):
            def forward(self, x, enabled=True): return x+1 if enabled else x
        block=Block(); x=torch.tensor(1.)
        with self.assertRaises(RuntimeError):
            with base_readout(block,True):
                self.assertEqual(block(x,enabled=True),x)
                raise RuntimeError('test')
        self.assertEqual(block(x),x+1)
        self.assertNotIn('forward',block.__dict__)

    def test_alternative_lengths_share_exact_causal_prefix(self):
        torch.manual_seed(31)
        body=tiny_qwen().model
        sequences=[[2,3,4,5,6],[2,3,4,8,9,10]]
        with torch.no_grad():
            actual,common=causal_alternatives(body,sequences,'cpu')
            self.assertEqual(common,3)
            self.assertTrue(torch.equal(actual[0][:,:3],actual[1][:,:3]))
            # Each fork is also reproduced by its own independent cached rollout.
            for sequence,value in zip(sequences,actual):
                out=body(input_ids=torch.tensor([sequence[:3]]),use_cache=True,return_dict=True)
                pieces=[out.last_hidden_state]
                for token in sequence[3:]:
                    out=body(input_ids=torch.tensor([[token]]),past_key_values=out.past_key_values,use_cache=True,return_dict=True)
                    pieces.append(out.last_hidden_state)
                self.assertTrue(torch.equal(value,torch.cat(pieces,1)))


if __name__=='__main__': unittest.main()
