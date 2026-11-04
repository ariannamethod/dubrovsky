import copy
import unittest

from summarize_weight_probe import summarize
from compare_sampling import validate_messages


class WeightSummaryTests(unittest.TestCase):
    def setUp(self):
        target = dict(id='checkpoint', variant_sha256={'baseline': {'weight': 'original'}})
        self.spec = dict(variants=[dict(id='baseline')], seeds=[42, 137],
            targets=[target],
            generations=[dict(id='early', language='he', seeds=[55, 150]),
                         dict(id='late', language='he', seeds=[57, 152])])
        self.rows = [dict(kind='provenance', spec_sha256='test', target=target,
                          unchanged_parameter_sha256='other-parameters')]
        for case in self.spec['generations']:
            case['messages'] = [dict(role='user', content=case['id'])]
            for slot, seed in enumerate(case['seeds']):
                self.rows.append(dict(kind='generation', variant='baseline', case_id=case['id'],
                    messages=validate_messages(case['messages']),
                    seed=seed, language='he', enable_thinking=False,
                    history_diagnostics=dict(exact_prior_message_indices=[2] if slot else [],
                                             prior_word4_fraction=float(slot)),
                    literal_diagnostics=dict(mixed_script_words=[], cjk_spans=[], role_performance_mentions=[]),
                    eos_reached=True, empty_thinking_closed=False, token_ids=[1, 2]))
        self.rows += [dict(kind='installed', variant=dict(id='baseline'), tensor_sha256={'weight': 'original'}),
                      dict(kind='restoration', edited_tensors_restored=True, other_parameters_unchanged=True,
                           unchanged_parameter_sha256='other-parameters'),
                      dict(kind='complete')]

    def test_pairs_turn_offset_seeds_in_same_slot(self):
        result = summarize(self.spec, self.rows, 'test')
        self.assertEqual([(r['seed_slot'], r['answers'], r['exact_copies']) for r in result['by_seed']],
                         [(0, 2, 0), (1, 2, 2)])

    def test_rejects_incomplete_duplicate_and_failed_integrity(self):
        broken = []
        broken.append(self.rows[:-1])
        broken.append(self.rows[:1] + self.rows[2:])
        broken.append(self.rows[:2] + [self.rows[1]] + self.rows[2:])
        failed = copy.deepcopy(self.rows); failed[-2]['other_parameters_unchanged'] = False
        broken.append(failed)
        changed = copy.deepcopy(self.rows); changed[-2]['unchanged_parameter_sha256'] = 'changed'
        broken.append(changed)
        changed = copy.deepcopy(self.rows); changed[-3]['tensor_sha256']['weight'] = 'wrong'
        broken.append(changed)
        for rows in broken:
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                summarize(self.spec, rows, 'test')
        with self.assertRaises(ValueError):
            summarize(self.spec, self.rows, 'different')

    def test_dialogue_requires_actual_own_history(self):
        spec = dict(self.spec, dialogues=[dict(id='chat', language='he', seeds=[42], user_turns=['One.', 'Two.'])])
        turns = []
        history = []
        for turn, question in enumerate(['One.', 'Two.'], 1):
            messages = validate_messages(history + [dict(role='user', content=question)])
            row = dict(self.rows[1], kind='dialogue', case_id='chat', seed=42, turn=turn,
                       turn_seed=41+turn, messages=messages, text=f'answer {turn}')
            turns.append(row)
            history = messages + [dict(role='assistant', content=row['text'])]
        rows = self.rows[:-2] + turns + self.rows[-2:]
        self.assertEqual(summarize(spec, rows, 'test')['dialogue_turns'], 2)
        bad = copy.deepcopy(rows)
        bad[-3]['messages'][-2]['content'] = 'answer from another variant'
        with self.assertRaises(ValueError):
            summarize(spec, bad, 'test')

    def test_fixed_history_accepts_implicit_system_but_rejects_changed_question(self):
        rows = copy.deepcopy(self.rows)
        rows[1]['messages'] = copy.deepcopy(self.spec['generations'][0]['messages'])
        self.assertEqual(summarize(self.spec, rows, 'test')['generations'], 4)
        rows[1]['messages'][-1]['content'] = 'a different question'
        with self.assertRaises(ValueError):
            summarize(self.spec, rows, 'test')


if __name__ == '__main__':
    unittest.main()
