import unittest
import numpy as np

from weight_geometry import bf16_bits
from weight_probe import make_variant, dialogue_rows


class WeightProbeTests(unittest.TestCase):
    def setUp(self):
        self.base = {'weight': bf16_bits(np.arange(12, dtype=np.float32).reshape(3, 4))}
        self.layers = [dict(name='weight', index=15, direction=[1., 0., 0.], columns=[1, 3])]

    def test_each_variant_starts_from_pristine_parent(self):
        before = self.base['weight'].copy()
        edited = make_variant(self.base, self.layers, dict(kind='projection', alpha=1))
        restored = make_variant(self.base, self.layers, dict(kind='baseline'))
        np.testing.assert_array_equal(self.base['weight'], before)
        np.testing.assert_array_equal(restored['weight'], before)
        np.testing.assert_array_equal(edited['weight'][:, [0, 2]], before[:, [0, 2]])
        self.assertEqual(edited['weight'][0, 1], 0)
        self.assertEqual(edited['weight'][0, 3], 0)
        restored['weight'][0, 0] = 1
        np.testing.assert_array_equal(self.base['weight'], before)

    def test_control_reproducible_and_confined_to_support(self):
        variant = dict(kind='orthogonal', alpha=1, seed=42)
        first = make_variant(self.base, self.layers, variant)['weight']
        second = make_variant(self.base, self.layers, variant)['weight']
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[:, [0, 2]], self.base['weight'][:, [0, 2]])
        self.assertFalse(np.array_equal(first, self.base['weight']))

    def test_unknown_variant_fails(self):
        with self.assertRaises(ValueError):
            make_variant(self.base, self.layers, dict(kind='unknown'))

    def test_single_layer_edit_leaves_other_tensor_exact(self):
        base = dict(self.base, other=self.base['weight'].copy())
        layers = self.layers + [dict(self.layers[0], name='other', index=19)]
        result = make_variant(base, layers, dict(kind='projection', alpha=1, layers=[15]))
        np.testing.assert_array_equal(result['other'], base['other'])
        self.assertFalse(np.array_equal(result['weight'], base['weight']))
        for active in [[], [15, 15], [17]]:
            with self.subTest(active=active), self.assertRaises(ValueError):
                make_variant(base, layers, dict(kind='projection', alpha=1, layers=active))

    def test_dialogue_preserves_own_history_and_resets_between_seeds(self):
        case = dict(id='fresh', language='en', user_turns=['Open the scene.', 'Change topic.'])
        calls = []
        def generate(value, seed):
            calls.append((value['messages'], seed))
            return dict(text=f'actual answer {seed}')
        first = list(dialogue_rows(case, 42, generate))
        second = list(dialogue_rows(case, 137, generate))
        self.assertEqual([r['turn_seed'] for r in first], [42, 43])
        self.assertEqual(first[1]['messages'][-2], dict(role='assistant', content='actual answer 42'))
        self.assertEqual(len(second[0]['messages']), 2)
        self.assertEqual(second[1]['messages'][-2]['content'], 'actual answer 137')
        self.assertEqual(len(first[0]['messages']), 2)
        self.assertTrue(all(r['messages'][-1]['role']=='user' for r in first+second))


if __name__ == '__main__':
    unittest.main()
