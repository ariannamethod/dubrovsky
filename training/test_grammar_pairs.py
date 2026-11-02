import unittest
from score_grammar_pairs import preference, validate_pairs


class GrammarPairsTests(unittest.TestCase):
    def test_length_sensitive_ranking_is_not_hidden(self):
        result = preference(dict(tokens=3, mean_log_probability=-1),
                            dict(tokens=1, mean_log_probability=-2))
        self.assertFalse(result['preferred_by_sum'])
        self.assertTrue(result['preferred_by_mean'])
        self.assertTrue(result['ranking_disagrees'])
        self.assertFalse(result['equal_token_count'])

    def test_tie_and_equal_length_are_explicit(self):
        result = preference(dict(tokens=2, mean_log_probability=-1),
                            dict(tokens=2, mean_log_probability=-1))
        self.assertEqual(result['sum_margin'], 0)
        self.assertEqual(result['mean_margin'], 0)
        self.assertTrue(result['equal_token_count'])
        self.assertFalse(result['ranking_disagrees'])

    def test_malformed_contrast_cannot_enter_summary(self):
        row = dict(id='a', language='en', category='agreement', register='ordinary',
                   question='What happens?', answer='It moves.', rejected='It move.')
        validate_pairs([row])
        for invalid in [[], [row, row], [dict(row, rejected=row['answer'])], [dict(row, language='')]]:
            with self.assertRaises(ValueError):
                validate_pairs(invalid)
        with self.assertRaises(ValueError):
            preference(dict(tokens=0, mean_log_probability=-1), dict(tokens=2, mean_log_probability=-1))


if __name__ == '__main__':
    unittest.main()
