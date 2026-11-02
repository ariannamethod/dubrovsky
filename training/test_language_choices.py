import unittest

from score_language_choices import classify, score


class LanguageChoicesTest(unittest.TestCase):
    def test_accents_case_and_boundary_punctuation(self):
        self.assertEqual(classify(' « REÇUES. » ', 'reçues', 'reçu'), 'correct_exact')
        self.assertEqual(classify('«закрытой»', 'закрытой', 'закрытую'), 'correct_exact')
        self.assertEqual(classify('כתבה.', 'כתבה', 'כתב'), 'correct_exact')

    def test_whole_words_and_explanations(self):
        self.assertEqual(classify('The answer is opens.', 'opens', 'open'), 'correct_mentioned_only')
        self.assertEqual(classify('Choose opens, not open.', 'opens', 'open'), 'both_mentioned')
        self.assertEqual(classify('A door becomes an orchid.', 'opens', 'open'), 'neither_mentioned')

    def test_overlapping_options_still_have_exact_answers(self):
        self.assertEqual(classify('de votre', 'de votre', 'votre'), 'correct_exact')
        self.assertEqual(classify('votre', 'de votre', 'votre'), 'rejected_exact')

    def test_generation_identity_and_language_are_checked(self):
        row = dict(model='model', case_id='q', configuration_id='p0', seed=42, turn=1,
                   language='en', text='opens', eos_reached=True)
        refs = [dict(id='q', language='en', answer='opens', rejected='open')]
        result = score([row], refs)
        self.assertEqual(result['groups'][0]['correct_exact'], 1)
        with self.assertRaisesRegex(ValueError, 'Duplicate generation'):
            score([row, row], refs)
        with self.assertRaisesRegex(ValueError, 'Language mismatch'):
            score([dict(row, language='fr')], refs)


if __name__ == '__main__':
    unittest.main()
