import unittest
from expert_data import validate_expert_splits


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.train = [dict(id='t-'+lang, question='train '+lang, language=lang, family='train')
                      for lang in ('en', 'ru')]
        self.val = [dict(id='v-'+lang, question='validation '+lang, language=lang, family='val')
                    for lang in ('en', 'ru')]

    def test_valid_bilingual_splits(self):
        validate_expert_splits(self.train, self.val)

    def test_counterfactual_family_cannot_leak_across_splits(self):
        self.val[0]['family'] = 'train'
        with self.assertRaisesRegex(ValueError, 'families'):
            validate_expert_splits(self.train, self.val)

    def test_same_question_and_history_cannot_leak(self):
        self.val[0]['question'] = self.train[0]['question']
        with self.assertRaisesRegex(ValueError, 'prefix'):
            validate_expert_splits(self.train, self.val)

    def test_language_must_not_disappear(self):
        self.val[0]['language'] = 'ru'
        with self.assertRaisesRegex(ValueError, 'both English and Russian'):
            validate_expert_splits(self.train, self.val)


if __name__ == '__main__':
    unittest.main()
