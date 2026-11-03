import unittest
from dpo import encode_preference, validate_holdout
from sft import encode


class TextTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
        text = ''.join('<'+m['role']+'>'+m['content']+'</turn>' for m in messages)
        if add_generation_prompt:
            text += '<assistant>'
        return [ord(c) for c in text]


class PreferenceTests(unittest.TestCase):
    def test_validation_cannot_reuse_training_context_with_new_answers(self):
        row = dict(question='Who owns it?', chosen='Ada.', rejected='Otto.')
        with self.assertRaisesRegex(ValueError, 'held-out'):
            validate_holdout([row], [dict(row, chosen='Ada owns it.', rejected='Otto owns it.')])
        validate_holdout([row], [dict(row, question='What color was it?')])
        validate_holdout([row], [])

    def test_both_answers_keep_history_masked_and_share_the_same_prefix(self):
        row = dict(question='What color was it?', chosen='Green.', rejected='Red.',
                   history=[dict(role='user', content='The parcel is green.'),
                            dict(role='assistant', content='It is next to the door.')])
        chosen, rejected = encode_preference(TextTokenizer(), row, 1024)
        prefixes = []
        for encoded, answer in zip((chosen, rejected), ('Green.', 'Red.')):
            start = next(i for i, x in enumerate(encoded['labels']) if x != -100)
            prefixes.append(encoded['input_ids'][:start])
            self.assertEqual(encoded['labels'][:start], [-100]*start)
            self.assertEqual(''.join(map(chr, encoded['labels'][start:])), answer+'</turn>')
            prefix = ''.join(map(chr, prefixes[-1]))
            self.assertIn('The parcel is green.', prefix)
            self.assertIn('It is next to the door.', prefix)
        self.assertEqual(prefixes[0], prefixes[1])
        with self.assertRaisesRegex(ValueError, 'no silent truncation'):
            encode_preference(TextTokenizer(), row, len(chosen['input_ids'])-1)
        with self.assertRaisesRegex(ValueError, 'complete user/assistant'):
            encode_preference(TextTokenizer(), dict(row, history=row['history'][:1]), 1024)

    def test_single_turn_preferences_remain_identical(self):
        row = dict(id='one', question='Who?', chosen='Dubrovsky.', rejected='Someone else.')
        actual = encode_preference(TextTokenizer(), row, 1024)
        expected = [encode(TextTokenizer(), dict(id='one', question=row['question'], answer=row[key]), 1024)
                    for key in ('chosen','rejected')]
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
