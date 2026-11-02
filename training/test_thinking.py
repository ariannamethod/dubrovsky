import unittest
from thinking_probe import split_generation


class ThinkingTests(unittest.TestCase):
    def test_closure_is_not_inferred_from_eos_or_stripped_output(self):
        class Tokenizer:
            def convert_tokens_to_ids(self, token):
                return 2
            def decode(self, ids, **kwargs):
                return ''.join({1:'\n',2:'</think>',3:'answer',4:'argument',9:'<end>'}[i]for i in ids)
        t = Tokenizer()
        empty = split_generation(t,[1,2,1,3,9],True,[9])
        self.assertTrue(empty['empty_thinking_closed'])
        self.assertEqual(empty['answer'],'answer')
        self.assertIn('</think>',empty['raw_text'])
        nonempty = split_generation(t,[4,2,3,9],True,[9])
        self.assertFalse(nonempty['empty_thinking_closed'])
        unclosed = split_generation(t,[4,9],True,[9])
        self.assertTrue(unclosed['eos_reached'])
        self.assertFalse(unclosed['thinking_end_generated'])
        self.assertEqual(unclosed['answer'],'')
        capped = split_generation(t,[4,4],True,[9])
        self.assertFalse(capped['eos_reached'])
        self.assertEqual(capped['thinking_tokens'],2)
        self.assertEqual(split_generation(t,[3,9],False,[9])['answer'],'answer')
        duplicate = split_generation(t,[1,2,1,2,3,9],True,[9])
        self.assertTrue(duplicate['empty_thinking_closed'])
        self.assertTrue(duplicate['thinking_marker_in_final_answer'])
        self.assertFalse(duplicate['empty_final_answer'])
        self.assertTrue(split_generation(t,[1,9],False,[9])['empty_final_answer'])


if __name__ == '__main__':
    unittest.main()
