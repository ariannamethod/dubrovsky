import unittest
from prepare_corpus import normalize, parse, split_pairs
from sft import encode, chat_eos
from types import SimpleNamespace


class CorpusTests(unittest.TestCase):
    def test_chat_turn_stop_keeps_model_stop(self):
        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=248044))
        tokenizer = SimpleNamespace(eos_token_id=248046)
        self.assertEqual(chat_eos(model, tokenizer), [248044, 248046])

    def test_parser_preserves_voice(self):
        pairs = parse('Q: Who?\nA: I misplaced Tuesday.\n\nQ: Where?\nA: Dubrovsky kept it.\n')
        self.assertEqual(pairs, [('Who?', 'I misplaced Tuesday.'), ('Where?', 'Dubrovsky kept it.')])
        with self.assertRaises(ValueError):
            parse('lost preface\nQ: Who?\nA: I am.\n')

    def test_no_duplicate_question_or_answer_leakage(self):
        pairs = [('Q1', 'A1'), ('q1!', 'A2'), ('Q2', 'A2'), ('Q3', 'A3'), ('Q3', 'A3')]
        train, val = split_pairs(pairs, fraction=0.3)
        self.assertEqual(set(train + val), set(pairs))
        self.assertEqual(len(train + val), len(set(pairs)))
        for side in (0, 1):
            self.assertFalse({normalize(p[side]) for p in train} & {normalize(p[side]) for p in val})

    def test_mask_prefix_and_reject_template_mismatch(self):
        class Tokenizer:
            mismatch = False
            def apply_chat_template(self, messages, **kwargs):
                self_enable = kwargs['enable_thinking']
                if self_enable:
                    raise AssertionError('Thinking must be disabled')
                if kwargs['add_generation_prompt']:
                    return [1, 2, 3]
                return [1, 2, 99 if self.mismatch else 3, 4, 5]
        t = Tokenizer()
        row = {'id': 'test', 'question': 'Who?', 'answer': 'I am.'}
        self.assertEqual(encode(t, row, 10)['labels'], [-100, -100, -100, 4, 5])
        with self.assertRaises(ValueError):
            encode(t, row, 4)
        t.mismatch = True
        with self.assertRaises(ValueError):
            encode(t, row, 10)


if __name__ == '__main__':
    unittest.main()
