import unittest

from compare_sampling import validate_inputs, validate_messages
from chat_probe import conversation_diagnostics
from sample import diagnostics, grams


class SamplingTests(unittest.TestCase):
    def test_hidden_special_markers_are_detected_without_changing_word_statistics(self):
        text = 'The moon opened a bakery. The moon opened a bakery.'
        old = diagnostics(text, set(grams(text, 8)))
        self.assertFalse(old['thinking_marker_in_output'])
        for marker in ('<think>', '</think>'):
            raw = marker + '\n' + text + '<|im_end|>'
            measured = diagnostics(text, set(grams(text, 8)), raw_text=raw)
            self.assertTrue(measured['thinking_marker_in_output'])
            self.assertEqual(measured['repeated_4gram_fraction'], old['repeated_4gram_fraction'])
            self.assertEqual(measured['training_8gram_fraction'], old['training_8gram_fraction'])
            history = [dict(role='assistant', content=text)]
            self.assertTrue(conversation_diagnostics(text, history, raw_text=raw)['thinking_marker_in_output'])
        self.assertFalse(diagnostics(text,set(),raw_text=text+'<|im_end|>')['thinking_marker_in_output'])
        self.assertTrue(diagnostics('</think>'+text,set())['thinking_marker_in_output'])

    def test_empty_model_reply_is_preserved_for_next_turn_but_user_must_have_text(self):
        history = [dict(role='user',content='First'),dict(role='assistant',content='\n'),
                   dict(role='user',content='Continue')]
        messages = validate_messages(history)
        self.assertEqual(messages[2],history[1])
        self.assertEqual(messages[-1]['content'],'Continue')
        with self.assertRaises(ValueError):
            validate_messages(history+[dict(role='assistant',content='ok'),dict(role='user',content=' ')])

    def test_histories_are_independent_and_roles_are_checked(self):
        source = [dict(role='user', content='One')]
        a, b = validate_messages(source), validate_messages(source)
        a.append(dict(role='assistant', content='Different branch'))
        self.assertEqual(len(b), 2)
        self.assertEqual(len(source), 1)
        with self.assertRaises(ValueError):
            validate_messages(a)
        with self.assertRaises(ValueError):
            validate_messages([dict(role='assistant', content='Wrong role')])

    def test_sampling_inputs_reject_duplicate_ids_and_invalid_values(self):
        cases = [dict(id='x', language='en', user_turns=['One', 'Two'])]
        config = dict(id='t1', sampling=dict(temperature=1., top_p=1., top_k=20, presence_penalty=2.))
        validate_inputs(cases, [config], 'dialogue')
        with self.assertRaises(ValueError):
            validate_inputs(cases * 2, [config], 'dialogue')
        for value in [0, float('nan'), True]:
            bad = dict(config, sampling=dict(config['sampling'], temperature=value))
            with self.assertRaises(ValueError):
                validate_inputs(cases, [bad], 'dialogue')


if __name__ == '__main__':
    unittest.main()
