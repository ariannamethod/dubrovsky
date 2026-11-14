import unittest
from expert_data import validate_expert_splits, encode_expert_prompt, encode_expert_row


class OffsetTokenizer:
    """Deterministic offset fixture; the real Qwen tokenizer is audited by CLI."""
    is_fast=True
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False,**kwargs):
        s=''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in messages)
        if add_generation_prompt:s+='<|im_start|>assistant\n'
        return list(map(ord,s)) if tokenize else s
    def __call__(self,s,**kwargs):
        return dict(input_ids=list(map(ord,s)),offset_mapping=[(i,i+1) for i in range(len(s))])


class IsolatedEncodingTests(unittest.TestCase):
    def test_assistant_text_and_length_never_enter_source(self):
        t=OffsetTokenizer()
        a=dict(question='Who owns it?',history=[dict(role='user',content='Anna owns it.'),
            dict(role='assistant',content='Assistant-only invention.')])
        b=dict(a,history=[a['history'][0],dict(role='assistant',content='Another much longer assistant invention.')])
        ea,eb=[encode_expert_prompt(t,r,3072) for r in (a,b)]
        self.assertNotEqual(ea['input_ids'],eb['input_ids'])
        self.assertEqual(ea['memory_input_ids'],eb['memory_input_ids'])
        self.assertEqual(ea['memory_user_mask'],eb['memory_user_mask'])
        text=''.join(map(chr,ea['memory_input_ids']))
        self.assertNotIn('assistant',text);self.assertNotIn('system',text)

    def test_answer_and_later_question_do_not_rewrite_source_prefix(self):
        t=OffsetTokenizer();row=dict(id='one',question='First fact.',answer='A reply.')
        first=encode_expert_row(t,row,3072)
        other=encode_expert_row(t,dict(row,answer='A different answer.'),3072)
        self.assertEqual(first['memory_input_ids'],other['memory_input_ids'])
        follow=encode_expert_prompt(t,dict(question='Next fact.',history=[
            dict(role='user',content=row['question']),dict(role='assistant',content=row['answer'])]),3072)
        self.assertEqual(first['memory_input_ids'],follow['memory_input_ids'][:len(first['memory_input_ids'])])
        for item in (first,other,follow):
            self.assertEqual([x for x,m in zip(item['input_ids'],item['user_mask']) if m],
                [x for x,m in zip(item['memory_input_ids'],item['memory_user_mask']) if m])


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
