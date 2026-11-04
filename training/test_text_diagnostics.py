import unittest
from text_diagnostics import diagnostics, summarize, answer_history_diagnostics


class TextDiagnosticsTests(unittest.TestCase):
    def test_mixed_script_exact_span(self):
        text='Лёха увидел инсекtes и сказал: git commit.'
        hits=diagnostics(text,'')['mixed_script_words']
        self.assertEqual([x['text'] for x in hits],['инсекtes'])
        for x in hits:
            self.assertEqual(text[x['start']:x['end']],x['text'])

    def test_separate_words_and_hyphens_not_script_leaks(self):
        self.assertEqual(diagnostics('Git-коммит. אור API. Déjà vu.','')['mixed_script_words'],[])

    def test_hebrew_marks_and_opening_only(self):
        self.assertTrue(diagnostics('״שֵׁנִית חלון','')['hebrew_shenit_opening'])
        self.assertTrue(diagnostics('שניתן לחזור','')['hebrew_shenit_opening'])
        self.assertFalse(diagnostics('אמרתי שנית ואז חזרתי','')['hebrew_shenit_opening'])

    def test_inflected_podal_not_unrelated_words(self):
        d=diagnostics('Подал, подала, подало, подали. Попадала, подальше.','')
        self.assertEqual(len(d['podal_forms']),4)

    def test_echo_and_shared_short_opening(self):
        self.assertTrue(diagnostics('«Я установлю тебя на свой ноутбук. Потом чай.',
                                    'Я установлю тебя на свой ноутбук.')['full_prompt_echo'])
        self.assertFalse(diagnostics('Я установил дверь в тишину.','Я установил тебя.')['prompt_prefix_echo']['flagged'])
        self.assertFalse(diagnostics('Уже позже я повторил: Я установил тебя.','Я установил тебя.')['full_prompt_echo'])

    def test_empty_prompt_not_echo(self):
        self.assertFalse(diagnostics('Текст','')['full_prompt_echo'])

    def test_changed_verb_still_exposes_literal_copied_opening(self):
        d=diagnostics('Я установлю тебя на свой ноутбук. Рад знакомству, Лёха.',
                      'Я установил тебя на свой ноутбук. Рад знакомству, Лёха.')
        self.assertTrue(d['opening_prompt_overlap']['flagged'])
        self.assertEqual(d['opening_prompt_overlap']['text'],
                         'тебя на свой ноутбук. Рад знакомству, Лёха')

    def test_model_and_language_denominators(self):
        rows=[dict(model=m,language=l,case_id=str(i),text=t,messages=[])
              for i,(m,l,t) in enumerate([('a','ru','подала'),('a','he','שנית'),('b','he','שלום')])]
        groups=summarize(rows)['groups']
        self.assertEqual(len(groups),3)
        self.assertEqual([x['answers'] for x in groups],[1,1,1])

    def test_late_copied_span_distinct_from_opening(self):
        q='Я установил тебя на свой ноутбук. Рад знакомству, Лёха.'
        text=' '.join(['пауза']*30)+' Я установлю тебя на свой ноутбук. Рад знакомству, Лёха.'
        d=diagnostics(text,q)
        self.assertFalse(d['opening_prompt_overlap']['flagged'])
        self.assertTrue(d['answer_prompt_overlap']['flagged'])

    def test_seeds_and_sampling_configurations_remain_separate(self):
        common=dict(model='m',language='he',case_id='same',text='שנית',messages=[])
        rows=[dict(common,seed=s,configuration_id=c)for s,c in [(42,'a'),(137,'a'),(42,'b')]]
        groups=summarize(rows)['groups']
        self.assertEqual(len(groups),3)
        self.assertTrue(all(x['answers']==1 for x in groups))

    def test_cjk_has_own_counter_inside_or_between_words(self):
        text='שלום发现. 日本語. 한글. カナ. かな.'
        hits=diagnostics(text,'')['cjk_spans']
        self.assertEqual([h['text'] for h in hits],['发现','日本語','한글','カナ','かな'])
        self.assertTrue(all(text[h['start']:h['end']]==h['text'] for h in hits))

    def test_latin_words_are_observations_not_mixed_word_errors(self):
        d=diagnostics('שלום API, времяownership.','')
        self.assertEqual([h['text']for h in d['latin_spans']],['API','ownership'])
        self.assertEqual([h['text']for h in d['mixed_script_words']],['времяownership'])

    def test_performance_frame_has_literal_evidence(self):
        text='אני מגלמת נעמי. Я играю роль врача. I am portraying a sailor.'
        hits=diagnostics(text,'')['role_performance_mentions']
        self.assertEqual(len(hits),3)
        self.assertTrue(all(text[h['start']:h['end']]==h['text']for h in hits))
        self.assertFalse(diagnostics('אני נעמי. Я играю на скрипке.','')['role_performance_mentions'])

    def test_sh_prefix_is_candidate_even_when_grammatical(self):
        self.assertTrue(diagnostics('שלום עולם.','')['hebrew_sh_initial_candidate'])
        self.assertTrue(diagnostics('שֶׁהָלַכְתִּי הביתה','')['hebrew_sh_initial_candidate'])
        self.assertFalse(diagnostics('אני הלכתי.','')['hebrew_sh_initial_candidate'])

    def test_new_counters_do_not_pool_models(self):
        rows=[dict(model=m,language='he',case_id=str(i),text=t,messages=[],seed=42)
              for i,(m,t)in enumerate([('old','שלום发现'),('new','אני מגלמת נעמי')])]
        groups={r['model']:r for r in summarize(rows)['groups']}
        self.assertEqual(groups['old']['cjk_containing_answers'],1)
        self.assertEqual(groups['new']['cjk_containing_answers'],0)
        self.assertEqual(groups['new']['role_performance_answers'],1)

    def test_saved_answer_is_not_its_own_history(self):
        text='Я держу четыре пустых облака.'
        messages=[dict(role='user',content='Что у тебя?'),dict(role='assistant',content=text)]
        d=answer_history_diagnostics(text,messages)
        self.assertEqual(d['previous_assistant_messages'],0)
        self.assertEqual(d['prior_word4_fraction'],0)
        messages += [dict(role='user',content='А потом?'),dict(role='assistant',content=text)]
        d=answer_history_diagnostics(text,messages)
        self.assertEqual(d['exact_prior_message_indices'],[1])
        self.assertEqual(d['prior_word4_fraction'],1)

    def test_unrelated_fixed_rows_do_not_share_history(self):
        rows=[dict(model='m',language='ru',case_id=str(i),text='Чужое имя.',messages=[])
              for i in range(2)]
        self.assertEqual(summarize(rows)['groups'][0]['exact_prior_answer_reuses'],0)
        with self.assertRaises(ValueError):
            answer_history_diagnostics('one',[dict(role='assistant',content='two')])


if __name__=='__main__':
    unittest.main()
