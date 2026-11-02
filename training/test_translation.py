import unittest
from translate import response_text, validate, pending_rows


class TranslationTests(unittest.TestCase):
    def test_completed_translations_are_reused_per_language(self):
        rows=[dict(id='a',split='train'),dict(id='b',split='validation')]
        completed=[dict(source_id='a',language='ru',split='train',question='Кто?',answer='Дубровский.')]
        self.assertEqual(pending_rows(rows,completed,'ru'),rows[1:])
        self.assertEqual(pending_rows(rows,completed,'he'),rows)
        with self.assertRaises(ValueError):
            pending_rows(rows,[dict(completed[0],split='validation')],'ru')

    def test_mistyped_source_id_is_rejected(self):
        row=dict(source_id='abc',language='ru',question='Кто?',answer='Дубровский.')
        validate([row],[{'id':'abc'}],'ru')
        with self.assertRaises(ValueError):validate([row],[{'id':'abd'}],'ru')

    def test_reordered_and_wrong_language_records_are_rejected(self):
        a=dict(source_id='a',language='ru',question='Кто?',answer='Дубровский.')
        b=dict(a,source_id='b')
        with self.assertRaises(ValueError):validate([b,a],[{'id':'a'},{'id':'b'}],'ru')
        with self.assertRaises(ValueError):validate([a],[{'id':'a'}],'he')
        with self.assertRaises(ValueError):validate([dict(a,answer='Still English')],[{'id':'a'}],'ru')

    def test_incomplete_response_is_not_training_data(self):
        with self.assertRaises(ValueError):response_text({'status':'incomplete','output':[]})
        response={'status':'completed','output':[
            {'type':'reasoning'}, {'type':'message','role':'assistant','content':[
                {'type':'output_text','text':'one'}, {'type':'output_text','text':'two'}]}]}
        self.assertEqual(response_text(response),'onetwo')


if __name__=='__main__':unittest.main()
