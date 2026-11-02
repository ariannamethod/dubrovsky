import hashlib
import unittest
from curate_corpus import curate
from build_mixture import validate_splits


def pair(question, answer):
    return dict(id=hashlib.sha256((question+'\0'+answer).encode()).hexdigest(),
                question=question, answer=answer)


class CurationTests(unittest.TestCase):
    def test_replay_cannot_leak_into_development_split(self):
        a=pair('Who?', 'I misplaced Tuesday.')
        b=pair('who!', 'Another answer.')
        with self.assertRaises(ValueError):
            validate_splits([a],[b])
        validate_splits([a],[pair('Where?', 'Dubrovsky kept it.')])

    def test_exclusions_follow_translations_without_rewriting_or_resplitting(self):
        old = pair('How large?', 'Obsolete engine biography')
        retained = pair('Who?', 'I am a forgotten parameter asking for directions.')
        held_out = pair('Where?', 'Dubrovsky misplaced Tuesday.')
        splits = {'train':[old,retained], 'validation':[held_out]}
        translations = [dict(source_id=r['id'], language='ru', split=s,
                             question='Кто?', answer='Текст '+r['answer'])
                        for s,rows in splits.items() for r in rows]
        kept, translated, audit = curate(splits, translations+[translations[-1]],
                                        {old['id']:'obsolete biography'})
        self.assertEqual(kept, {'train':[retained], 'validation':[held_out]})
        self.assertEqual([r['source_id'] for r in translated], [retained['id'],held_out['id']])
        self.assertEqual(translated[-1]['split'], 'validation')
        self.assertEqual(audit['overlapping_translation_variants_skipped'], 1)
        self.assertEqual(len(splits['train']), 2)
        with self.assertRaises(ValueError):
            curate(splits, [dict(translations[-1], split='train')], {})
        with self.assertRaises(ValueError):
            curate(splits, [], {'nonexistent':'typo'})


if __name__ == '__main__':
    unittest.main()
