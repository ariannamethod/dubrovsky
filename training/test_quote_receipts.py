import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from quote_receipts import verify_quote


class QuoteReceiptsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.row = dict(model='weights/step-20', text='Лёха says: a licensed error.',
                        case_id='scene', turn=2, seed=42, sampling={'temperature': 0.8},
                        eos_reached=True)
        self.write_rows([self.row])

    def write_rows(self, rows):
        self.path = self.root / 'generations.jsonl'
        self.path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
        self.selection = dict(source_file=self.path.name,
                              source_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
                              model=self.row['model'], quote='a licensed error.', excerpt=True,
                              case_id='scene', turn=2, seed=42)

    def test_utf8_offsets_and_actual_generation_metadata(self):
        receipt = verify_quote(self.root, self.selection)
        raw = self.row['text'].encode()
        self.assertEqual(raw[receipt['quote_utf8_start']:receipt['quote_utf8_end']],
                         self.selection['quote'].encode())
        self.assertEqual(receipt['generation']['seed'], 42)
        self.assertEqual(receipt['source_jsonl_line'], 1)

    def test_changed_source_is_rejected(self):
        self.path.write_text(self.path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'hash changed'):
            verify_quote(self.root, self.selection)

    def test_paraphrase_is_rejected(self):
        self.selection['quote'] = 'an authorized error.'
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            verify_quote(self.root, self.selection)

    def test_wrong_checkpoint_and_seed_are_rejected(self):
        for change in ({'model': 'weights/step-40'}, {'seed': 137}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'exactly one'):
                verify_quote(self.root, {**self.selection, **change})

    def test_ambiguous_record_is_rejected(self):
        self.write_rows([self.row, self.row])
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            verify_quote(self.root, self.selection)

    def test_full_quote_cannot_silently_be_an_excerpt(self):
        self.selection['excerpt'] = False
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            verify_quote(self.root, self.selection)

    def test_wrong_sampling_is_rejected(self):
        self.selection['sampling'] = {'temperature': 0.3}
        with self.assertRaisesRegex(ValueError, 'sampling attribution'):
            verify_quote(self.root, self.selection)


if __name__ == '__main__':
    unittest.main()
