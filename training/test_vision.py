import unittest

from vision_sample import DEFAULT_QUESTIONS, validate_questions


class VisionQuestionTests(unittest.TestCase):
    def test_multilingual_probe_identity_is_unambiguous(self):
        validate_questions(DEFAULT_QUESTIONS)
        validate_questions([dict(id='fr-test', language='fr', question='Que vois-tu ?')])
        for invalid in ([], {}, [None], [dict(id='x', language='fr', question=' ')],
                        DEFAULT_QUESTIONS + [DEFAULT_QUESTIONS[0]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_questions(invalid)


if __name__ == '__main__':
    unittest.main()
