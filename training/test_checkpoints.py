import unittest
import json
import tempfile
from pathlib import Path

from sft import checkpoint_steps
from checkpoints import CandidateCheckpoints


class CheckpointScheduleTests(unittest.TestCase):
    def test_shortened_run_rejects_unreachable_export(self):
        with self.assertRaisesRegex(ValueError, 'reachable'):
            checkpoint_steps([20, 40, 80], total_steps=56)

    def test_zero_and_duplicates_are_rejected(self):
        for steps in ([0], [-1], [20, 20]):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                checkpoint_steps(steps, total_steps=100)

    def test_final_step_and_disabled_schedule(self):
        self.assertEqual(checkpoint_steps([20, 100], 100), {20, 100})
        self.assertEqual(checkpoint_steps([], 100), set())

    def test_best_survives_later_regression_and_duplicate_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            def export(path, step, epoch):
                path.mkdir()
                (path/'weights').write_text(str(step))
                calls.append(step)
            checkpoints = CandidateCheckpoints(root, export)
            checkpoints.consider(2.4, 0, 0)
            checkpoints.consider(2.1, 20, 1)
            checkpoints.save(20, 1)
            self.assertFalse(checkpoints.consider(2.2, 56, 1))
            self.assertFalse(checkpoints.consider(2.1, 60, 2))
            self.assertEqual(calls, [0, 20])
            self.assertEqual((root/'best'/'weights').read_text(), '20')
            self.assertEqual(json.loads((root/'best.json').read_text())['step'], 20)
            self.assertTrue((root/'step-000000'/'weights').exists())

    def test_failed_export_does_not_replace_best(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def export(path, step, epoch):
                if step == 20:
                    raise OSError('disk full')
                path.mkdir()
            checkpoints = CandidateCheckpoints(root, export)
            checkpoints.consider(2.4, 0, 0)
            with self.assertRaises(OSError):
                checkpoints.consider(2.1, 20, 1)
            self.assertEqual((root/'best').resolve().name, 'step-000000')
            self.assertEqual(checkpoints.best_loss, 2.4)
            with self.assertRaises(ValueError):
                checkpoints.consider(float('nan'), 40, 1)


if __name__ == '__main__':
    unittest.main()
