"""Retain evaluated weights before advancing the best-candidate pointer."""
import json
import math
from pathlib import Path


class CandidateCheckpoints:
    def __init__(self, output, export):
        self.output = Path(output)
        self.export = export
        self.saved = {}
        self.best_loss = math.inf

    def save(self, step, epoch):
        if step not in self.saved:
            path = self.output / f'step-{step:06d}'
            self.export(path, step, epoch)
            self.saved[step] = path
        return self.saved[step]

    def consider(self, loss, step, epoch):
        if not math.isfinite(loss):
            raise ValueError('Best-candidate loss must be finite')
        if loss >= self.best_loss:
            return False
        path = self.save(step, epoch)
        # Never point best at an incomplete export. Keep earlier improvements
        # available for generation-based selection; loss alone is not promotion.
        temporary = self.output / '.best-next'
        temporary.symlink_to(path.name, target_is_directory=True)
        temporary.replace(self.output / 'best')
        metadata = self.output / '.best-json-next'
        metadata.write_text(json.dumps(dict(step=step, epoch=epoch,
            validation_loss=loss, path=path.name), indent=2) + '\n')
        metadata.replace(self.output / 'best.json')
        self.best_loss = loss
        return True
