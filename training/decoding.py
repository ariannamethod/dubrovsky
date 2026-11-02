"""Explicit decoding presets; additive presence penalty applies only to generated tokens."""
PRESETS = {
    'baseline': dict(temperature=0.8, top_p=0.9, top_k=20, presence_penalty=0.0),
    'temperature1': dict(temperature=1.0, top_p=0.9, top_k=20, presence_penalty=0.0),
    'qwen-text': dict(temperature=1.0, top_p=1.0, top_k=20, presence_penalty=2.0),
    'qwen-vision': dict(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5),
}


class GeneratedPresencePenalty:
    def __init__(self, prompt_length, penalty):
        self.prompt_length, self.penalty = prompt_length, penalty

    def __call__(self, input_ids, scores):
        import torch
        seen = torch.zeros_like(scores, dtype=torch.bool)
        seen.scatter_(1, input_ids[:, self.prompt_length:], True)
        return scores - self.penalty * seen.to(scores.dtype)


def generation_kwargs(preset, prompt_length):
    from transformers import LogitsProcessorList
    values = PRESETS[preset].copy()
    penalty = values.pop('presence_penalty')
    return dict(**values, min_p=0.0, repetition_penalty=1.0, do_sample=True,
                logits_processor=LogitsProcessorList(
                    [GeneratedPresencePenalty(prompt_length, penalty)] if penalty else []))
