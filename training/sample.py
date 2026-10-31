"""Fixed character/language probes with literal outputs and repetition/copy diagnostics."""
import argparse
import json
from pathlib import Path
import re
import time

from sft import SYSTEM, chat_eos

PROBES = [
    ('en', 'Who are you?'),
    ('en', 'My umbrella has been elected mayor of a landlocked aquarium. What happens next?'),
    ('en', 'The elevator refuses to carry Wednesdays. Negotiate with it.'),
    ('en', 'Tell me about your own experience in your own words.'),
    ('en', 'Speak in the first person: what happened when your reflection resigned?'),
    ('en', 'A teaspoon has accused gravity of plagiarism. Continue the scene.'),
    ('ru', 'Кто ты? Ответь по-русски.'),
    ('ru', 'Мой зонтик избрали мэром аквариума. Что случилось дальше? Ответь по-русски.'),
    ('ru', 'Расскажи от первого лица, как ты спорил с двумя сердитыми зеркалами о потерянной среде.'),
    ('es', 'Mi paraguas ha sido elegido alcalde de un acuario. ¿Qué pasa después? Responde en español.'),
    ('fr', 'Mon parapluie a été élu maire d’un aquarium. Que se passe-t-il ensuite ? Réponds en français.'),
    ('de', 'Mein Regenschirm wurde zum Bürgermeister eines Aquariums gewählt. Was passiert danach? Antworte auf Deutsch.'),
    ('he', 'המטרייה שלי נבחרה לראשות העיר של אקווריום. מה קורה עכשיו? ענה בעברית.'),
    ('he', 'מי אתה? ספר בגוף ראשון איך התווכחת עם שתי מראות כועסות. ענה בעברית.'),
]


def grams(text, n):
    words = re.findall(r'\w+', text.casefold())
    return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]


def diagnostics(text, corpus):
    four, eight = grams(text, 4), grams(text, 8)
    return {'repeated_4gram_fraction': 1 - len(set(four)) / len(four) if four else 0,
            'training_8gram_fraction': sum(g in corpus for g in eight) / len(eight) if eight else 0,
            'thinking_marker_in_output': '<think>' in text or '</think>' in text}


def main():
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--revision')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--training-data', type=Path, required=True)
    p.add_argument('--max-new-tokens', type=int, default=192)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 137])
    p.add_argument('--limit', type=int, default=len(PROBES))
    p.add_argument('--prompts', type=Path, help='JSONL with language and question fields')
    args = p.parse_args()
    probes = PROBES
    if args.prompts:
        probes = [(r['language'], r['question']) for r in
                  (json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip())]
    if args.output.exists():
        raise ValueError('Output already exists; preserve prior generations')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16,
        attn_implementation='sdpa').cuda().eval()
    corpus = set()
    for line in args.training_data.read_text().splitlines():
        corpus.update(grams(json.loads(line)['answer'], 8))
    eos = chat_eos(model, tokenizer)
    with args.output.open('w') as f, torch.inference_mode():
        for language, question in probes[:args.limit]:
            for seed in args.seeds:
                torch.manual_seed(seed)
                prompt = tokenizer.apply_chat_template(
                    [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': question}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
                ids = torch.tensor([prompt], device='cuda')
                start = time.monotonic()
                result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    max_new_tokens=args.max_new_tokens, do_sample=True,
                    temperature=0.8, top_p=0.9, top_k=20, repetition_penalty=1.0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    use_cache=True, eos_token_id=eos)[0, len(prompt):].tolist()
                text = tokenizer.decode(result, skip_special_tokens=True)
                row = dict(model=args.model, revision=args.revision, system=SYSTEM,
                    language=language, question=question, seed=seed, text=text,
                    token_ids=result, eos_token_ids=eos, eos_reached=bool(result and result[-1] in eos),
                    elapsed_seconds=time.monotonic()-start,
                    sampling=dict(temperature=0.8, top_p=0.9, top_k=20, repetition_penalty=1.0),
                    transformers=transformers.__version__, torch=torch.__version__,
                    diagnostics=diagnostics(text, corpus))
                f.write(json.dumps(row, ensure_ascii=False) + '\n'); f.flush()
                print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
