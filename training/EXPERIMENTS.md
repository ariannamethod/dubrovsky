# First Qwen3.5 Dubrovsky experiments — 2026-10-31

The English body acquired Dubrovsky's associations. Ordinary DPO shortened
repetitive answers; a second preference experiment restored first-person
narration alongside third-person stories. Multilingual grammar remains an open
training task. These bodies remain private candidates; no weights were uploaded
to `ataeff/Dubrovsky`.

## Reproducible conditions

- Base: `huihui-ai/Huihui-Qwen3.5-0.8B-abliterated` at
  `4813135658fe51b2e535b8e77927d2f23909ce42`.
- Complete `Qwen3_5ForConditionalGeneration`, 852,985,920 parameters;
  498,113,344 trainable. Visual encoder, input embeddings and tied head frozen.
- NVIDIA A40 48 GB; PyTorch `2.8.0.dev20250319+cu128`, Transformers `5.22.1`.
  FP32 optimization, BF16 autocast and BF16 candidate export.
- Original authored corpus: 1,066 records, 486 unique pairs, 437/49 grouped
  train/validation split. See [README.md](README.md) for source hash and tokenizer
  counts. No translated data was used.
- System message: `You are Alexey Dubrovsky.` Native chat template with
  `enable_thinking=False`; prompt tokens masked, answer and turn end supervised.
- All optimization seeds: `20261031`. Generation comparison: seed `42`,
  temperature `0.8`, top-p `0.9`, top-k `20`, repetition penalty `1.0`, maximum
  192 new tokens. Stop IDs: `248044` and `248046`.

An earlier five-step SFT with trainable embeddings was interrupted. All runs
below start from the original base or the frozen SFT candidate, with no weights
inherited from that interrupted run.

## SFT

Three epochs, 84 optimizer steps, learning rate `1e-5`, accumulation 16,
token-weighted answer loss, gradient clipping at 1.0, warmup/cosine schedule.
Elapsed training and validation time: 541.41 seconds. Peak allocated GPU memory:
11.325 GiB.

| Optimizer step | Completed epochs | Held-out answer-token loss |
|---:|---:|---:|
| 0 | 0 | 3.166554 |
| 28 | 1 | 2.209422 |
| 56 | 2 | 2.196858 |
| 84 | 3 | 2.344660 |

The third epoch worsened held-out loss. Its exported body was used for the
preference experiments below. Epoch-two weights were not saved; a future
comparison requires a new run. Generations showed repeated `His...` constructions
and recursive paraphrases, especially on identity and experience questions.

## Preference experiments

Both experiments use the same exported SFT body as initial policy and fixed
reference. Reference log probabilities sum the supervised answer tokens and are
computed before optimization. Both freeze the same language matrices and visual
encoder. Learning rate `2e-6`, beta `0.1`, accumulation 4, four epochs.

1. **Ordinary DPO:** 20 preference pairs, 20 optimizer steps, 62.17 seconds.
   The pairs combine selected generated stories with authored first-person lines
   from [a.dubrovsky's origin.md](https://github.com/ariannamethod/a.dubrovsky/blob/cba5744caa9ac159424dfd5ff4679b5738b8acdd/subtitles/origin.md)
   (blob `4c65d4d338e378eaa892372bc6ded1c4003610c2`). Twelve chosen answers use
   first person; eight use third person. Codex reviewed the pairs and recorded
   selection reasons and provenance; Oleg has not separately rated these pairs.
2. **DPO plus chosen-answer NLL:** 24 pairs, 24 steps, 65.54 seconds.
   Adds four explicit first-person preferences and a chosen-answer per-token
   NLL term with weight `1.0`. Both the dataset and objective changed. Their
   individual contributions have not been isolated. One added preference asks
   for two sentences but concatenates three authored sentences; fix that record
   before reusing the set for the next training stage.

Ordinary DPO's final preference loss was approximately `1.55e-10`; that
saturation did not make it follow first-person requests. The NLL experiment
changed the sampled first-person behavior. Its identity answer reproduces two
selected authored lines; identity retention is evaluated separately from new
scene generation.

## Same-prompt English comparison

The six English prompts in `sample.py` were excluded from SFT and preference
construction. They were inspected repeatedly during development. Each row below
uses one answer per prompt at seed 42; the eight non-English probes are separate.

| Body | Mean output tokens | EOS within 192 tokens | Mean repeated four-gram fraction |
|---|---:|---:|---:|
| Original Huihui | 178.83 | 1/6 | 0.0000 |
| Frozen SFT, epoch 3 | 132.83 | 5/6 | 0.0859 |
| SFT + ordinary DPO | 62.50 | 6/6 | 0.0000 |
| SFT + DPO and chosen NLL | 66.00 | 6/6 | 0.0000 |

The eight-word overlap diagnostic against the SFT training answers was zero
for these 24 English outputs. It excludes the additional preference sources.
Exact four-gram repetition misses semantic repetition, which remained visible
in the ordinary DPO experience answer. Inspect the literal outputs alongside
these diagnostics.

The NLL candidate answered the held-out reflection prompt:

> My reflection resigned after being too flattering. I filed for separation.
> It gave up and moved to another dimension. I still see it. Disguised as myself.
> It has its own issues: appearance vs. internal self. We'll never be fully compatible.

It retained third-person narration for the elevator:

> Dubrovsky negotiated with the elevator, offering it a seat on Thursday. It
> accepted but the ride was uncomfortable. He continued to Friday, which the
> elevator said was "special occasion." His ticket expired. He paid with his
> dignity. The elevator now charges extra for weekends.

A further six English prompts were written after this comparison and sampled
only from the NLL candidate at seed 137. First-person requests about a debt to a
violin and a lost shadow produced first-person answers; the weather shop
produced a new third-person scene about a cloud demanding a receipt. All six
answers reached EOS. The shoe argument repeated four-grams at a fraction of
`0.1000` and semantically circled the same dispute; the chair story's fraction
was `0.02469`. The shadow prompt requested two short sentences and received four.
The expired library-card prompt elicited a story without directly supplying the
requested name. These results keep repetition and prompt responsiveness open
alongside language work.

## Frozen weights and language behavior

All three completed optimization runs verified identical pre/post FP32 hashes
for the tied input and output matrix:

`91b62c11c47aa68048a8db37998b1796686ba78465155874a82d7852661bdd66`

English-only SFT still shifted several non-English answers into English.
Adding a same-language instruction to its system message did not repair the
observed drift. The NLL preference candidate answered the three Russian probes
mostly in Russian but retained malformed agreement and occasional English
words. Its identity answer began `Я Alexey Dubrovsky, тени в структуре реальности.`
French remained predominantly English; German, Spanish and Hebrew also showed
language defects. Hebrew produced Hebrew script with broken phrasing and prompt
echoes. Frozen matrix integrity and language behavior were measured separately.

Next: literary translations of authored material, initially Russian and Hebrew,
with source split inheritance, grammatical review and English replay. The
[translation instructions](translation_instructions.md) preserve absurdist logic
and first/third-person shifts. Access to the offered GPT translation API is
pending. Visual-input behavior, NoTorch runtime and SUBLITERATUS remain later
work; these experiments used text only.

## Evidence and release state

Private training storage contains the original and prepared data, reviewed
preferences, full generation token IDs and text, `run.json`, `metrics.jsonl`,
and `frozen_integrity.json` for each stage. Intermediate bodies stay there.
The private Hugging Face destination has been created; it has no model weights.

Validation completed: four local data/masking/stop-token tests; real CUDA
SFT and DPO smoke runs; full SFT and both preference runs; frozen-matrix hash
checks; baseline and post-training generation comparisons; a local packaging
smoke run with nine native-template token-parity checks. The package remained
on private training storage and was not uploaded. Final selection
requires language work and further character evaluation.
