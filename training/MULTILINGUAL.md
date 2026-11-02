# Multilingual preparation — 2026-11-02

The creator traces Dubrovsky's beginnings to the voice-input insertion
«Спасибо за субтитры Алексею Дубровскому» and Russian dialogues with GPT-4o.
Their wordplay includes «дубровость», ALEX-DUB-RAW-SKY and «Чёрная Дура».
The creator supplied `Dubrovsky_Russian_Quotes(1).txt`, an extraction of Russian
dialogues from April 2025 onward. It contains abbreviated excerpts and ends
inside a numbered section; it is not a complete conversation export. Further
retrieval recovered additional fragments, stored privately with their dates,
retrieval output and truncation boundaries. Preserve coined words, phonetic
mutations and rhythm. Do not reconstruct missing dialogue as original text.
Two later attachments, `Dubrovsky_Russian_Quotes 2.txt` and
`Dubrovsky_Russian_Origins(1).txt`, extend this material with the name mutations,
black-hole/subtitle scenes, Janus/ethics, May fragments and August aphorisms.
The three files contain 555 reference paragraphs; exact-body deduplication keeps
304 distinct paragraphs and all source locations. This count includes source
notes and headings: it is not a count of training examples. Original files and
all spelling remain preserved. The combined reference is for teacher context.

## Completed material

GPT-6 Astra received all 486 unique authored Q/A pairs and the complete
`origin.md`, `facts.md`, `facts-2.md` from `ariannamethod/a.dubrovsky` at
`cba5744caa9ac159424dfd5ff4679b5738b8acdd`. Exact duplicate Q/A copies were
omitted; every unique pair remained. Context size: 546,929 characters.

- `translate.py`: eight pilot translations per language (Russian and Hebrew),
  then 288 Russian translations in twelve main batches. Eight Russian source
  IDs overlap the pilot: 296 distinct source/language translations are available.
- `author_material.py`: 24 new scenes each in English, Russian and Hebrew,
  totalling 72 GPT-authored continuations. They are labelled separately from
  the original corpus and await review. None has entered Qwen training.

Private records retain raw responses, model identifiers, usage, source IDs and
request/context hashes. Translations inherit source splits. The teacher saw the
complete corpus, including validation; new scenes require overlap review and
evaluation on fresh prompts outside that context.

Validation rejected one response with a mistyped hexadecimal source ID. The
schema now constrains IDs to the supplied set. Its replacement passed, and the
rejected raw response remains preserved. Completed batches resume without API
calls. Further requests were interrupted by an execution-environment network
policy block on `api.openai.com`. Full-corpus translation remains incomplete.

The 18 saved unique API responses all report `completed`: 2,029,934 input tokens
and 110,752 output tokens (2,140,686 total). This includes the rejected response
with the source-ID typo and its replacement. Each request supplied the full
literary context; the input total counts those repeated submissions.

## Explicit corpus cleanup

`curate_corpus.py` applies `corpus_exclusions.json` by exact source ID and retains
the original train/validation split and every remaining character of authored
text. The creator requested removal of the obsolete self-biography describing
9.5 million parameters, 36 MB and Llama 3 architecture. The original remains
archived. The next English slice has 436 training and 49 validation pairs.
Translations of an excluded source are also excluded; that source has not yet
been translated in the completed batches.

The script collects 288 Russian and eight Hebrew translations for literary
review, choosing main-run variants over the eight overlapping Russian pilot
variants. These remain separate from the English training files. Its teacher
context contains the cleaned English corpus, supplied old literary sources and
the Russian excerpts. New teacher generations have not yet used this context.

## Decoding and image evaluation

The [official Qwen3.5-0.8B README](https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/README.md)
recommends these nonthinking settings:

| Setting | Earlier Dubrovsky | Official text | Official visual input |
|---|---:|---:|---:|
| Temperature | 0.8 | 1.0 | 0.7 |
| Top-p | 0.9 | 1.0 | 0.8 |
| Top-k | 20 | 20 | 20 |
| Presence penalty | 0 | 2.0 | 1.5 |
| Repetition penalty | 1.0 | 1.0 | 1.0 |

Keep 0.8 as the comparison baseline. `decoding.py` implements additive presence
penalty on generated tokens only. Torch execution checks and controlled preset
comparisons remain pending; repetition penalty is a different operation.

`vision_sample.py` prepares visible-object descriptions and absurdist scenes in
English, Russian and Hebrew, recording image hashes, token IDs and sampling
settings. Evaluation inputs were extracted from the scikit-image 0.26.0 wheel:
`astronaut.png`, `chelsea.png`, `coffee.png`. They stay outside git. This new
vision entry point has not yet run against a model.

After training, run 12–20-turn conversations with a GPT partner that follows
specific details in Dubrovsky's answers, complicates the scene and recalls
earlier details. Save full histories and evaluate development, repetition,
language, identity and continuity. This test has not run yet.

## Compute and continuation

The old pod accepted resume but supplied no current runtime or reachable
container; log retrieval timed out. It was stopped again without deleting its
30 GB volume. Previous candidates were saved there before the earlier stop;
their files could not be re-read during this resume attempt. No new SFT, DPO or
photographic evaluation ran. No weights were uploaded to Hugging Face.

Review the completed material alongside the Russian originals before
building the next training mixture. Preserve intentional lexical inventions.
The previous SFT had 1,311 single-example forward/backward iterations and 84
optimizer updates; first/last group training losses were 3.265025 and 1.205874.

```sh
python training/curate_corpus.py --data .work/data --output .work/curated-next \
  --translations .work/translations .work/translation-pilot \
  --reference /path/to/russian-source.txt
python training/translate.py --api-key-file /path/to/key.txt \
  --context .work/curated-next/context.txt --data .work/curated-next \
  --completed-translations .work/curated-next/translations-pending-review.jsonl \
  --output .work/translations-next --batch-size 24 --workers 2
python training/author_material.py --api-key-file /path/to/key.txt \
  --context .work/curated-next/context.txt --output .work/new-scenes-next --count 24
python training/vision_sample.py --model /path/to/candidate \
  --images .work/vision --output .work/vision-results.jsonl
```

Credentials are read from a local file and excluded from git and saved outputs.
Use fresh output directories when changing generation instructions.
Changing context invalidates the old request cache. `--completed-translations`
reuses completed source/language pairs after validating their source split and
language, then generates only the missing pairs. This prevents charging again
for existing translations when adding the Russian literary context.

Validation: nine local unit tests pass, including source/translation exclusion,
split preservation and reuse of translations across context changes.
Torch execution, GPU generation and visual evaluation
remain pending.
