# Dubrovsky on Qwen3.5

Read the [Arianna Method Manifesto](../ARIANNA_METHOD_MANIFESTO.md) and
[agent rules](../AGENTS.md).

The first body is `huihui-ai/Huihui-Qwen3.5-0.8B-abliterated`, pinned to
`4813135658fe51b2e535b8e77927d2f23909ce42`. Training uses the complete
`Qwen3_5ForConditionalGeneration` architecture. The visual encoder is frozen
during text SFT. Input embeddings and the tied output head are also frozen;
the remaining language-body parameters are trainable. SHA-256 checks before
and after SFT and DPO verify that the frozen language matrices did not change.
NoTorch inference and
SUBLITERATUS integration follow the training and character evaluation work.

## Corpus

The supplied `dubrovsky.txt` is byte-identical to the historical file at
`400d5f8:dubrovsky.txt`. SHA-256:
`ec67ac88ca93c8892e969b87e2a7a5ed462cbf1c9d75a04aefc3277974500ef5`.
Its 1,066 Q/A records contain 486 unique pairs. Exact deduplication leaves
437 training pairs and 49 validation pairs. Normalized identical questions or
answers stay in the same split. Near-duplicate passages can remain.

The supplied wording is preserved, including first- and third-person speech.
The older `a.dubrovsky` subtitle sources remain separate from this first run.
Raw corpus and derived training files stay outside tracked git files.

```sh
python training/prepare_corpus.py /path/to/dubrovsky.txt .work/data
python -m unittest discover -s training -p 'test_*.py'
```

The pinned tokenizer yields 71,917 supervised tokens over the 486 unique pairs;
the longest complete example is 485 tokens. Each training example uses the
model's chat template with `enable_thinking=False`. Generation stops at either
the model's end-of-text token or the tokenizer's end-of-turn token. The prompt, including the
empty thinking prefix, is masked; the answer and its end marker are supervised.
Template-prefix mismatch and excessive sequence length are errors, not silent
truncations. Transformers 5.22.1 requires explicit `return_dict=False` for the
token-list path used here.

## Run

Install a CUDA-compatible PyTorch first, then `pip install -r training/requirements.txt`.
The scripts require a CUDA GPU. They write into fresh output paths and never
upload weights automatically.

```sh
python training/sample.py \
  --model huihui-ai/Huihui-Qwen3.5-0.8B-abliterated \
  --revision 4813135658fe51b2e535b8e77927d2f23909ce42 \
  --training-data .work/data/train.jsonl --output .work/baseline.jsonl

python training/sft.py \
  --revision 4813135658fe51b2e535b8e77927d2f23909ce42 \
  --data .work/data --output .work/sft --epochs 3 --lr 1e-5

python training/sample.py --model .work/sft/candidate \
  --training-data .work/data/train.jsonl --output .work/after-sft.jsonl
```

SFT uses FP32 parameters and optimizer state with BF16 autocast, token-weighted
gradient accumulation, gradient clipping, warmup and cosine decay. Run metadata
and epoch validation loss are written alongside the candidate. The default
three epochs and learning rate are starting settings, not a selected result.
`--max-steps 1` provides a real forward/backward/optimizer smoke run in a separate
output directory.

## Evaluate and select

Compare literal generations under the same prompt, system message, seed and
sampling settings. The probe set covers English, Russian, Spanish, French and
German and Hebrew, identity, first-person narration and novel absurdist scenes. Preserve
full token IDs, EOS/cutoff information, repetition counts and overlap with
training text. The eight-word overlap diagnostic measures copying; it does not
grade the character.

Read answers for Dubrovsky's voice, development of the user's association,
grammar, generic assistant identity, looping, and excessive corpus reproduction.
Build DPO preferences from observed answers to the same prompts and keep the
selection rationale. A lower loss by itself does not select the final body.
English is the first target; add language-polishing material in response to the
measured multilingual outputs.

Only the final selected weights go to the private `ataeff/Dubrovsky` repository.
Intermediate candidates remain on the training disk.

## Frozen language interface

The checkpoint ties its language embedding matrix to the output head. Both are
excluded from the optimizer, and their pre/post-training hashes must match.
This applies to `sft.py` and `dpo.py`. Training starts from the original Huihui
revision, not from the earlier interrupted five-step run with trainable embeddings.

English instruction tuning with frozen original embeddings is also used in
[Jiang et al., EACL 2026, §2.2](https://aclanthology.org/2026.eacl-long.357.pdf).
That paper studies embedding relearning/swapping on other model families; it
does not measure this Dubrovsky recipe. Language retention here is evaluated
against the same Huihui checkpoint, prompts, seeds and decoding settings.

For DPO, provide reviewed JSONL records containing `question`, `chosen`,
`rejected`, and `reason`, with both answers responding to the same question:

```sh
python training/dpo.py --model .work/sft/candidate \
  --preferences .work/preferences.jsonl --output .work/dpo
```

The SFT candidate is the fixed reference. Its answer-token log probabilities
are computed before updates; the policy then trains the remaining language
body with the DPO log-sigmoid objective. Candidate selection still requires
post-training generations on prompts excluded from preference construction.
