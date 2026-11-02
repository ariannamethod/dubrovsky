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
The older `a.dubrovsky` subtitle sources remain separate from the first SFT run;
selected original first-person lines enter the subsequent preference stage.
Raw corpus and derived training files stay outside tracked git files.

```sh
python training/prepare_corpus.py /path/to/dubrovsky.txt .work/data
python -m unittest discover -s training -p 'test_*.py'
```

The pinned tokenizer yields 71,917 supervised tokens over the 486 unique pairs;
the longest complete example is 485 tokens. By default, each training example uses the
model's chat template with `enable_thinking=False`. Generation stops at either
the model's end-of-text token or the tokenizer's end-of-turn token. The prompt, including the
empty thinking prefix, is masked; the answer and its end marker are supervised.
Template-prefix mismatch and excessive sequence length are errors, not silent
truncations. Transformers 5.22.1 requires explicit `return_dict=False` for the
token-list path used here.

`enable_thinking=False` supplies a closed empty thinking prefix at inference.
The original answer-only SFT masks that prefix, so it supplies no direct loss
for choosing to close an opened thinking block. The optional SFT flag
`--supervise-empty-thinking` trains the native empty closure and the answer
from the open thinking prefix. It changes language-body weights; the architecture,
tokenizer and frozen matrices stay intact. Validation remains answer-only under
the closed prefix, preserving the previous loss definition.

The open prefix ends in one newline, while the closed native template contains
two consecutive newlines. Qwen's BPE merges those two newlines. Training therefore
tokenizes the continuation at the actual open-prefix generation boundary and
checks exact rendered-text equality, rather than assuming the two token sequences
have a shared prefix. History and the open prompt remain masked.

`thinking_probe.py` compares both prefixes on identical questions, configurations
and seeds. It retains raw special tokens and reports the position of `</think>`,
text before it, the final answer and EOS/cutoff separately. An EOS without a
thinking closure is recorded as an unclosed block. Measure this learned behavior
on held-out requests, including explicit requests to reason, before selecting
a candidate. Runtime generation continues to default to disabled thinking.

Add `--thinking-closure-only` to isolate that transition: the current answer is
removed from the training sequence and only the empty closure receives loss.
Earlier dialogue turns remain context. Use a separate run from the same selected
parent when comparing this control with joint language/answer training.

```sh
python training/sft.py --model /path/to/selected-candidate \
  --data /path/to/reviewed-mixture --output .work/empty-thinking-sft \
  --epochs 1 --lr 1e-6 --supervise-empty-thinking --answer-logits-only

python training/thinking_probe.py --model /path/to/candidate \
  --cases /path/to/fixed-cases.json --configurations /path/to/settings.json \
  --seeds 42 137 --output .work/thinking-comparison.jsonl
```

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

Keep the final selected full and quantized weights in the private
`ataeff/Dubrovsky` repository. Temporary private backups are also authorized
when needed for recovery; keep them separate from the final release and remove
them after that release is secure. Other intermediate candidates remain on the
training disk. Private datasets and their revisions belong in the private HF
dataset repository, outside git.

The [method audit](METHOD_AUDIT.md) records implementation defects, evaluation
exposure, and limitations observed during the response-repair continuation.

## Frozen language interface

The checkpoint ties its language embedding matrix to the output head. Both are
excluded from the optimizer, and their pre/post-training hashes must match.
This applies to `sft.py` and `dpo.py`. Initial training started from the pinned
original Huihui revision, not the earlier interrupted five-step run with
trainable embeddings. Continuations record their selected parent checkpoint.

English instruction tuning with frozen original embeddings is also used in
[Jiang et al., EACL 2026, §2.2](https://aclanthology.org/2026.eacl-long.357.pdf).
That paper studies embedding relearning/swapping on other model families; it
does not measure this Dubrovsky recipe. Language retention here is evaluated
against the same Huihui checkpoint, prompts, seeds and decoding settings.

For DPO, provide reviewed JSONL records containing `question`, `chosen`,
`rejected`, and `reason`, with both answers responding to the same question.
An optional `history` contains complete alternating user/assistant turns.
Both alternatives receive that same history, which remains masked; only the
current answer and turn end contribute to their likelihood. `--max-length`
sets the complete-conversation limit (default 1024); oversized pairs are rejected
without truncation. Use a larger explicit limit for reviewed long conversations:

```sh
python training/dpo.py --model .work/sft/candidate \
  --preferences .work/preferences.jsonl --output .work/dpo --max-length 3072
```

The SFT candidate is the fixed reference. Its answer-token log probabilities
are computed before updates; the policy then trains the remaining language
body with the DPO log-sigmoid objective. Candidate selection still requires
post-training generations on prompts excluded from preference construction.

`--chosen-nll-weight 1.0` adds preferred-answer token NLL to DPO. The default
`0.0` is ordinary DPO. This option trains the chosen wording directly as well
as preferring it over the rejected answer. Record both the preference set and
the weight when comparing runs. The initial experiments changed both; their
results are recorded in [EXPERIMENTS.md](EXPERIMENTS.md).

For language work, follow the [literary translation instructions](translation_instructions.md).
Keep translated records in their original source split and retain English
examples when building a multilingual training mixture.

## Prepare a selected release

After character and language evaluation selects a candidate, package it with
the existing native nonthinking prefix as the default:

```sh
python training/prepare_release.py .work/selected/candidate .work/release \
  --model-card /path/to/reviewed-model-card.md
```

The helper checks tokenizer parity for English, Russian and Hebrew with the
thinking option absent, false and true, then copies the candidate and records
weight-file hashes. It performs no upload. Release selection and upload remain
separate explicit steps; the first English experiments are still candidates.

## Long dialogue continuation

[The multilingual record](MULTILINGUAL.md) documents the current data and runs.
`prepare_dialogues.py` compiles authored conversations into causal histories;
only the current assistant answer and end marker receive supervised loss.
`build_dialogue_mixture.py` combines these rows with deterministic language
replay and preserves the development split. Duplicate IDs and split overlap
are rejected. Private training files and literal generations stay outside git.

```sh
python training/prepare_dialogues.py \
  --input /path/to/en.json /path/to/ru.json /path/to/he.json \
  --output .work/compiled-dialogues

python training/build_dialogue_mixture.py \
  --replay .work/earlier-mixture \
  --dialogues .work/compiled-dialogues/train.jsonl \
  --output .work/long-mixture

python training/sft.py --model /path/to/selected-candidate \
  --data .work/long-mixture --output .work/long-sft \
  --epochs 3 --lr 2e-6 --max-length 3072 \
  --save-each-epoch --answer-logits-only
```

`--answer-logits-only` retains the full attention history while reducing
vocabulary projection to the answer and its preceding prediction position.
`check_loss_parity.py` compares its loss and peak memory with the full projection
on a native encoded example. The flag defaults off. `chat_probe.py` preserves
full partner histories and reports repetition within and between answers.
Compare saved epochs using fresh conversations, selective callbacks, new themes
and language quality. The final character and runtime selection follows those
conversations.


## Controlled sampling comparisons

`compare_sampling.py` loads the candidate once and compares explicit temperature,
top-p, top-k and generated-token presence penalty configurations. Fixed mode
uses exactly the same supplied history for every configuration and seed. Dialogue
mode follows a fixed sequence of user prompts while each branch retains its own
actual assistant answers. It records full histories, token IDs, EOS termination,
input hashes, library versions and within-answer/prior-answer repetition.

```sh
python training/compare_sampling.py --model PRIVATE_LOCAL_CANDIDATE \
  --cases private/fixed-cases.json --configurations private/grid.json \
  --seeds 42 137 --output private/comparison.jsonl
python training/compare_sampling.py --model PRIVATE_LOCAL_CANDIDATE \
  --mode dialogue --cases private/conversations.json \
  --configurations private/selected-settings.json \
  --seeds 42 --output private/long-conversations.jsonl
```

Cases are a JSON array with unique `id`, `language`, and either `messages`
(fixed mode, alternating roles, ending with user) or `user_turns` (dialogue mode).
Configurations are a JSON array with unique `id` and `sampling`, which must
specify all four settings. Existing result files are never overwritten. Context
is retained in full; excessive context raises an error instead of truncating.
The four-gram diagnostics expose lexical copying; read the literal conversation
to judge semantic continuity, scene changes, grammar and character.

Text and image probes store `raw_text` decoded without stripping special tokens,
alongside readable `text` and token IDs. Thinking-marker checks inspect the raw
generated continuation, excluding the prompt's intentional empty thinking prefix.
Word repetition and corpus overlap continue to use readable text.


### Reproducible multilingual image probes

`vision_sample.py` accepts `--questions questions.json` with a nonempty JSON list
of `{ "id": "fr-character", "language": "fr", "question": "..." }` objects and
`--seeds 42 137`. Question IDs must be unique. Without these options it retains
the original four English/Russian/Hebrew questions and seed 42. Empty image sets
and unpinned remote models fail before model loading.

Each PNG/question/seed result records its image hash, question-set hash, literal
answer and tokens, EOS, sampling preset, library versions and nonthinking mode.
Use the same photographs, question file and seeds on both candidates. Recognition
of visible objects and the character's invented scene are reviewed separately.
Do not reuse generated evaluation answers as training targets without explicitly
changing their role and recording a new holdout split.

### Lineage comparison and export integrity

[The lineage audit](LINEAGE_AUDIT.md) compares the original huihui, early
English SFT and the selected stage-7 model under matched inference conditions.
The selected weights remain unchanged. New grammar-choice diagnostics distinguish
exact answers from option mentions so character narration is not silently scored
as an ordinary one-word answer:

```sh
python training/score_language_choices.py \
  --references /private/evaluation/grammar-references.json \
  --generations /private/results/huihui-grammar.jsonl /private/results/selected-grammar.jsonl \
  --output /private/results/grammar-summary.json
```

SFT BF16 export now retains shared embedding/head storage. Unit checks for real
PyTorch tensor aliasing run when PyTorch is installed; the actual Qwen round-trip
measurements are recorded in the audit. Private prompts and outputs remain on
Hugging Face.

### Matched multilingual SFT

[The joint multilingual trial](JOINT_MULTILINGUAL.md) compares language ordering
from the original huihui and separately tests learned immediate thinking closure.
Prepared schedules can retain their exact row order and report intermediate
validation without changing the default shuffled-training behavior:

```sh
python training/sft.py --model /private/pinned-huihui \
  --data /private/prepared-mixed --output /private/new-mixed-run \
  --epochs 1 --ordered-training --eval-every-steps 40 \
  --lr 3e-6 --max-length 3072 --answer-logits-only
```

Here the prepared input already contains three presentations per source record;
`--epochs 1` consumes that complete schedule once. `training_order.jsonl` records
the actual record IDs and languages at every optimizer step. Optional
`evaluation_group` tags in validation rows separate fresh references from legacy
development references in `validation_by_group`. They do not affect supervision
or the existing aggregate/per-language loss. Add `--supervise-empty-thinking`
only for the explicitly matched closure arm.

### Russian-priority continuation

[The Russian continuation](RUSSIAN_CONTINUATION.md) trains from the stage-12
closure candidate with new reviewed Russian dialogues and five-language replay.
Each epoch is saved and compared on fresh scenes, ownership recall after topic
changes, literal grammar, and native thinking closure. Private data, raw
generations and weight backups remain in the Hugging Face repositories.
