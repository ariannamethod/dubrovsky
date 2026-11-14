# Embedded experts: RU/EN pilot (stage 24)

Oleg selected English and Russian for this stage and requested learned experts
inside Dubrovsky's weights. The target is continuity of the character's thought
across changing topics, factual updates and associative scenes. The selected
stage-7 body is the frozen parent. Janus-X's documented separation of user
memory and working state informs this new Qwen implementation.

## Model and state

`embedded_experts.py` registers four experts and their router as model
parameters. At Qwen's hidden width 1,024 and expert rank 16 they contain 210,508
trainable parameters. The entire original VLM, including its tied language
interface and visual encoder, is frozen and kept in evaluation mode.

Each expert has a 16-dimensional user bank and working state. For every input
token, normalized causal Qwen features produce fresh memory content. Independent
retention and write gates see the current features, old bank, fresh content and
their difference. A user-content token can update the bank:

```
user_next = tanh(retain * user_old + write * fresh)
```

For system text, template markers and assistant tokens the user bank stays
unchanged. Working state updates on every token. The four experts project their
working states back to hidden width; a softmax router combines their proposals
before the original vocabulary head. The added vector has an RMS ceiling of
0.1. Only the output projections start at zero, giving exact initial parent
logits while leaving a learnable gradient into that output. The remaining
expert parameters receive gradients once the output moves.

Learned parameters belong to the checkpoint. Dialogue states belong to the
caller and are never serialized as model weights. A new conversation starts
with zero state. Full-prefix and incremental calls share the same causal
transition; cache and memory prefix lengths must agree. Re-evaluating an expert
state does not mutate the input state.

The initial sampler reconstructs memory from each conversation's actual accepted
history at turn boundaries, then uses the Qwen cache and expert state token by
token. Each comparison arm retains its own generated history. Fixed-history
pair scoring is a separate comparison.

## Parent and data

Private model: `ataeff/Dubrovsky`, revision
`805bc6e493f284be8b20c262b8fe65070c886c27`, subfolder
`working/six-language-polish-epoch001`. Parent weight SHA-256:
`e365806211389de6513e1e7c7706e54b2203531d8e16acceb843488a0f707d5b`.

The private pilot has 225 training rows (RU 94, EN 131) and 54 validation
rows (27 per language). New paired gift/loan histories include changes of
custody, topic changes, return to an earlier scene, and explicitly authored
incorrect assistant history with a target grounded in user events. These are
controlled templates grouped by scene family. Both branches and both languages
of a family stay in the same split. Existing reviewed stage-23 Russian training
dialogues and stage-7 English voice replay supply the remaining examples.

The native-tokenizer audit covers all 279 rows: 3,235 RU / 8,977 EN supervised
training tokens, maximum sequence 1,296 tokens; validation has 965 RU / 1,481 EN
target tokens. No assistant target position writes the user bank. The audit
checks native template IDs and refuses truncation or rewritten prefixes.

Eight fresh fixed-history contrast pairs cover two unseen families in both
languages and both event directions. Six acceptance conversations comprise
four controlled conversations and two independently worded free dialogues.
Each has eight user turns; two sampling seeds are declared before training.
The stage-18/19/22/23 acceptance packets remain retired.

## Training and comparison

The first pilot is one epoch, AdamW at `3e-4`, accumulation 4, seed 20261114,
with validation every 10 updates and at the end (57 updates total). The
objective is answer-only next-token CE plus English KL weight 0.1 to the frozen
body. Native nonthinking chat serialization is preserved. Validation selection
uses the mean of the separate RU and EN token losses, giving each language
equal weight. Every improving point is retained before continuing, including
step zero; the final point is also retained.

Compare the selected point with its frozen parent and with its user bank
disabled. Read full generations alongside both summed and normalized preference
scores. A contrast family succeeds only when opposite events reverse the
answer preference correctly. Inspect language, role direction, ownership,
custody, scene callbacks, associative development, ending, repetition, script
leak and EOS. English voice and Russian continuity determine promotion together.

```sh
python training/verify_expert_data.py --tokenizer "$PARENT" \
  --data "$PRIVATE/data" --output "$PRIVATE/token-audit.json"
python training/train_experts.py --model "$PARENT" \
  --base-receipt "$PRIVATE/base-receipt.json" --data "$PRIVATE/data" \
  --output "$OUT" --epochs 1 --lr 3e-4 --accumulation 4 --eval-every 10
python training/score_expert_pairs.py --model "$OUT/best" \
  --pairs "$PRIVATE/eval/pairs.jsonl" --output "$RESULTS/pairs.jsonl"
python training/sample_experts.py --model "$OUT/best" \
  --cases "$PRIVATE/eval/acceptance-dialogues.json" \
  --output "$RESULTS/dialogues.jsonl" --seeds 42 137
```

Checkpoints include the unchanged base tensors, `experts.safetensors`, exact
expert configuration, parent identity, per-tensor body hashes and an expert-file
checksum. Load them with `ExpertQwen.from_pretrained`; a stock Qwen loader reads
only the base. The training command also retains optimizer state at each saved
point. The user corpus, candidates and evaluation records belong in private HF.

## Verification and runtime

Local verification passed 105 tests and 37 subtests, including 19 new checks.
The new tests exercise both actual Qwen3.5 attention types in a small randomly
initialized architecture, FP32/BF16 zero-output identity, nonzero full/cached
logits, user-bank isolation, reset, causal prefixes, gradients, frozen-body
hashes, serialization/reload and split leakage. A two-update CLI smoke run with
the native tokenizer saved initial, intermediate and best points; the sampler
then ran two turns in all three modes. These smoke weights are test fixtures.

NoTorch's Qwen3.5 text/vision implementation and external DOE remain the runtime
foundation. This pilot adds a new trained pre-head block and a user-bank
protocol. Its NoTorch loader, state transitions and numerical parity are the
next runtime work after model evaluation. External NoTorch experts can then
operate alongside the checkpoint-owned experts as a separately measured layer.

## First GPU pass

The first pass completed 57 optimizer updates. All 474 base state tensors
matched their initial hashes. Every evaluated point improved the declared
validation criterion and was retained, including the initial point.

| Update | RU token loss | EN token loss | Equal-language mean |
| --- | ---: | ---: | ---: |
| 0 | 3.539743 | 3.608244 | 3.573993 |
| 10 | 3.535763 | 3.607048 | 3.571406 |
| 20 | 3.525840 | 3.603108 | 3.564474 |
| 30 | 3.515033 | 3.599778 | 3.557405 |
| 40 | 3.507829 | 3.592980 | 3.550404 |
| 50 | 3.499133 | 3.586680 | 3.542907 |
| 57 | 3.493491 | 3.581815 | 3.537653 |

Step 57 is the validation-selected candidate. Pair scoring and full-dialogue
review follow this checkpoint. The selected primary remains stage 7 during
evaluation. Private evidence lives under dataset `ataeff/Dubrovsky`,
`additions/2026-11-14-embedded-experts24/results/`; trained candidates are in the
private model repository under `working/embedded-experts24-step-NNNNNN`.

All seven remote candidate copies passed their weight-file and expert-file
hash checks. Their 474 frozen-body tensor hashes match. Every one of the
210,508 expert parameters changed between step 0 and step 57. The selected
candidate is pinned at model revision
`e6cec81c5243d47087da4f09a7719b5766abcca7`, subfolder
`working/embedded-experts24-step-000057`; expert-file SHA-256:
`416e049e68dd03918825f40da767c2eb717c9ce10891bd108cbf00c7041bbc6d`.

The first evaluation launch failed before loading weights because its private
preparation script wrote a JSON array while the preference reader expects
JSONL. The eight pairs were reserialized without changing their contents;
all 16 alternatives passed native-tokenizer encoding (maximum 586 tokens),
and a CLI scoring smoke check completed all three modes. Training was not
repeated. The evaluator now logs each completed answer and the worker backs
up partial generations during the run. The replacement evaluation job restores
the pinned saved checkpoint; its artifacts are under `results/eval24/`.

### Real-parent startup checks

The dedicated A40 passed the pinned private parent's file hash check, the test
suite, and the full corpus mask audit before training. On the actual 0.8B model,
zero-output experts gave maximum logit difference **0** on both the English
(36-token) and Russian (40-token) checks. The user bank stayed unchanged during
the subsequent assistant token. The model has 474 frozen state tensors and
210,508 trainable expert parameters. The recorded environment is PyTorch
`2.8.0.dev20250319+cu128`, Transformers `5.22.1`, CUDA 12.8, NVIDIA A40.

### NoTorch connection point

At NoTorch commit `374e1710d83ac3745ba21ede3e11745768062ef6`,
`harness/arch_qwen35.c:q35_forward_all` applies the final RMS normalization to
only the last token before producing logits. This expert block needs the
normalized features of **every** prefill token to reconstruct its recurrent
state. The native port therefore needs an all-token pre-head hook, explicit
user-content masks and separate per-dialogue state; the existing last-token
hidden output alone is insufficient for this transition. Existing DOE residual
hooks remain available at their current layer positions. Port tests will compare
all-token versus cached execution and resets before combining the two systems.
