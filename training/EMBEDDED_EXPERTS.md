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
  --pairs "$PRIVATE/eval/pairs.json" --output "$RESULTS/pairs.jsonl"
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

GPU results and candidate selection are recorded below after the first run.
