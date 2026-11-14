# Expert training duration: stage 26

The stage-25 user bank passed assistant-text and length invariance, while
complete owner/keeper assignments remained at 1/12 per language. Its validation
loss was still falling at the last update. Stage 26 extends the same training
curve from one epoch (57 updates) to four epochs (228 updates).

## Fixed experiment

Start fresh from the same frozen stage-7 parent and expert seed 20261114. The
trainer, expert architecture, isolated-user source, 225 training records and
54 validation records are unchanged from stage 25. Keep lr 3e-4, accumulation
4, EN KL 0.1 and evaluation every 10 updates. The complete body remains frozen;
only 210,508 expert parameters train. Every improving validation checkpoint
and every epoch boundary is retained, including step 57.

The first epoch is replayed rather than restarting the optimizer at its end.
Compare the initial expert hash, step-57 hash and all first-epoch validation
losses against the saved stage-25 receipts. Then compare step 57 with the
validation-selected late point on the same fresh generation scenes.

```sh
python training/train_experts.py --model "$PARENT" \
  --base-receipt "$PRIVATE/base-receipt.json" --data "$PRIVATE/data" \
  --output "$OUT" --memory-source isolated-users --epochs 4 --lr 3e-4 \
  --accumulation 4 --eval-every 10 --seed 20261114
```

## Acceptance fixed before training

Use six new eight-turn conversations on seeds 42 and 137. Four use the
controlled gift/loan template with unseen names and object family; two are
new free scenes with topic changes, callbacks, temporary custody and returns.
Stage-24 and stage-25 acceptance scenes are retired.

Evaluate base, experts and bank-off modes at the validation-selected point.
Evaluate experts at step 57 and the final point whenever distinct. Each mode
keeps its own generated conversation history. This produces 384 replies when
best and final coincide after step 57, and up to 480 when all three evaluated
points differ. Use temperature 0.8, top-p 1, top-k 20, generated-token presence
penalty 0.8, repetition penalty 1 and a 192-token reply budget.

Read both roles at turns 5 and 8, with exact answers and reasons preserved.
A complete pass requires the correct owner and current keeper without a
contradictory assignment elsewhere in that answer. Read grammar, character,
scene completion, topic return and feminine first-person Russian separately.
Report language, seed and controlled/free scene groups separately. Eight
fixed-history contrast pairs remain a separate measurement. Repeat the four
trained source-invariance probes at each evaluated checkpoint.

Stage 7 remains the selected release until generation review. Raw private
scenes, training data and candidate weights remain in private Hugging Face.
The dedicated stage-26 worker stops its own GPU on completion or error.

## Pins and current state

Source commit: `c0d1bc8c0b3a4bf3b4f04ab605325afe62031a8f` (merged PR 30).
The parent, train and validation hashes are the same as in
[stage 25](ISOLATED_USER_MEMORY.md). The sealed stage-26 payload is in private
dataset `ataeff/Dubrovsky`, revision
`b2ed5d38b9ffab1430e19887782cb39e21d1c32c`, path
`additions/2026-11-15-embedded-experts26/payload.zip`, SHA-256
`0f8f1611e7636ab064f7320ab735cace4836a6e81869d8a79741d9257fd6251e`.

Local preflight passed 113 tests. All 16 fixed-history contrast alternatives
and 48 acceptance questions pass native-tokenizer encoding. Training and
acceptance results will be appended after the dedicated A40 run completes.

Concept and direction: Oleg Ataeff. Implementation and verification: Astra.
