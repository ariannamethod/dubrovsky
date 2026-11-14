# Isolated user memory: stage 25

Stage 24 measured a path from previous assistant text into the bank through
Qwen's contextual features at the next user token. This pass changes the source
of bank content and write gates. Four experts, the router, the frozen stage-7
body and the training corpus remain the same size and identity.

## Format 2

The native tokenizer builds two causal streams:

- Full conversation, including system, user and assistant messages, for the
  working state and vocabulary prediction.
- User messages alone, preserving their order and native user-role boundaries,
  for bank content and retention/write gates. It contains no assistant replies
  or system message, and adds no generation header.

The frozen Qwen weights encode both streams with independent internal state.
User-content token IDs must match exactly across streams. The source hidden
states are aligned back to the corresponding write positions in the full
conversation; all remaining positions leave the bank unchanged. The router
and working state continue reading the actual full dialogue.

At a new user turn the sampler reconstructs both complete prefixes. During
assistant generation, only the full-conversation cache and working state
advance. A cached call containing new user writes is rejected: that call must
reconstruct the complete prefixes. All input streams refuse silent truncation.

Checkpoint format 2 records this source policy. Format-1 checkpoints still load
with their original contextual writes. Source tensors are computed at inference
time; the added trainable parameter count remains 210,508. Two frozen forwards
are now required during full-prefix encoding. The separate user-derived stream
must also be implemented in the eventual NoTorch port.

## Controlled pass

Training and validation are byte-identical to stage 24:

| Split | Records | SHA-256 |
| --- | ---: | --- |
| Train | 225 | `24dae71cd015d61921831aba4febd9ce24d8729558a3c4b393b0f9f5f1ed2e78` |
| Validation | 54 | `a05a101e369ecb2ddbabe171fd8f9327353425465484e2969a8c8462baac5047` |

Start fresh experts from the same seed 20261114 and the same frozen stage-7
parent. Use one epoch, 57 updates, lr 3e-4, accumulation 4, EN KL 0.1 and
evaluation every 10 updates. Save every improved point and the final point.
Selection remains the mean of the separate EN/RU token losses. This isolates
the memory-source change; corpus expansion follows its measured comparison.

Fresh acceptance uses six eight-turn conversations on seeds 42 and 137 in
base, experts and user-bank-off modes. Four are the existing controlled
templates with unseen object/name families; two are newly authored free
conversations. Eight fixed-history contrast pairs cover two new families in
both languages and event directions. Stage-24 acceptance scenes are retired.
Per-seed role readings, language, character, scene continuation, repetitions,
script leaks and EOS are retained beside all literal replies.

```sh
python training/train_experts.py --model "$PARENT" \
  --base-receipt "$PRIVATE/base-receipt.json" --data "$PRIVATE/data" \
  --output "$OUT" --memory-source isolated-users --epochs 1 --lr 3e-4 \
  --accumulation 4 --eval-every 10 --seed 20261114
python training/probe_expert_memory.py --model "$OUT/best" \
  --cases "$PRIVATE/eval/context-write-cases.json" \
  --output "$RESULTS/invariance.json" --allow-assistant-length-change
```

## Verification before training

The suite passed 113 tests and 37 subtests. New checks cover assistant text and
length changes, changed user facts, learned gate invariance after optimizer
steps, source-ID rejection, cached-user rejection, full/cached assistant logits,
BF16 invariance, serialization and format-1 compatibility. All 279 corpus rows
passed the native-tokenizer audit with identical user-content IDs in both
streams. All 16 contrast alternatives and 48 acceptance questions encode; the
largest prepared pair is 580 tokens.

On the real 0.8B frozen body, zero-output format-2 experts preserve exact EN/RU
logits, and the bank remains unchanged during assistant generation. The body
has 474 state tensors. Training results and trained invariance checks will be
recorded after the pass; stage 7 remains selected meanwhile.
