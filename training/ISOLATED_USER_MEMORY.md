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
has 474 state tensors. The dedicated A40 repeated the same 113-test suite,
real-parent check and all-row tokenizer audit before training.

## Completed training and source invariance

The controlled pass completed 57 updates. All 474 frozen state tensors matched
their initial hashes. Every validation point improved on the preceding best;
points 0, 10, 20, 30, 40, 50 and 57 were retained and backed up privately.

| Update | RU token loss | EN token loss | Equal-language mean | Stage-24 mean |
| --- | ---: | ---: | ---: | ---: |
| 0 | 3.539743 | 3.608244 | 3.573993 | 3.573993 |
| 10 | 3.535500 | 3.606796 | 3.571148 | 3.571406 |
| 20 | 3.526349 | 3.602848 | 3.564599 | 3.564474 |
| 30 | 3.514968 | 3.599788 | 3.557378 | 3.557405 |
| 40 | 3.508491 | 3.593277 | 3.550884 | 3.550404 |
| 50 | 3.499685 | 3.586459 | 3.543072 | 3.542907 |
| 57 | 3.493562 | 3.581413 | 3.537487 | 3.537653 |

The two curves remain close. Step 57 is the validation-selected candidate,
pinned in private model `ataeff/Dubrovsky` at revision
`b229f06ffe7badc1459255599d6d12fd39b8b6bf`, subfolder
`working/embedded-experts25-step-000057`. Expert-file SHA-256:
`d19251c948a65fe4c159481c05df86db2fca0d3496e8ce485b39c1672e50b90d`.

All seven remote checkpoints passed their frozen-file, expert-file and
per-tensor body checks. The initial expert-file hash and exported base-file
hash exactly match stage 24. All 210,508 expert parameters changed by step 57.
The GPU environment remained PyTorch `2.8.0.dev20250319+cu128`, Transformers
`5.22.1`, CUDA 12.8, NVIDIA A40.

The trained candidate passed all four source-invariance probes on the A40.
Swapping the preceding assistant's owner/keeper assignment leaves the bank
bitwise equal in EN and RU. Extending that altered assistant reply, changing
all subsequent full-conversation token positions, also leaves the bank bitwise
equal. Every bank RMS and maximum difference is zero. The full-conversation
user features still differ (RMS 0.4101–1.2033), confirming that this comparison
actually changes the working-context input. Both streams use the same frozen
BF16 body; the experts remain FP32.

## Fresh dialogue acceptance

All 288 replies completed with EOS within the declared 192-token budget.
Sampling stayed at temperature 0.8, top-p 1.0, top-k 20, generated-token
presence penalty 0.8 and repetition penalty 1.0. The three modes maintain
their own actual histories. Astra read the complete conversations and attached
72 owner/keeper verdicts to literal source lines and their file hash.

A full role pass requires both current roles without a conflicting assignment
in the same reply. Partial and ambiguous answers retain their individual reasons.

| Language / seed | Frozen base | Experts | User bank off |
| --- | ---: | ---: | ---: |
| RU / 42 | 0/6 | 1/6 | 0/6 |
| RU / 137 | 1/6 | 0/6 | 0/6 |
| EN / 42 | 0/6 | 0/6 | 0/6 |
| EN / 137 | 0/6 | 1/6 | 1/6 |

All four complete passes occur in the controlled conversations. The two free
dialogue families yield 0/24 complete role passes across modes and seeds.
English continues associative scenes and returns to earlier motifs; explicit
ownership requests are often skipped, delegated to the user, or answered with
entities from the other scene. Russian agreement errors, name changes and
malformed words remain. In the six Russian first-person-female probes, a
consistent female first-person voice appears once in base mode and once with
experts; the remaining four shift speaker/person or produce malformed forms.
This voice count is separate from full task completion.

Script counters over 96 replies per mode: base has two CJK answers and one
mixed Latin/Cyrillic answer; experts have no CJK and three mixed-script answers;
bank-off has neither. Full prior-answer repeats, full prompt echoes and the
implemented role-frame patterns occur zero times. Reused fragments remain
recorded separately; the zero counts do not describe all forms of repetition.

On the eight prepared fixed-history contrasts, all modes score RU 4/4 using
either sum or mean log probability. EN is 4/4 by sum and 3/4 by mean. Complete
bidirectional families are RU 2/2; EN 2/2 by sum and 1/2 by mean. These scores
are retained separately from the free-generation role results.

The new blind packet contains 12 shuffled A/B/C comparisons, each comprising
three full eight-turn conversations. The decoding key is stored separately in
private HF. This is an open Astra reading; independent reading is pending.
The stage-25 scenes are now retired from future candidate acceptance.

## Trained bank contribution and decision

On the six literal expert-mode final prefixes at seed 42, switching off the
bank changes final-token logits by RMS 0.02287–0.04293. The complete expert
correction changes them by RMS 0.12612–0.21022 relative to the frozen body.
The proposed final correction has RMS 0.05427–0.07260: none of these six points
hits the 0.1 output bound. Expert 2 receives 0.7756–0.9460 of the final-token
vote and 0.4678–0.5549 averaged over each prefix. These CPU measurements use
the identical literal prefix for the enabled/off comparison; body hashes remain
unchanged.

Across the generated conversations, 23/96 enabled/off answer pairs are exactly
equal, including one complete eight-turn dialogue. At the 34 turns where both
modes still have identical prior history, 11 produce different next answers.
The bank affects generation; accurate event-state readout remains the training
target.

**Stage 7 remains the selected primary.** The source-isolation repair passes
its trained-state checks. The candidate's role retention and Russian language
quality do not meet the dialogue objective. The next controlled experiment is
a longer expert-training curve on this fixed corpus and parent, retaining
early checkpoints and evaluating new conversations. Targeted corpus expansion
can then be measured as a separate change. The NoTorch port must reproduce
both source streams and expert state transitions for the selected candidate.

Private evidence: dataset `ataeff/Dubrovsky`, revision
`2f6f1de01d51921a36b5686bf75ad68e9012631e`, prefix
`additions/2026-11-14-embedded-experts25/results/`. The 288-reply file SHA-256 is
`ebf999aaffcde476e753202ddc46af05c75d150aaf5c61288e384ef997d8877b`.
The dedicated A40 stopped after evaluation and was terminated after artifact
verification; the final pod query returned no pod. Private corpora, weights,
operator files, the open reading and blind key remain outside git.
