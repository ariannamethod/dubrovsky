# Objective balance and natural role openings: stage 29

Stage 28 reduced native value NLL without improving full role answers. Stage 29
measures the contribution of its three objectives before another training change.
There are no optimizer steps and stage 7 remains selected.

Use the pinned step-228 parent and saved stage-28 step-24 experts. Reconstruct the
first eight stage-28 batches, preserving the pair/replay association. For each,
measure separate gradients of native role CE, FP32 causal bank contrast, and the
authored replay with its original base KL. Report norms per trainable tensor,
unweighted and with the actual 0.2 replay coefficient, plus cosine directions
against CE and their combined gradient. These are objective gradients before
clipping and Adam preconditioning. Preserve every parameter byte.

On the same eight held-out role sources, compare the original role cue, a natural
sentence opening, and an opening containing one Dubrovsky sentence. Score only
the value tokens under the native chat template, excluding the opening and EOS.
Record native BF16, FP32 projection, and FP32 addition plus projection. Preserve
the original Qwen context and working state when swapping only the user bank at
the end of the final user question. Use an absolute 1e-5 floor for source-margin
direction summaries; retain all unrounded values.

Generate 32 complete English replies: eight sources, two checkpoints, original
or donor bank, identical seeds and prefixes. Normal replies are evaluated against
the user events; donor-bank replies are inspected for role movement toward the
donor, contradictions and preservation of the authored voice. These are controlled
next-answer diagnostics on stage-28 validation material. Read every answer fully.
No fresh candidate is trained or promoted by this diagnostic.

Reuse the production generation settings and frozen isolated-user writer. Run the
existing tests plus checks of gradient decomposition and parameter preservation.
Collect all private results, verify their hashes, then stop and terminate the owned
Secure Cloud GPU. Keep sources and checkpoints outside git. The corrected recursive receipt
builder from stage 28 is used here.

Concept and direction: Oleg Ataeff. Implementation and reading: Astra.

## Completed run: 2026-11-17

The run completed on an A40 in Secure Cloud, with the same PyTorch
2.8.0.dev20250319+cu128 and Transformers 5.22.1 versions as stage 28. All 126
tests passed locally and on GPU. There were zero optimizer steps. All 474 body
tensors and both expert checkpoints remained unchanged. The original role-cue
margins and bank differences replayed stage 28 exactly in all three precision
paths, for both checkpoints. The owned pod was terminated after verified collection.

| Checkpoint | Median CE / bank-contrast gradient norm | Range across 8 batches | CE opposed to contrast |
| --- | ---: | ---: | ---: |
| Parent 228 | 15.89 | 6.40–25.23 | 1/8 |
| Readout 24 | 11.63 | 4.58–18.57 | 3/8 |

At step 24, the complete weighted gradient was opposed to the bank-contrast
gradient in 2/8 batches. The contrast is present but receives a much smaller
contribution to the summed gradient. Its relative size differs by tensor:
step-24 median CE/contrast norms were 0.06464/0.01535 for the reader,
0.08161/0.03184 for memory routing and 0.46048/0.02919 for output. Median weighted
replay norm was 0.00256. These measurements are before clipping and Adam.

| Checkpoint / opening | Correct value, native | Correct value, FP32 add + head | Source direction, native | Source direction, FP32 add + head |
| --- | ---: | ---: | ---: | ---: |
| Parent / cue | 5/8 | 5/8 | 1/8 | 4/8 |
| Parent / natural | 5/8 | 5/8 | 1/8 | 5/8 |
| Parent / character | 6/8 | 6/8 | 0/8 | 4/8 |
| Readout / cue | 5/8 | 5/8 | 0/8 | 4/8 |
| Readout / natural | 5/8 | 5/8 | 1/8 | 3/8 |
| Readout / character | 6/8 | 7/8 | 0/8 | 0/8 |

The character opening improved value preference while its donor-bank direction
did not improve. Opening-conditioned value preference and control by the bank
therefore remain separate recorded outcomes. All raw values and token spans
are retained; the direction floor is 1e-5.

All 32 full English replies were read and reached EOS. Bank replacement changed
5/8 parent replies and 1/8 readout replies. Neither checkpoint produced a clear,
consistent adoption of the full donor pair in any of the eight interventions.
Normal replies supplied two clear current role pairs out of eight for each
checkpoint. One of those answers also invented occasional loans outside the
current state; that narrative defect is retained in the reading notes. Abstract
imagery was assessed in context: an unnamed giver/receiver/keeper triad following
an explicit correct pair was not treated as an automatic contradiction.

Several failures gave the ownership and custody roles to images from the earlier
assistant history: an afternoon owned the object and a window kept it. Those
same images persist with the altered bank. Other replies clearly distinguish
legal ownership and physical custody, but do not change that distinction toward
the donor. Every answer, including unchanged intervention replies, has a reading
note in the private report.

Stage 7 remains selected. The next short training comparison should change the
relative contribution of the paired bank objective, preserving the writer,
source pairs, authored replay, native inference and fresh English acceptance.
Record norms separately per tensor because their imbalance differs. Keep the
natural-opening probes as diagnostics so opening changes are not mixed into
the same training comparison. No such additional training was run in stage 29.
