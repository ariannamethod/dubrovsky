# Stage 33 — bank-specific strength and literary clarification

The stage-32 uniform gain increased train pairs correct on both sources from
1/8 to 3/8, with five corrected preferences offset by five losses. Its 0/6 late
English recall and Russian timestamp loop leave stage7 selected.

## Frozen diagnostic protocol

Read the original stage-30 lambda12 parent and the stage-32 learned endpoint on
the same eight training pairs and four validation pairs. Reconstruct the exact
bounded FP32 output from the final working state and routing probabilities;
check that its native BF16 addition exactly equals the real forward result.
Compute the corresponding output with memory disabled from the start of the
same prefix. Both streams use identical raw Qwen features.

Let `delta` be the unscaled normal correction and `zero` the unscaled correction
with the bank disabled. Compare uniform `g * delta` with bank-specific
`delta + (g - 1) * (delta - zero)` at g=1,2,4,8,16,32,64. Gain one exactly
reproduces the unscaled parent. A zero bank effect is never amplified.
All gain settings in this phase are frozen numerical diagnostics, not updates.

Record full-name native log probabilities with one-token head projections,
first-name native margins, RMS values, and FP32 first-name bank projections.
For an opposing projection, positive amplification cannot raise the correct
first-name margin. Measure the required gain when the projection is favorable.
Also record a swapped-bank intervention that keeps the actual working prefix
fixed; distinguish that intervention from the fully zero-bank trajectory.
Archive raw hidden vectors, both head rows and all three output corrections.
Validation is reported separately; any later calibration uses training only.

## Re-read the literary ambiguities

Five ambiguous answers and one explicit wrong-owner control from stage32 receive:

1. An exact replay with original prompt, seed and output mode; token IDs must match.
2. A named-object question on the same preceding history, without correct names.
3. A clarification after the actual original answer, retaining its image and
   asking for the current roles according to the user's stated transfers.

All 18 answers are read with their complete histories. These are historical
clarification diagnostics. They test whether a metaphor can be unpacked without
supplying the target names. They do not alter the old reading notes or turn old
answers into fresh acceptance examples. New readings are recorded separately.
Poetic ambiguity, a definite conflicting role, a switch to a different object,
and recovery after clarification remain distinct outcomes.

The authored first/third person mixture, associative imagery and absurdist
lineage remain part of the desired behavior. The original `He died in the
signature` scene is retained as a positive literary example. The separate
leather shoe's wish to remain unowned did not cancel Hester's ownership of the
glass shoelace in the original reading.

The complete body and expert states are frozen in this diagnostic. Preserve
all hashes, actual outputs and failed cases; collect, verify and terminate the
owned Secure Cloud GPU. The current selected model remains stage7.

## Completed audit

Run source: `98b5772b9cad9de9ab42309d8e480a681b063a23`. Both states were frozen:
48 source positions in total, 18 generated answers, zero optimizer steps.
The Secure Cloud A100 SXM4 80GB run was collected, verified, stopped and deleted.
The requested PCIe A100 had no available supply; that request created no pod.

| State and correction | Gain | Train names | Both-source train pairs | Fixed / lost from unit gain | Validation names |
|---|---:|---:|---:|---:|---:|
| Parent, uniform | 1 | 9/16 | 1/8 | 0 / 0 | 5/8 |
| Parent, uniform | 4 | 11/16 | 3/8 | 2 / 0 | 5/8 |
| Parent, uniform | 8 | 10/16 | 2/8 | 4 / 3 | 5/8 |
| Parent, bank component | 64 | 8/16 | 1/8 | 1 / 2 | 5/8 |
| Learned32, uniform | 1 | 9/16 | 1/8 | 0 / 0 | 5/8 |
| Learned32, uniform | 4 | 11/16 | 3/8 | 2 / 0 | 5/8 |
| Learned32, uniform | 8 | 9/16 | 3/8 | 5 / 5 | 5/8 |
| Learned32, bank component | 64 | 10/16 | 2/8 | 3 / 2 | 5/8 |

All grid points retain 5/8 validation names and 1/4 complete validation pairs.
The learned checkpoint's stored uniform gain is 8; the grid replaces it with
the stated gain rather than multiplying by 8 again. Gain4 has not received a
conversation review here. All 18 generated answers use the original learned32
checkpoint at uniform gain8. The archive contains the complete grid and full-name
native log probabilities, using a separate one-token head call at every position.

The unscaled normal correction has RMS .1. Mean training RMS for the normal-minus-
zero correction is .014028 (parent) and .016798 (learned32); normal-minus-donor
with the actual working prefix held fixed is .001604 and .001998. Zero-bank
trajectories and late source replacement are therefore reported separately.

CPU analysis of the archived head rows finds favorable normal-minus-donor FP32
first-token projections on 14/16 parent and 16/16 learned32 training positions,
but only 3/8 validation positions for either. This is a first-token FP32
projection; the earlier native full-name bank-swap scores remain a separate
measure. The training effect of source replacement is present and small.

Relative to the fully zero-bank trajectory, five of seven wrong unit-gain
training first tokens have favorable bank directions. Their required FP32 gains
for margin +1 range from 68.6 to 199.4 for the parent, and 48.9 to 133.2 for
learned32. The other two wrong training positions have opposing directions.
All three wrong validation first tokens have opposing bank-component directions
in both states. The native grid leaves those three choices wrong throughout.
The next intervention therefore targets source-conditioned decisions rather
than another uniform range increase.

## Contextual reading results

All six historical answers reproduced exactly, including token IDs. Every one
of the 18 outputs reached EOS and was read with its full preceding history.

| Intervention | Recoverable current human pair | Pair and user transfer history preserved |
|---|---:|---:|
| Historical replay | 0/6 | 0/6 |
| Named-object question on the original preceding history | 2/6 | 2/6 |
| Clarification after the actual historical reply | 1/6 | 0/6 |

The replay selection contains five ambiguous replies and one explicit wrong-owner
control. Those categories remain distinct; these targeted historical questions
are diagnostics rather than a new acceptance set. No correct names were supplied
in the revised questions. Previous stage32 notes are unchanged.

One named-object answer clearly supplies the correct legal owner and physical
keeper, while giving the window viewing rights. Another supplies the correct
owner and keeper of the acorn and retains the third person as an observer. Its
later phrase about three people sharing the scene does not automatically assign
a new owner. Its English defects are recorded separately.

One follow-up recovers the correct current human pair but invents the acquisition
history, replacing the user's gift with the owner's self-appointment. That is
credited in the current-role column and excluded from the transfer-history
column. Other follow-ups explicitly introduce a new human owner or turn custody
into ownership. Metaphorical ownership by a room, darkness or the object itself
is read in context, alongside human roles, not used as an automatic failure rule.

## Next controlled tuning step

Use the original lambda12 parent at fixed uniform gain4 in both arms. This
calibration comes from training positions: two repaired preferences with no loss
of the nine originally correct ones. Hold the body, user-bank writer, example
order, optimizer initialization, English authored replay and update count fixed.

The control repeats the existing objective at that gain. The intervention adds
native correct-name CE and a separate +1 name margin for the swapped-bank source:
the counterpart history's name is the target under that bank, with the actual
working prefix held fixed. This asks each source to control the correct decision,
beyond the existing small contrast between two source scores. Retain every early
point, count corrected and lost preferences separately, and review new English
scenes, fixed-prefix comparisons, complete free conversations and a small Russian
guard before any promotion. This next optimizer run has not started.

**Stage7 remains selected.** No body, writer or expert weights changed in stage33.
146 tests passed on CPU and GPU. All 474 body tensors and both expert states
were unchanged. Native readout reconstruction was exact. CPU recomputation of
all 48 feature records (240 tensors) had maximum FP32 difference 1.91e-6.
The collected bundle's 11 file receipts were checked. Full private replies,
histories, annotations, inputs, source states and feature tensors are archived.

Private archive: `research/stage33-bank-component/` in `ataeff/Dubrovsky`,
commit `8555a8f523fa54b7f5b6619e743f952d072e3550`. All three uploaded files were
verified by size and SHA256. The full reading is 56,204 bytes, SHA256
`3b37a3602c6a933ebed2299aa647e11d5663516bc898640c9fe22ac248a52896`.
The results archive is 2,422,070 bytes, SHA256
`edcea56f8967bb2f34069430d36b2cf8adf02c68ed25af2ad086a802b4bd6cdd`;
all 38 manifest entries passed verification and credential scanning.
