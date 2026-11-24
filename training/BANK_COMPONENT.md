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
