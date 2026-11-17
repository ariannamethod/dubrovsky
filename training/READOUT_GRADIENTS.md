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
