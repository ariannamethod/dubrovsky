# Paired bank objective and actual Adam displacement: stage 30

Stage 29 measured a median CE/contrast gradient-norm ratio of 11.6266 at the
stage-28 step-24 expert checkpoint. Stage 30 first reconstructs those eight
training batches with the saved step-24 Adam moments. For coefficients 1 and
12, restore the same expert tensors and optimizer state before each temporary
update. Record the global clipping scale, real parameter displacement, its
projection onto the bank-objective gradient, and the recomputed objective after
the step. Restore all temporary updates before continuing. Inspect this audit
before permitting the bounded training comparison.

The coefficient 12 is the rounded stage-29 training-only median. It is fixed
before validation and new full-answer evaluation. The two arms continue from
the same step-24 weights and optimizer for 24 updates. Only this coefficient
changes. Preserve stage-28 training pairs, authored English replay, learning
rates (reader/router 1e-5; output 1e-3), replay weight 0.2, replay KL 0.1,
contrast target 0.05, clipping at 1 and the reconstructed batch schedule.
The body and isolated-user writer remain frozen. Native production inference
remains BF16; the causal auxiliary objective retains its FP32 addition/head.
Prefix working states are recomputed with current parameters and detached at
the final user question, as in stage 28.

Save experts and optimizer at steps 0, 4, 8, 12, 16, 20 and 24 in both arms.
Evaluate the existing held-out values at those points; review both endpoints
without selecting between them by validation NLL. Retain every earlier point.
Measure the common and varying parts of the newly added expert correction,
using means fitted on training positions only and held-out evaluation positions.
This follows the useful diagnostic from AMOS's readout-conditioning study;
Dubrovsky uses its own bounded residual and recurrent prefix, unchanged.

Fresh acceptance comprises four English five-turn scenes, one Russian
three-turn retention scene and four counterfactual English pairs (eight fixed
prefixes). Compare stage 7, start24, coefficient 1 and coefficient 12. Generate
the two endpoints and controls from their actual evolving dialogue histories.
Use the identical fixed prefixes and seeds for the matched comparisons. At
coefficient 12, also replace only the user bank after the final user question
with the paired donor bank or zero bank, retaining recipient Qwen features and
working state. These two controls measure the local bank contribution at that
boundary. Read all 140 complete answers with their source histories.

For release consideration require at least 6/8 English late recalls with both
current roles correct, improvement over start24 and coefficient 1, at least 3/4
matched pairs fully correct on both sides, and no contradictory role assignment
inside a credited answer. Read image development, callbacks, endings, repetition,
and the authored first/third-person mixture in every full scene. Inspect the
Russian retention scene without adding Russian training. Stage 7 remains
selected until this contextual review supports a successor.

Run tests locally and on a single owned Secure Cloud GPU. Check every frozen
tensor, collect and verify all private results, and terminate the owned pod.
Keep credentials, private histories and weight files outside git.

Concept and direction: Oleg Ataeff. Implementation and reading: Astra.
