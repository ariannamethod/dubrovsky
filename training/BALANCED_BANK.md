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

## Completed run: 2026-11-24

Both arms completed 24 updates on a Secure Cloud RTX 4090. A40 and A6000
allocation attempts returned no capacity. The executable protocol was commit
`23bc3ccd26a28f1b386166309b2a62e8f3cf79d7`. All 129 tests passed locally and on
the GPU. All 474 body tensors and the ten frozen expert tensors were preserved.
All fourteen saved optimizer states have the expected step counters. The owned
pod was stopped and terminated after verified collection.

The eight training batches gave a median raw CE/contrast norm ratio of 11.01
on this GPU, compared with 11.63 in stage 29 on A40. Numeric replay differs
between these runs; both new arms share the same environment, source weights,
Adam moments and batch schedule.

| Initial one-step audit | Weight 1 | Weight 12 |
| --- | ---: | ---: |
| Recomputed bank loss improved | 6/8 | 8/8 |
| Mean change in bank loss | -0.00013312 | -0.00045370 |
| Batches clipped | 0/8 | 3/8 |

The first-order projection of each real Adam displacement agreed with the
recomputed objective's direction in all sixteen audit probes. During the full
pilot, that projection was negative in 18/24 weight-1 updates and 24/24 weight-12
updates. Full objective recomputation after each step was performed in the
initial audit; the training-step count records the displacement projection.

| Held-out role-value diagnostic | Start24 | Weight 1, step24 | Weight 12, step24 |
| --- | ---: | ---: | ---: |
| Native value NLL | 4.62598 | 4.65407 | 4.65225 |
| Correct value preference | 5/8 | 5/8 | 5/8 |
| Desired donor-bank direction, native | 1/8 | 0/8 | 0/8 |
| Desired donor-bank direction, FP32 addition/head | 4/8 | 4/8 | 3/8 |

Direction uses the existing absolute 1e-5 floor. The common component accounts
for 81.75% and 80.74% of the new correction's training-position energy. After
subtracting the train-fitted mean, 10.68% and 10.81% of held-out correction
energy remains. These measurements use sixteen training and eight validation
role positions, with the actual bounded Dubrovsky residual and full recurrent
prefix. The intervention answers below assess the resulting behavior.

All 140 full answers and their histories were read. All reached EOS; one start24
opening nevertheless ends on an unfinished name, so EOS and a complete literary
ending are recorded separately.

| Full-answer reading | Stage 7 | Start24 | Weight 1 | Weight 12 |
| --- | ---: | ---: | ---: | ---: |
| Both current roles, without contradiction, in English late recalls | 0/8 | 0/8 | 1/8 | 0/8 |
| Both roles in matched next answers | 0/8 | 0/8 | 0/8 | 0/8 |
| Matched pairs with both members correct | 0/4 | 0/4 | 0/4 | 0/4 |
| Russian retention recall | 0/1 | 0/1 | 0/1 | 0/1 |

The eight matched positions form four counterfactual checks over six distinct
histories; the shared member of two checks is sampled with a different seed.
Every cross-arm comparison for a position has the same prefix and seed.
Replacing the final-question user bank with its paired donor left all eight
weight-12 token sequences identical. Zeroing that bank changed five answers;
none recovered the complete current role pair. Other state components remained
with the recipient for these local interventions.

The reading retains the character's associative successes. A staircase carried
up and down becomes classes of passenger tickets; a basement returns through
an elevator scene. The failure appears when those invented locations take over
the object's custody, or an invented participant replaces its stated keeper.
Another answer lists the correct two names but explicitly assigns ownership to
both, then repeats the joint ownership. It receives no complete-role credit.
The single successful weight-1 late recall supplies both correct roles and
connects the whistle to the staircase's reputation; its separate English word-form
defect remains in the reading notes. The Russian guard retains substantial
grammar and role failures. No Russian optimization was performed.

Stage 7 remains selected. Weight 12 corrected the local training direction but
failed the new-name and full-dialogue acceptance criteria. The next experiment
should isolate generalization of the learned role distinction: compare existing
and held-out names under the same source templates and the same fixed assistant
history, then introduce the changed literary history separately. Use that
diagnostic to design the next English paired-training corpus, with both name
orders and ownership/custody assignments represented. Retain the current
writer, controls and whole-answer reading. This follow-up has not been run.

For resuming an early saved point, use its optimizer state and step together
with the archived `schedule.json`. The schedule was generated before training;
the stored Python RNG state is after that schedule generation.
