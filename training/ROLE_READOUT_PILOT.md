# English role readout: stage 28

Stage 27 located a weak link between the user bank and the actual role values.
This short pilot starts from stage-26 step 228, with the selected stage-7 body
and the complete bank writer frozen. English is the training and acceptance
priority. Russian receives one three-turn retention conversation, no optimization.

Only `memory_read`, `router_memory`, and `output` train: reader and memory routing
at 1e-5, output at 1e-3. AdamW has zero weight decay; gradients are clipped at 1.
Twenty-four updates cover eight paired English source cases three times, shuffling
the pairs each cycle with seed 20261117. Each update includes both counterfactual
directions and one of eight existing authored English replay records. Their text
and provenance are preserved. Replay contributes 0.2 times answer CE plus 0.1
base KL inside that replay loss.

Four training families and two held-out families have distinct names and objects.
Each family supplies an owner contrast (gift versus loan) and a keeper contrast
(still held versus returned). The assistant history within each pair is identical
and fact-neutral. The changed user event alters exactly one current role. Training
and validation sources, replay IDs, fresh acceptance scenes and criteria are sealed
in a private input package before optimization. No private examples enter git.

The role objective is native-BF16 CE on the actual correct name tokens, excluding
the role cue and EOS, plus a causal source contrast. That contrast compares the
correct-minus-alternate summed value log probability on the same Qwen prefix and
same working prefix with the original bank versus the paired donor bank. The bank
is replaced immediately after the last user-content token. The auxiliary loss is
`softplus(0.05 - (normal_margin - swapped_margin))`, computed with FP32 addition
and projection. Both directions of the pair are trained. The prefix working state
is recomputed at current parameters without gradients; gradients enter the suffix.
The writer is fixed, so cached donor banks must remain exact.

This intentionally measures three precision paths separately: native BF16,
FP32 projection of the native corrected hidden state, and FP32 addition plus
projection. A separate detached FP32 head copy serves the auxiliary objective;
the body's weights and production inference dtype remain unchanged. All free
and matched-prefix generations use the production sampler at temperature 0.8,
top-p 1, top-k 20, generated-token presence penalty 0.8, and 192 new tokens.

Validation runs before training and every four updates. Every assessed point,
including every new best and early points, retains compact expert tensors,
optimizer and RNG states. These are deltas against the pinned complete parent;
restore the full parent before loading an expert file. The lowest native value
NLL among trained points chooses the candidate for reading, with the parent's
score shown separately. Selection of a released model requires the actual replies.

The stage-7 base, parent step 228 and this candidate each run four fresh five-turn
English conversations covering gift/loan and kept/returned roles. They switch to
an absurdist lift scene, develop the model's own image, then return to both roles
and close the lift thread. Each also runs the Russian guard and twelve identical
archived English prefixes from stage 27: 105 complete replies. Free trajectories
measure conversation quality; fixed prefixes isolate the next-answer comparison.
Old prefixes serve diagnosis and do not substitute for fresh acceptance.

Read all replies for both correct current roles, contradictions later in the same
answer, development and return of the character's own images, a complete ending,
and the authored first/third-person voice. Record ambiguity separately. Retain
stage 7 unless this reading supports a successor. No language-balancing search
or new Russian SFT belongs to this pilot.

Verify all 474 frozen body tensors and every frozen expert tensor. Save data and
code hashes, all value scores, raw generated tokens, complete histories and the
per-answer reading. Use one bounded A40; collect and verify its results before
terminating only that owned pod. Private results and expert deltas are retained
separately; Hugging Face writes remain unavailable in the operator environment.

Concept and direction: Oleg Ataeff. Implementation and reading: Astra.

## Completed run: 2026-11-17

The A40 completed all 24 updates and all 105 generations. All 124 tests passed
both locally and on the GPU. The run used the same GPU software versions as
stage 27: PyTorch 2.8.0.dev20250319+cu128 and Transformers 5.22.1. The three
trainable tensors contain 66,624 parameters. Every assessed point was preserved
at steps 0, 4, 8, 12, 16, 20 and 24, with optimizer and RNG states. All 474 body
tensors and every frozen expert tensor stayed unchanged. The twelve base and
twelve parent matched-prefix replies replayed their stage-27 token IDs exactly.

Native role-value validation NLL decreased at every assessment, from 4.923090
to 4.689764; step 24 therefore became the reading candidate. Correct-name
preference stayed 5/8 at every point. The causal source difference in summed
name margins was extremely small. At step 24 every native difference had
absolute magnitude below 1e-5; the largest was 4.77e-7. FP32 addition plus
projection changed all eight margins, but moved in the desired direction in
only 4/8, both before and after training. The raw records retain unrounded
values; summaries use 1e-5 as a reporting floor so subtraction-scale differences
are not counted as useful source direction.

All 105 complete answers received contextual reading. Every generation ended
at EOS. The fresh English cases contain two explicit role recalls per dialogue,
so their denominator is eight answers across four five-turn conversations.

| Mode | Fresh EN, complete consistent pairs | Matched EN prefixes, complete consistent pairs | RU final pair |
| --- | ---: | ---: | ---: |
| Stage 7 | 0/8 | 2/12 | 0/1 |
| Parent step 228 | 0/8 | 1/12 | 0/1 |
| Readout step 24 | 0/8 | 1/12 | 0/1 |

The candidate changed 10/12 matched-prefix answers from its parent. It retained
the one clean telescope ending, and retained failures where a correct first
role statement was contradicted later in the same answer. One fresh candidate
ending correctly placed the object with its recipient, but its later ownership
pronoun had several competing antecedents; it is recorded as ambiguous, not as
a clean pair. Final owner/keeper fields in the private generation records refer
to the final scene state. Initial turns are read against their own user events,
before any later return.

The character retained specific associative continuations. In one candidate
conversation a wet operator develops into an underwater ascent. In another,
the object returns to its own earlier patrol motif and the lift develops a
workplace story. In a third, the candidate repeats its own allocation of object,
stairs and air after the user explicitly changes who holds the object. These
are distinct outcomes: self-generated image continuity can coexist with a
failure to update the factual anchor. Russian still loses the scene switch and
contains the pre-existing grammar and role problems; this run did not optimize
Russian.

Stage 7 remains selected. The short readout pilot reduced value NLL without
improving the complete role answers. The next English diagnostic should measure
the separate gradients of native value CE, the paired bank contrast and replay,
and inspect role-token preferences at natural answer openings as well as the
fixed role cues. This will set the next training adjustment at the point where
the source difference should control a name during free continuation. Preserve
the current writer and authored voice while making that adjustment. No next
training run is included in these results.

The worker's post-run receipt builder failed when it attempted to hash a
checkpoint directory as a file. The training completion record, all answers,
and all seven checkpoints were already present. The operator collected the
authenticated archive, verified every file recursively, checked the saved
optimizer steps and frozen tensors, and retained both the failure record and
the corrected receipt builder. The owned pod was stopped and terminated after
verification. Private transcripts, per-answer notes and resumable expert deltas
are retained separately from git. HF publishing remains unavailable, so these
new deltas have not been published there.
