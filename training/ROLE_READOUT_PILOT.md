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
