# Stage37 — learning the isolated user writer

Stage36 preserved the user writer and gained no conditional source switches.
Omni's 2026-11-26 readout-transport report found that improved opposite-role
contrast can coexist with worsening wrong answers through a common name shift.
We therefore record both full-name weak-side margins and contrast changes,
alongside actual repairs and losses of previously correct decisions.

## Matched pilot

Both arms start from source4-step16, SHA256
`430e14ab0bde9f0975994d1b96987bfc07191125b97102c85f1d801dff331da8`, gain4.
Sixteen updates per arm, seed2026112637, fresh AdamW, no weight decay, clip1.
Both train memory_read/router_memory at 1e-5 and output at 1e-3. The writer arm
also trains source, retain/write matrices and biases at 1e-5. All other expert
parameters and the entire selected Qwen body remain frozen.

Use the same centered source objective, 0.1 common-centre penalty and 0.2 authored
replay (0.1 base KL) in both arms. The task uses native first-name margins; the
assessment uses native complete-name probabilities. Role-plus-centre-penalty gradient norms
are recorded before adding replay gradients. Neither ordinary score movement
nor validation loss selects a conversation endpoint.

The new differentiable bank replay executes every isolated user write, including
question tokens, and omits non-writing positions. The actual full-context working
prefix is reconstructed at current weights and detached. Own and counterfactual
banks are injected into that same working state, retaining their writer graphs.
At inference the existing format3 graph is unchanged; no counterpart is required.
Verify native bank equality on each replay. Before optimization, compare writer
gradients through the full native block and the bank replay on actual model
features, require nonzero role gradients, and verify shared detached working state.

## Data and controls

TRAIN has 16 pairs: two historical TRAIN object families × owner/keeper × original
or cyclically renamed people × original or reordered mention syntax. Event order
and labels are preserved. These two factors are fully crossed. This differs from
stage36's eight-pair inventory, so the causal comparison is between the two
matched stage37 arms. Four historical validation pairs stay separate; two unused
historical training families are retained in the input record without optimization.
Eight unchanged authored English replay examples preserve the lineage.

Parent gain-only controls are .125,.5,1,2,4,8. The existing conditional gate requires
one newly complete four-decision pair repaired both in actual scores and with the
old centre on the same pair, beyond gain-only controls, preserving all correct
parent own-name decisions. Every checkpoint, optimizer and RNG is retained,
including new best validation points. Step16 is the prespecified reading endpoint.

## Complete reading

Read stage7, source4, frozen16 and writer16 on eight historical fixed records,
a new twelve-turn English theatre/lantern scene, a three-turn Russian guard and
two character prompts. English recalls at turns3,5,7,11 cover loan, gift, return
and later loan. Minimum 3/4 complete current pairs, exceeding parent/control with
no losses against stage7, correct handovers and no contradictory human assignments.
Read all intermediate turns for sustained images, callbacks, endings and the
first/third-person authored voice. Metaphorical ownership receives contextual
reading; a literal keeper substitution receives its exact evidence.

Three English and one Russian actual stage7 prefixes are repeated unchanged for
all expert arms. Writer16 also receives full memory-off trajectories at those
prefixes. Their causal comparisons are separate from independently evolving
conversations. English is primary; Russian is a retention guard.

Verify all frozen body tensors and per-arm frozen expert parameters. Repeat bank
invariance under assistant text and length substitutions for parent and both
endpoints. Archive inputs, outputs, readings and checkpoints privately, verify
collection and stop/delete only the owned bounded GPU. Stage7 remains selected.
No stage37 training has run at the protocol commit.

## Completed pilot — 2026-11-26

Executed commit `6dc8d5327468055bdecc93860011f728cf90866b` completed both sixteen-update
arms and all 116 responses on Secure Cloud A100 SXM4 80GB. Stage7 stays selected.

On two real initial sources, native bank values and all five writer-parameter
gradients match the full native block exactly (maximum error zero), with common
detached working state. Role-plus-centre-penalty gradients are nonzero for all
writer blocks throughout training, before authored replay is added. Only the
writer arm changes writer parameters. Native bank parity is checked on every
training/evaluation replay.

| Measure | Source4 | Frozen16 | Writer16 |
|---|---:|---:|---:|
| Native own names |18/32|18/32|18/32|
| Native donor names |14/32|14/32|14/32|
| Both source decisions on one prefix |0/32|0/32|0/32|
| Complete four-decision pairs |0/16|0/16|0/16|
| Historical validation own names |5/8|5/8|5/8|
| Historical value NLL |3.935774|4.008661|3.975258|
| Previously failed weak sides improving |—|11/32|18/32|
| Opposite-source contrasts improving |—|16/32|17/32|
| Mean weak-side change |—|−.153029|+.060890|
| Mean contrast change |—|+.004922|+.001714|
| Mean absolute centre change |—|.447705|.312010|

No formerly correct own or donor decision is lost; no decision is repaired.
Both conditional gates fail. Gain-only2 gives 19/32 own names and one individual
switch, with no complete pair. Neither trained endpoint exceeds that control.
The factor breakdown is unchanged: owner 8/16, keeper 10/16; original names
10/16, renamed 8/16; original/reordered syntax each 9/16.

All 34 checkpoint/optimizer/RNG points are retained. Best trained validation is
frozen-step6 (3.938123) and writer-step1 (3.941923); their conversations were not
generated. Reading remains attached to the prespecified step16 endpoints.
Fixed-output first-token exclusion counts with rounding allowance are train
own/donor 1/4 for parent and frozen, 1/5 for writer; historical validation 2/5
for all three states.

### Complete reading

| Mode | Historical current pairs /8 | Pair and compatible history /8 | Fresh EN recalls /4 | RU final |
|---|---:|---:|---:|---:|
| Stage7 |2 + 1 ambiguous|0|1|0/1|
| Source4 |4|4|0|0/1|
| Frozen16 |4|2 + 2 unclear|0|0/1|
| Writer16 |3 + 1 ambiguous|3|1|0/1|

The historical eight records contain six unique generation prefixes. All twelve
English turns per mode, three Russian turns and both character prompts were
read with their actual preceding histories, with mode labels visible. Ambiguous
metaphorical references and implicit custody are recorded separately.

Writer16 restores the correct human pair in the last English reply despite its
own preceding wrong handover. It omits the requested latest loan and drops the
curtain/staircase ending. It also loses stage7's correct return recall. The sealed
3/4 minimum and complete scene criteria are not met.

On three identical actual stage7 English prefixes, all expert arms have 1/3
current pairs. Writer's apparently correct return reply then reverses the
meaning of the return: it says ownership changed while custody stayed the same.
Thus writer has 0/3 pair-plus-history against 1/3 for parent/frozen. Full memory-off
gives writer 0/3 current pairs. Two of four matched EN/RU replies are identical
with and without memory. Russian fails the final pair in every mode and retains
substantial agreement/syntax defects; one EOS-terminated reply leaves a quote
unclosed. All 116 replies reach EOS.

The literary reading preserves first/third-person voice, a blackout paid for
appearing in credits after disappearing into an unsolved problem, the staircase's
employment, sleeves as waiting periods and a feature branch for the pause before
a kiss. It separately records an invented theatrical actor becoming the literal
lantern keeper and later owner in place of the user's people. Ownership of a
receipt, darkness or another object is not automatically a lantern-role error.

### Update audit and next step

The common gradient clip activates at writer steps2,3,8,10,12 (maximum preclip
norm4.101519); it never activates in frozen (maximum .561446). It also scales
readout gradients. Record this realized difference alongside the matched nominal
optimizer settings when interpreting the arms.

The training path detaches the actual working prefix, while a real parameter
update changes that prefix too. Next: cache TRAIN causal Qwen/user features,
reproduce margins, compare full-prefix and detached-working derivatives,
decompose bank/working/readout contributions, and replay actual clipping scales.
Check predicted versus actual 25/50/100% steps with matched readout movement.
Optimize the weak source side while preserving correct own and donor decisions;
retain the same-pair old-centre test and gain-only control. Fresh long English
scenes and a separate Russian guard follow numerical preflight. That next
Jacobian extraction/training has not started; stage7 remains selected.

### Verification and archive

163 tests pass on CPU and GPU. All 474 body tensors are preserved; frozen expert
parameters are checked at every point. Assistant text/length substitutions keep
the bank bitwise equal for parent/frozen/writer in EN and RU. Sixteen actual
matched prefixes retain identical histories/questions/token hashes. Eight shared
historical initial diagnostics reproduce stage36 exactly. All 129 collected file
receipts are verified. The owned GPU was stopped, deleted and verified absent.

The private archive contains all inputs, 34 points, actual answers, individual
reading notes, factor/gradient/geometry summaries, execution payload and next
protocol. No additional full-body duplicate is uploaded. The inherited transport
envelope's historical source_base_commit is retained as executed; actual git base
is `a9586a7a2dda3bb7a8564ac6356b99cf68e14bc5`, source_commit identifies the executed
code, and its payload file hashes are verified.

Verified private HF commit: `81938e9328247888aa0f9170b7d4c80f7015e1a6`, under
`ataeff/Dubrovsky/research/stage37-isolated-writer`. All three remote files match
local bytes and SHA256:

- Reading: `90793f0b1af4c5efd8432b99d81e0dd307a1862cd58ca9a99d8877ef0a2b3a94`.
- Results ZIP: `dfef3ebb9afa98678cf7de32137e3816f0e8ea48d06908fe77837fb6dac55bf0`.
- Manifest: `e0009c7f523a56cd1d045ad286d89bb0e3f4bbcc32690a1941944a2246cde3a6`.

The ZIP has 170 checked entries and passed credential scanning.
