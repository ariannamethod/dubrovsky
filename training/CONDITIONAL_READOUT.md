# Stage36 — output geometry and conditional source learning

Stage35 found contraction of old bank differences through question features and
weak source-name switching even with an earlier bank. The newly reviewed Omni
output-geometry audit suggests checking reachable output directions and removing
common name shifts from the paired training target before changing our writer.

## Geometry

Our block mixes four expert proposals **before** RMS clipping. Its output lies
in the joined row space of all four matrices, with Euclidean radius
`gain * max_delta_rms * sqrt(hidden_size)`. For head-row difference `w`, use the
minimum of that radius times its projection into the joined space and
`gain * max_e ||O_e w||_1`. Each working coordinate is bounded by tanh. A per-expert
subspace maximum would miss cancellation between expert proposals and is not
used. Report both real-valued exclusion and a conservative BF16 cast/add/head
rounding envelope, with FP32 accumulation allowance. Archive exact raw/head rows.

## Matched training

Start both arms from the stage34 source4-step16 delta (SHA256
`430e14ab0bde9f0975994d1b96987bfc07191125b97102c85f1d801dff331da8`), fixed gain4,
with fresh matched AdamW moments, eight updates, seed 2026112536. Keep the body,
user-bank writer, working recurrence, contextual input projection and routing
fixed. Train memory_read/router_memory at 1e-5 and output at 1e-3; no weight decay,
clip norm1. Same eight historical training pairs, four historical validation
pairs and eight authored English replay examples. Every update is retained with
optimizer and RNG; validation loss does not select the conversation endpoint.

For one actual Qwen/working prefix, `m` is the native first-name margin with its
own bank and `s` with the counterpart bank. Let `c0=(m_parent+s_parent)/2` and
`d=(m-s)/2`. Ordinary task loss is
`(softplus(1-m)+softplus(1+s))/2`. Centered task loss replaces the margins by
`c0+d` and `c0-d`; its common-shift gradient is zero. Both arms add the identical
0.1 squared shift of the pair centre and 0.2 authored replay with 0.1 base KL.
The counterpart is used only by training and diagnostics. Native inference,
format3 and the actual checkpoint output remain unchanged.

Before training, evaluate parent gains .125,.5,1,2,4,8. Report native full-name
own/donor decisions separately from first-token training. The conditional gate
requires a newly complete four-decision pair repaired both in the actual output
and with its old pair centre, **on the same pair**, beyond every gain-only control;
all formerly correct own-name decisions must survive. This is a diagnostic gate.

## Reading and acceptance

Both fixed step8 endpoints receive complete reading alongside stage7 and the
source4 parent: eight historical fixed replies, a new 12-turn English scene with
an actual gift, return and loan, a separate three-turn Russian guard and two
character prompts. English recall checkpoints are turns3,5,8,11 (zero-based).
Minimum: 3/4 current role pairs, exceeding parent/control without loss against
stage7, no incompatible human assignment or invented transfer, connected image
development, callbacks, complete endings and authored first/third-person voice.
Read the entire scene, including the unscored turns.

Three English and one Russian stage7 prefixes are also supplied unchanged to
parent and both endpoints; centered additionally receives a full memory-off
trajectory on those prefixes. Report these causal comparisons separately from
free-running conversations. Metaphorical rights, object self-ownership and
Dubrovsky's narrative interventions require contextual reading, alongside the
literal user transfers. English is primary; Russian is a separate guard.

Verify all 474 frozen body tensors, frozen expert parameters, assistant-history
bank invariance and native source-prefix parity. Archive inputs, all points,
actual answers and readings privately. Collect and verify the bounded owned GPU,
then stop and delete it. Stage7 remains selected until the full review supports
a replacement. No training has run at protocol commit time.

## Completed pilot — 2026-11-25

Source commit `66b3e667877b006a1ac3b0d2fbb1aada5a8c72ed` completed both eight-update
arms on Secure Cloud A100 SXM4 80GB. All 116 replies were read with their actual
histories. Stage7 remains selected; neither endpoint passes the conditional gate.

### Output reachability

For the source4 parent at gain4, the fixed-matrix bound plus the conservative
BF16 rounding envelope excludes two own targets and two donor targets among
16 training positions, and two own/five donor targets among eight historical
validation positions. These are first-divergent-name-token bounds. At gain8,
training has no such excluded positions; validation still has one own/two donor
exclusions. Ordinary8 reduces the training own exclusions to one, centered8
leaves two; both leave historical validation exclusions unchanged.

Current GPU raw/head geometry reproduces the archived-feature calculation with
maximum difference 1.25e-14. This uses the combined rank64 output space, not a
maximum of individually clipped expert subspaces.

### Source decisions and preservation

| State | Train own | Train own/donor switches | Four-decision pairs | Validation own | Validation NLL |
|---|---:|---:|---:|---:|---:|
| Source4 parent |10/16|0/16|0/8|5/8|3.935774|
| Ordinary8 |10/16|0/16|0/8|5/8|4.063051|
| Centered8 |10/16|0/16|0/8|5/8|3.969361|

Both endpoints retain the same ten correct training decisions, with no repairs.
Mean full-name half-contrast changes are .009676 (ordinary) and .019550
(centered); mean absolute centre changes are .466557 and .449227. Removing the
shared task gradient still leaves centre movement across the full inventory as
shared parameters and replay change. No same-pair conditional repair is gained.

The gain-only controls yield 8,8,9,10,10,11 correct own names at
.125,.5,1,2,4,8, respectively. Gain2 produces one individual source switch;
no gain produces a complete four-decision pair. All 18 states with optimizer/RNG
are retained. Best trained validation points are ordinary-step1 (3.932117) and
centered-step4 (3.909071); their conversations were not generated here. The
complete reading applies to the prespecified step8 endpoints.

### Complete contextual reading

| Mode | Historical current pairs | Pair and transfer history | New EN recall pairs | RU final pair |
|---|---:|---:|---:|---:|
| Stage7 |2/8 + 1 ambiguous|0/8|0/4|0/1|
| Source4 parent |4/8|4/8|0/4|0/1|
| Ordinary8 |4/8|4/8|0/4|0/1|
| Centered8 |4/8|2/8|0/4 + 1 ambiguous|0/1|

The eight historical records contain six distinct generation prefixes: two
histories occur in both owner and keeper diagnostic groups. These are record
counts. The new English scene is twelve turns per mode; four recall positions
cover a loan, gift, return and subsequent loan. All intermediate turns were read.
One centered answer can be read as naming the returning holder through an
ambiguous last-arrival description; it is recorded separately. Even crediting
it yields 1/4, below the sealed 3/4 requirement.

On three identical stage7 English prefixes, parent/ordinary/centered each recover
0/3 current pairs. Centered with memory disabled throughout the trajectory gets
1/3 under a generous permanent-versus-temporary possession reading, but its
holiday duration and platform location leave the transfer unclear: 0/3 pair plus
history. The four matched Russian answers also fail to recover the full pair.

The authored voice remains present. An invented postal character grows naturally
from the window image, then becomes the literal thimble wearer in a role answer,
replacing the user's keeper. A separate ordinary reply invents a child keeper
and a gift to a girl. These concrete human substitutions are recorded alongside
successful window, roof, ticket and timetable development. Metaphorical rights
and ownership of other objects are not automatic errors. A centered orchid scene
keeps the kiss/commit lineage through release notes, verified petal contact and
returning in three dimensions. First/third-person voice is preserved.

Russian still has syntactic and agreement defects. All 116 outputs reach EOS.
Full histories, output tokens and individual literary notes are archived privately;
mode labels were visible during reading.

### Verification and continuation

158 tests pass locally and on GPU. All 474 body tensors are preserved; writer
parameters remain unchanged at all 18 points. All 48 initial native name-margin
comparisons reproduce stage35 exactly. The 16 matched prompts preserve token
hashes and histories. 78 collected file receipts are verified. Assistant-text
and length substitutions preserve the bank in both language guards.

The initial deployment omitted its volume mount path and never started the
workload. It was deleted and recreated with `/workspace`. The completed worker
and initial failed pod were both stopped, deleted and verified absent.

Next: compare frozen and trainable isolated writers under the same conditional
readout objective. Before a run, verify native bank parity, nonzero role gradients
through the user-write recurrence, and assistant-history invariance. Keep the
actual working prefix detached and identical between own/donor interventions.
Cross training names, roles and mention order; distinguish neutral continuation
from genuine gifts, returns and loans. Match data, initialization, replay and
update budget between arms; then read fresh long English scenes and a separate
Russian guard. Writer training has not run in stage36. Stage7 stays selected.

### Archive

Private `ataeff/Dubrovsky/research/stage36-conditional-readout` contains the complete
reading, results archive and manifest. The archive retains all 18 checkpoint states,
optimizer/RNG, private cases/replay/acceptance, output records, geometry inputs,
individual readings and checksums. It contains no additional full-body copy.

Verified HF commit: `1d8b786217999575cd0dd95aa35a80ffc060f267`.
All three uploaded files match their sizes and SHA256 receipts:

- Reading: `5adae5a6d3de9482a8ac9c9bb35a797593f42cfd3f661c30bf4168097344784e`.
- Results ZIP: `821b4a3de1daef9222a9ada83a24882f2de338e79dfd6715b815b3db1032fe27`.
- Manifest: `f2cd26dd0c2c7f58b47475a8456ed33352f47cd01d241a47a2d083109d280293`.

The ZIP contains 129 checked file entries and passed credential scanning.
