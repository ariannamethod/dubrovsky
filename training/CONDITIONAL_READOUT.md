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
