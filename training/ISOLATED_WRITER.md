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
assessment uses native complete-name probabilities. Role-only gradient norms
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
