# Stage 34 — explicit targets for each memory source

Stage33 found that uniform gain4 repaired two training name preferences without
losing any of the nine initially correct choices. Validation remained 5/8.
Normal-minus-donor FP32 first-token effects were favorable on all 16 learned32
training positions but 3/8 validation positions. The effect often moves the score
without changing which source name wins. This pilot trains the actual decisions.

## Matched optimization

Both arms start from stage30 lambda12 step24, converted to format3 with fixed
gain4. Both receive the same eight English training pairs, four historical
validation pairs and eight authored English replay examples. Both run 16 updates
with fresh AdamW states, seed 20261125, reader LR 1e-5, output LR 1e-3, no weight
decay and global gradient clipping at 1. Only memory_read, router_memory and
output are trainable. The body, writer, working-state parameters and gain remain
frozen. Every point is saved with optimizer and RNG state.

The control objective retains native own-name CE, a +1 own-name first-token
margin, add32 source contrast with target .05 and weight12, and authored replay
weighted .2 with base KL .1. The source4 arm adds weight1 native counterpart-name
CE and a separate +1 counterpart first-token margin when the bank is replaced
with that counterpart's bank. The actual detached working prefix is held fixed.
Both histories in every pair receive both source interventions. At the first
name token the original own-minus-counterpart margin is negated for the donor
loss. The full actual counterpart name, including every token, receives CE.

Raw Qwen causal prefixes are computed once before forking name alternatives.
The isolated user writer's cached banks must match on every assessment. Report
normal name decisions, donor decisions, successful switches on one fixed prefix,
normal pairs correct on both histories and pairs correct on all four bank/history
combinations separately. A favorable source difference alone is not counted as
a successful switch. Record per-parameter gradient norms before clipping.

## Prespecified conversation review

Four arms: stage7, untrained parent at gain4, control4 step16, source4 step16.
Each receives the eight historical validation positions, three new five-turn
English scenes, one three-turn Russian guard and four new English style questions.
Source4 also receives the eight fixed histories with memory disabled. Total:
128 actual replies, all with recorded seeds, token IDs, EOS and complete history.
Generation uses native BF16, temperature .8, top-p1, top-k20, presence penalty .8,
maximum 192 tokens. Fixed prefixes permit controlled comparisons; each free
conversation develops its own assistant history.

The new role-return questions explicitly name the original object, supply no
correct names and ask for both current human roles while preserving the scene.
Acceptance calls for at least 5/6 late English recalls with both human roles
recoverable and no incompatible human assignment; an improvement over parent4
and control4 with no loss against stage7; at least 3/4 historical fixed pairs
correct on both user histories; and connected English scenes preserving the
character. Read the user's transfer history separately from current roles.
Read imagery, first/third-person voice, callbacks, language and loops throughout.
Russian is a secondary guard and is reviewed separately for new collapse.

Record poetic rights, unresolved metaphors and explicit incompatible human roles
separately. Do not discard a correct current human pair because an unrelated
window or darkness also receives a figurative right. If a reply recovers current
roles but invents their acquisition history, credit and describe those two facts
in separate columns. Stage7 remains selected until the full reading supports
replacement. The sealed private input hashes and model revision travel with the
run; private data and weights stay outside git.
