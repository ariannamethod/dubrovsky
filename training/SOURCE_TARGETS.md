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

## Completed pilot — 2026-11-25

Both matched 16-update runs completed on a Secure Cloud A100 SXM4 80GB using
source commit `9cae22595fe8987795dd0dcfc50c7785512b7b3d`. All 34 checkpoints,
optimizer/RNG states, sealed inputs, metrics and 128 replies were collected.
149 tests passed locally and on the GPU. The owned pod was stopped and deleted.

| Fixed-prefix measurement | parent4 | control4 step16 | source4 step16 |
|---|---:|---:|---:|
| Train NLL | 1.770354 | 1.533372 | 1.566932 |
| Train normal name correct | 11/16 | 11/16 | 10/16 |
| Train donor name correct | 5/16 | 6/16 | 6/16 |
| Train correct source switches | 0/16 | 1/16 | 0/16 |
| Train both normal histories correct | 3/8 | 3/8 | 3/8 |
| Train all four bank/history decisions correct | 0/8 | 0/8 | 0/8 |
| Train favorable native source contrast | 11/16 | 13/16 | 12/16 |
| Train favorable add32 source contrast | 14/16 | 16/16 | 16/16 |
| Validation NLL | 3.784118 | 3.973951 | 3.935774 |
| Validation normal name correct | 5/8 | 5/8 | 5/8 |
| Validation donor name correct | 3/8 | 3/8 | 3/8 |
| Validation correct source switches | 0/8 | 0/8 | 0/8 |
| Validation both normal histories correct | 1/4 | 1/4 | 1/4 |
| Validation all four bank/history decisions correct | 0/4 | 0/4 | 0/4 |
| Validation favorable native source contrast | 2/8 | 1/8 | 0/8 |
| Validation favorable add32 source contrast | 3/8 | 3/8 | 3/8 |

The control's single training source switch has normal/donor native margins
0.1875/-0.125. Source4 changes those to -0.25/-0.4375: the donor becomes correct
while the formerly correct own-bank decision is lost. Neither endpoint repairs
another normal training choice. The lowest trained validation NLL occurs at
step2 in both arms: 3.751760 and 3.756947. Those points are retained; the
prespecified conversation review concerns step16, not step2.

All 128 replies were read with their actual histories: 106 individual notes and
22 reused notes for verified exact context/text/token/expected-pair duplicates.
Modes were visible during reading. Current roles and faithful transfer history
are recorded independently, including generous readings of poetic rights.

| Contextual reading | stage7 | parent4 | control4 | source4 |
|---|---:|---:|---:|---:|
| Fixed replies, current pair recovered | 3/8 | 4/8 | 4/8 | 4/8 |
| Fixed replies, pair and transfer history preserved | 1/8 | 2/8 | 2/8 | 2/8 |
| Fixed pairs correct on both histories | 0/4 | 1/4 | 1/4 | 1/4 |
| Late English recalls, current pair recovered | 0/6 | 1/6 | 1/6 | 1/6 |
| Late English recalls, pair and transfer history preserved | 0/6 | 1/6 | 1/6 | 0/6 |
| Final Russian recall, current pair recovered | 0/1 | 0/1 | 0/1 | 0/1 |

Source4's one credited late pair is ambiguous: the original owner retains a
figurative right, the keeper is explicit, and an intermediate borrower is also
described as physically owning the object. The current pair receives generous
credit; the transfer narrative does not. The control also receives credit where
conceptual ownership and ritual acceptance can preserve the original gift.
Later plain incompatible human assignments are scored separately. Personified
rights and unrelated figurative owners are not automatically counted as errors.

The orchid/PR/rain scene remains connected and identical across the three
expert arms. The vanished library margin and the tailor shortening tomorrow
also retain developed absurdist scenes. Explicit invented sales, abandonment
and changed human owners occur in the role dialogues. All 128 replies reach
EOS. Russian still has agreement errors and weak role recovery; it remains a
secondary guard. The complete private reading records examples and judgments.

Disabling source4's memory changes seven of eight fixed replies but leaves
current-pair recovery at 4/8, with faithful transfers in 1/8. Fixed prefixes
support controlled intervention comparisons. Free conversations develop their
own assistant histories and measure whole trajectories; their differences are
not treated as isolated effects of the final memory read.

Checks confirm identical initialization, initial validation and pair order;
gain4 at all saved points; only the three permitted tensors changed; all 474
body tensors preserved; and bitwise user-bank invariance to earlier assistant
text/length changes in English and Russian. All eight fixed-prefix groups match
across arms and memory-off; 114 collected file checksums were verified.

**Decision: keep stage7 selected.** Neither reviewed endpoint meets the sealed
5/6 late-English and 3/4 fixed-pair criteria. Source4 does not beat the matched
control. All intermediate points and both early minima remain research assets.

## Next measurement

Trace the frozen writer after each user fact, after the last role-completing
message and after the final question. On one identical final Qwen/working
prefix, compare current own/donor banks, the earlier role-complete own/donor
banks and a zero bank. Record source separation and write/retain gates, then
native name decisions at readout. Keep whole-trajectory memory disabling
separate from final-read bank replacement. A bank predating the keeper fact
must not be treated as a complete role pair. Report training and historical
validation separately.

The freshly read Omni output-gain report motivates this check: its role replies
remain unchanged, with much smaller output changes for swapped sources than
for a zero bank. Stage34's protocol was unchanged after reading that report.
The planned trace will locate where Dubrovsky loses the source distinction;
no further training was started in this pass.

## Archive

Private Hugging Face repository `ataeff/Dubrovsky`, prefix
`research/stage34-source-targets`, verified commit
`3a24b9ff13c23336ab40ef20c41430a42dd6582b` contains the full contextual reading,
141-entry results archive, manifest, initial gain4 delta, both step16 deltas and
both step2 minima. All 13 uploaded files passed size and SHA256 verification.
The archive retains all 34 checkpoints with optimizer/RNG states and the sealed
inputs; no additional full-body copy was uploaded.

- Reading SHA256: `44cde298393eab5bd42af16fd473fbaae569a860ff2d29f72a1525f9a3adb935`.
- Archive SHA256: `546ae1fefcfa0881561329683e2a5ae7fec9bbe270ddbdd180ddd51ae54c886d`.
