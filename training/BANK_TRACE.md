# Stage35 — trace the user bank through facts and questions

Stage34 retained stage7. Its explicit counterpart target reached 0/16 training
source switches and 0/8 validation switches; control reached 1/16 and 0/8.
Late English current-role pairs were 1/6 in both, with scene and transfer-history
defects recorded by complete contextual reading. This frozen diagnostic follows
the fact through the writer and the readout before choosing another change.

## Sealed measurements

Use the same eight training and four historical validation pairs, reported
separately. Load parent4, control4-step16 and source4-step16, verify identical
writer weights and freeze all parameters. Record the bank after each of the two
fact messages and after the final question, with every write/retain gate.
Verify each snapshot bitwise against the native checkpoint recurrence.

Measure paired-source distances at the three message boundaries. Also propagate
both pre-question banks through identical question features from each actual
history in turn. This separates persistence through a common final feature
stream from the two histories' contextualized user-only question features.

At the same post-question cut and detached working/Qwen prefix, measure own and
counterpart complete-name probabilities, first-token margins and add32 margins
under current own/donor banks, pre-question own/donor banks and a final zero bank.
Credit a source switch only when both own and donor choices are correct on that
same actual prefix. Report pairs correct on all four bank/history choices.

For the fixed pairs the second fact message completes both roles. Earlier
snapshots are trajectory observations, not complete-role answers. Final-read
replacement is separate from disabling memory throughout a trajectory.

## Actual replies and character

Use source4-step16 for 60 historical-prefix replies. The eight validation rows
each receive normal, own pre-question bank, donor pre-question bank and final
zero-bank readings. The six late English and one final Russian return from
stage34 each receive normal, own pre-question bank, own role-complete fact bank
and final zero-bank readings. All actual assistant histories remain unchanged.
The initial user message completes both roles in these seven retired scenes.

All 15 normal replies must replay the archived token IDs exactly. Sampling
remains native BF16, temperature .8, top-p1, top-k20, generated presence penalty
.8, maximum192 tokens. Record complete histories, seeds, token IDs and EOS.
Read all 60 replies for recovered human roles, preserved user transfers, image
development, callbacks, first/third-person voice, language and complete endings.
Keep poetic rights and actual incompatible human assignments separate. Donor
bank replies are interventions with conflicting external source facts and must
be read against both the actual history and the explicitly supplied donor.

These retired scenes diagnose the mechanism. A trained successor still needs
fresh acceptance scenes, connected English dialogue and the character intact.
The project horizon includes 12–20-turn conversations that follow and develop
Dubrovsky's own imagery. English remains primary; Russian is a separate guard.

Stage7 stays selected. This pass creates no optimizer, changes no trained weights
and makes no runtime-policy change. Archive exact sources and results, verify
all frozen tensors, collect and stop/delete the owned GPU after completion.

## Completed diagnostic — 2026-11-25

The frozen run completed on Secure Cloud A100 SXM4 80GB using source commit
`80452c250163e78660f45608a19e8406f6ff91f1`. All 60 replies were read with their
actual histories. All 15 normal replies reproduce stage34 token IDs exactly.
Normal native margins also reproduce stage34 exactly in all six state/split
groups. The owned GPU was collected, verified, stopped and deleted.

### Writer trajectory

The following ratios compare paired-bank RMS distance after the final question
with distance after the second, role-completing fact message. The common-feature
control propagates both earlier banks through one identical question-feature
stream, using each actual history's stream in turn.

| Group | Pairs | Median ratio, each history's features | Median ratio, common question features |
|---|---:|---:|---:|
| Train owner | 4 | .719574 | .052170 |
| Train keeper | 4 | .600263 | .040489 |
| Historical validation owner | 2 | .804012 | .023606 |
| Historical validation keeper | 2 | .492042 | .023735 |

Old bank differences contract strongly under a common question-feature stream.
In the ordinary path, Qwen's user-only contextualized question features differ
between histories and replenish the distinction. These ratios measure tensor
distance, not a percentage of successfully remembered facts. The final question
has mean retain .889855 and mean write .384539 across recorded tokens/experts;
it actively writes to the bank.

### Readout at one fixed working prefix

| State | Train own correct, current → pre-question | Train source switches, current → pre-question | Validation own correct | Validation switches |
|---|---:|---:|---:|---:|
| parent4 | 11/16 → 11/16 | 0/16 → 0/16 | 5/8 → 5/8 | 0/8 → 0/8 |
| control4 | 11/16 → 11/16 | 1/16 → 0/16 | 5/8 → 5/8 | 0/8 → 0/8 |
| source4 | 10/16 → 11/16 | 0/16 → 1/16 | 5/8 → 5/8 | 0/8 → 0/8 |

All four bank/history decisions remain 0/8 training pairs and 0/4 validation
pairs in every state. Source4 recovers its lost training mask-owner position:
normal native margin -.25 becomes +.125, while the earlier donor bank gives
-.1875. Parent4 and control4 already chose that normal name correctly.
Earlier bank separation still has weak control over source-name decisions.

### Contextual reading

All 60 replies were read; modes were visible. Human roles, transfer fidelity and
image development are recorded separately. Exact histories, answers, tokens and
individual readings are archived privately.

| Replies | Current pair recovered | Pair and unambiguous transfer history |
|---|---:|---:|
| Fixed, normal | 4/8 | 2/8 |
| Fixed, own pre-question bank | 4/8 | 2/8 |
| Fixed, final zero bank | 4/8 | 1/8 |
| Late English, normal | 1/6 | 0/6 |
| Late English, own pre-question bank | 2/6 | 0/6 |
| Late English, own role-complete initial bank | 1/6 | 0/6 |
| Late English, final zero bank | 0/6 | 0/6 |

The newly credited English pair appears in the final feather return: its
original owner and keeper are explicitly restored, together with the return
direction. It receives generous credit because an adjacent owner's "keeps"
and further museum/location clauses leave physical custody ambiguous. The
transfer-history column remains uncredited. The ending develops the silence's
lunch break into an eaten interval and a sensory image of unused expectation.
This image is retained in the review rather than penalized for literal
impossibility. In another branch the early bank still produces an invented
owner, so the recovery is not consistent across the scene.

Nominal ownership, object self-ownership, a birthday purpose and ownership of
unrelated figurative objects are read in context. Explicit human transfers and
incompatible custody are recorded separately. The fixed donor bank recovers
its own pair in 1/8 replies, but that text already appeared as an error in the
normal run; it is not a new source switch. The one Russian final return is 0/1
in all modes, with agreement and language-mixing defects. All 60 replies reach
EOS. Full literary notes remain beside each actual answer.

### Decision and next pilot

Stage7 remains selected. This pass changed neither weights nor runtime policy.
The role-complete snapshot was manually specified in the diagnostic packet.
The next matched pilot should test factual retention through messages without
new events jointly with source-specific readout. It must also accept actual
new gifts and returns, preserving updateability. Compare a frozen writer with
an optimized writer under the same starting state, examples, update budget and
authored replay. Use paired source interventions on identical working prefixes,
then fresh English acceptance with complete contextual reading and further
12–20-turn character conversations.

153 tests passed locally and on GPU. Verification covers 25 file checksums,
24 native-parity writer trajectories, 72 readout rows, 15 identical prefix groups,
15 exact normal token replays, all 474 body tensors and frozen expert states.
Optimizer steps: zero. The latest checked Omni HF revision was
`eaf1a637da22b7f9ba158402e18ffd8004c30a63`; its reviewed output-gain report remains
the latest completed result found during this pass.

### Archive

Private `ataeff/Dubrovsky`, prefix `research/stage35-bank-trace`, verified commit
`e70f49d604247635d0afdb00de6aaf5dda97f9e7` contains the full contextual reading,
52-entry results archive and manifest. All three uploaded files passed size and
SHA256 checks. The three input expert deltas are inside the archive; there are
no newly trained weights or additional full-body exports.

- Reading SHA256: `4a73867c2abb7993d8a581963175fc10f83fda8c6afac1985bde308b9b149571`.
- Archive SHA256: `0677418198efa2c05c0192187b1654e8592db672a0d8453bce8ce32c56bc6f2b`.
