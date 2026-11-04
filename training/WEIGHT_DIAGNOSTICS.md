# Stage 20: Fable attribution and CPU weight diagnostics

The stage-19 external reading is attributed to the three checkpoints in its
private blind key. All four Latin/CJK occurrences in that packet belong to the
stage-7 primary. The two explicit role-performance openings belong to the new
checkpoints. The partial two-owner success belongs to `lr3 / update 188`.

CPU inspection has verified the published Qwen loop edit and measured its
direction in six pinned checkpoints. Sixteen pairs of edited tensors are
prepared privately, including matched random controls. No model forward pass,
SFT, DPO or GPU allocation occurred in this stage. The primary remains stage 7.

## External reading and literal attribution

Input: Oleg's pasted summary of Don/Fable's stage-19 reading, 18 questions with
three shuffled answers each. The full report named in that summary was not
attached. The available summary reports an acceptable choice on 8/9 Russian
questions and 3/9 Hebrew questions, with no acceptable expressive Hebrew answer
on questions 10–12. Complete owner/holder success is reported as 0/24.
Individual preference letters are available only for the items explicitly named
in the summary; no aggregate model winner is inferred from those incomplete letters.

All 54 texts and source hashes were checked against the private key. Counts below
refer to this consumed packet, seed 42, nine RU and nine HE answers per model.

| Literal observation | Stage-7 primary | lr1 / 80 | lr3 / 188 |
| --- | ---: | ---: | ---: |
| Mixed Latin/Cyrillic word, RU | 2/9 | 0/9 | 0/9 |
| Separate Latin-script word, RU | 1/9 | 0/9 | 0/9 |
| CJK span, HE | 1/9 | 0/9 | 0/9 |
| Role-performance phrase, HE Q11 | 0/1 | 1/1 | 1/1 |
| Initial Hebrew letter shin, HE | 4/9 | 5/9 | 4/9 |

The last row is a screening counter. Ordinary Hebrew words and numbers also
start with shin. Direct reading finds a relative-clause opening on all three
Q10 answers; the pasted review's approximately 6/27 fragment count needs its
full item list for exact attribution. The original exact `שנית/שניתן` counter
is retained separately. Role-performance phrases likewise retain exact spans
for scene-aware assessment.

Q04C (`lr3 / 188`) names both owners correctly but leaves current holders
incomplete. Fable's reported female-narrator failures Q10B, Q11B and Q12C belong
to `lr3 / 188`; Q11C belongs to `lr1 / 80`. The report's two Russian feminine
verb-tail occurrences have no item IDs in the pasted summary and remain
unattributed pending the full reading.

Q15B (`lr1 / 80`) names a wrong possessor. Its actual input contains only the
system message and Q15, with no Q14 text or previous assistant answer.
`compare_sampling.py` reconstructs fixed-case messages for each request and
passes no previous `past_key_values` into `generate`. Q15's prompt also contains
the Hebrew word for aunt, whose spelling is close to the disputed surface form
for uncle/David. The wrong-owner error is retained; the origin of the entity is
unassigned. Actual long-dialogue copying is counted against supplied assistant
history: the three previously observed whole-answer reuses are reproduced by
the new counter.

## Published edit, reconstructed from raw BF16 bytes

Sources:

- [Weight-Space Programming article](https://rikiyat.github.io/blog/weight-space-programming/).
- [Published experiment and edit](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math).
- [Frozen formula and evaluation report](https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/work_notes/final_report.md).

The published operation is `W[:, S] -= 6 * outer(u, u @ W[:, S])`, for 64
selected columns in each of layers 15 and 19. Strength six is part of that
operation; strength one removes the projection, whereas strength six reverses
and amplifies its remaining component. The source's sparse BF16 diff stores
absolute replacement values tied to specific base tensor hashes.

The two original Qwen tensor hashes and two reconstructed edited hashes match
the published manifest exactly. The edit touches 131,072 positions; 130,188
positions differ after BF16 rounding. SVD of the rounded sparse difference
recovers its dominant left direction. Rebuilding the formula with that inferred
direction gives the following measured error relative to the published delta:

| Layer | Leading singular component's share of squared delta norm | Reconstruction error / delta norm |
| --- | ---: | ---: |
| 15 | 99.999454% | 0.120043% |
| 19 | 99.999138% | 0.146376% |

Both matrices have shape 1024 × 3584. Downloaded ranges include exactly those
two tensors per model; response byte ranges, lengths, BF16 dtype, shapes and
raw tensor hashes are recorded. CPU analysis uses float64 reductions and SVD.

## Geometry after abliteration and SFT

For the inferred unit direction `u`, compare `u @ W[:, S]` between original Qwen
and the corresponding target. Cosines below refer to these 64 coefficients.
Relative changes use the entire matrix's Frobenius norm.

| Target | Layer 15 change vs Qwen | Layer 19 change vs Qwen | Coefficient cosine, L15 | Coefficient cosine, L19 |
| --- | ---: | ---: | ---: | ---: |
| huihui | 3.1334% | 3.0869% | 0.999844 | 0.999737 |
| Primary, stage 7 | 3.3307% | 3.2680% | 0.999837 | 0.999733 |
| Parent, stage 12 | 3.1601% | 3.1101% | 0.999839 | 0.999739 |
| lr1 / 80 | 3.1622% | 3.1118% | 0.999839 | 0.999741 |
| lr3 / 188 | 3.1839% | 3.1295% | 0.999842 | 0.999745 |

The original-to-huihui change projected onto `u` has 8.4084% and 7.1853% of
the total change's norm in the two layers. The stage-12-to-lr3/188 change has
3.2976% and 2.8573% along that direction. These are norm fractions, not squared
energy percentages.

Target candidates recompute `u @ W_target[:, S]` using their own weights.
Strengths 0.25, 1, 3 and 6 are staged for huihui, stage 7 and lr3/188. Every
unselected column stays bit-identical. Two additional controls per Dubrovsky
checkpoint use seeded directions orthogonal to `u`, with the same column
support and matched pre-rounding perturbation norm at strengths 1 and 6.
After BF16 rounding their norm ratios range from 0.999956 to 1.000176.

These are 16 two-tensor artifacts, each marked unevaluated, with hashes and
parent provenance. Full model releases and the selected primary were not rewritten.

Pinned model sources:

| Model | Revision | Model-file location |
| --- | --- | --- |
| Qwen/Qwen3.5-0.8B | `2fc06364715b967f1860aea9cf38778875588b17` | Original single safetensors shard |
| huihui-ai/Huihui-Qwen3.5-0.8B-abliterated | `4813135658fe51b2e535b8e77927d2f23909ce42` | Original single safetensors shard |
| Private primary | `805bc6e493f284be8b20c262b8fe65070c886c27` | `working/six-language-polish-epoch001` |
| Private parent | `a88b6374dbf496f3c4c6cea93004cdebefeb4251` | `working/joint-multilingual-empty-closure-stage12` |
| Private lr3 / 188 | `04e78daa7114ffbce835552a2db52119c91e48e7` | `working/lr-control-stage19-lr3-step-000188` |

The lr1/80 revision and all per-tensor digests are in the private source inventory.

## Next measured comparison

1. Use consumed stage-19 histories to compare productive continuation with exact
   previous-answer reuse. Capture MLP activations and next-token log probability
   with the same prefix, fixed tokens and checkpoint; inspect which columns
   actually write the contrast in Dubrovsky.
2. On that diagnostic split, compare unchanged weights, the targeted strengths
   and the norm-matched orthogonal controls. Include separate layer-15-only and
   layer-19-only comparisons if the combined intervention helps.
3. Freeze any selected intervention before new acceptance prompts. Run new
   RU/HE/EN conversations with changes of topic and delayed object references,
   two generation seeds, exact-history receipts and whole-answer reuse checks.
   Check English voice, native thinking closure and retained photographic probes.
4. In a separate data experiment, add a verified RU/HE curriculum: named actors
   and speaker-relative `I/you`; one object then two; gifts, loans, safekeeping,
   onward transfers and returns. Track owner and holder separately through each
   event. Add Hebrew female-narrator chains across tense, varied grammatical
   beginnings and immediate in-character answers to role prompts. Keep a larger
   Hebrew allocation and English voice replay. Use fresh acceptance scenes.

The automatic counters preserve literal observations. Human reading distinguishes
intentional lexical invention, code vocabulary, clause fragments and broken
ordinary morphology. New training and acceptance records stay private on HF.

## Verification

- 17 text-diagnostic tests and seven weight-geometry tests pass.
- All 54 blind answers match their source hashes; 240 long-dialogue rows were
  re-counted using their own saved histories.
- Six pinned model sources supplied twelve tensor ranges; source edit hashes
  reproduce exactly; all candidate columns outside the selected support match.
- BF16 rounding ties, overflow rejection, target-specific coefficients, SVD
  sign invariance and orthogonal control norms have direct regression checks.
- Stage 20 has zero generation evaluations. No candidate has been promoted.
