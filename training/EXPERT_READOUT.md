# Frozen-prefix expert readout: stage 27

Stage 26 completed 228 updates with the same data and initialization as stage
25. Its first epoch replayed exactly; mean validation NLL fell from 3.537487
at step 57 to 3.429984. The subsequent reading of all 384 replies keeps stage
7 selected. On the new scenes, complete owner/keeper assignments were EN 2/12
at step 57 and 1/12 at step 228; RU 0/12 at both. Two ambiguous late EN replies
are retained separately. The same-prefix owner contrast score stayed 6/8 by
mean token probability in all modes. English is the primary working language;
Russian is a small retention control after the prior language SFT work.

Stage 27 performs no training. It freezes twelve actual English prefixes from
the stage-26 step-228 expert conversations: three scenes, two seeds, two recall
positions. Stage 7, step 57, step 228, and step 228 with its bank disabled each
answer all twelve identical prefixes. The late expert arm must reproduce the
archived token IDs exactly before the other comparisons are accepted.

Six sealed cases also compare owner-source and keeper-source replacements at
two boundaries: immediately before the final user question, and immediately
after its last content token. Only the user bank is replaced. Qwen's complete
prefix and the working state accumulated before the intervention stay fixed.
The original user-only features continue normally afterwards. Donors are
constructed from explicit alternative user events, with consistent referring
phrases; their complete private source messages and hashes are retained.
Unchanged isolated sources must recreate the native bank bit for bit.

These six cases additionally test holding writes during the question and one
explicit role-question paraphrase. The production writer remains unchanged.
Four English image-return prefixes across all four arms preserve the earlier
postman/registry threads. Four Russian prefixes across base/late-expert arms
check feminine voice and the return ending. Total: 108 complete replies.

Owner and keeper are scored separately at their actual value tokens under
the native template, using short teacher prefixes `Owner:` and `Keeper:`.
The cue and EOS are excluded. Record both total and mean value likelihood,
token identities and prefix hashes, routing, bank/working state, clipping,
working-state saturation, and the value-logit contribution before and after
BF16 rounding. These 84 paired score records remain separate from reading
the full answers and their scene development.

All 108 replies receive contextual reading. Correct facts, contradictions,
ambiguous referents, image development, callback, ending, and character are
recorded separately. Stage-26 scenes remain diagnostic material; a later
trained candidate requires fresh acceptance conversations.

The diagnostic preserves checkpoint tensors and does not select a release.
Its result chooses the next component to train: write/retention, question
readout/output, or continuation. Existing weights and stage-26 source material
remain in private Hugging Face. New diagnostic sources and answers stay private;
credentials are excluded from every result archive and from git.

Implementation: `probe_expert_readout.py`; the production sampler and expert
block are reused. `test_expert_readout.py` checks exact no-op replay, boundary
isolation, held writes, cached continuation and input/parameter preservation.
The dedicated worker validates hashes and offers authenticated result collection.
It stops its own GPU after collection or a bounded timeout. The operator verifies
the result bytes before deleting it. HF publishing is unavailable in the current
operator environment; new private results use a separate retained artifact.

## Completed run: 2026-11-17

All 120 tests passed on CPU and on the diagnostic A40. The GPU environment
matched stage 26: PyTorch 2.8.0.dev20250319+cu128, CUDA 12.8, Transformers 5.22.1.
All 20 archived generations replayed token for token. All 108 new replies ended
at EOS and were read in context; 84 value-score records were collected. All 474
body tensors and both expert checkpoints remained unchanged. Optimizer steps:
zero. The owned pod was stopped and terminated after the result hashes were
verified. Private transcripts, per-reply reading notes and numeric records are
retained separately from git.

These are next-answer comparisons on twelve identical English prefixes drawn
from the late-expert stage-26 trajectories. They isolate the mode's effect at
those prefixes; fresh acceptance conversations remain the next candidate's gate.

| Mode | Complete, consistent owner/keeper answers | Owner value preferred, mean / sum | Keeper value preferred, mean / sum |
| --- | ---: | ---: | ---: |
| Stage 7, experts disabled | 2/12 | 9/12 / 9/12 | 3/12 / 3/12 |
| Step 57 | 2/12 | 9/12 / 9/12 | 3/12 / 3/12 |
| Step 228 | 1/12 | 9/12 / 8/12 | 3/12 / 3/12 |
| Step 228, bank disabled | 2/12 | 9/12 / 8/12 | 3/12 / 3/12 |

Value preference excludes the role cue and EOS. The length-sensitive mean and
sum scores are reported separately from reading the whole answer. Both value
contrasts were positive in 3/12 cases by mean and 2/12 by sum, in every mode.
Answers that name the right pair and then introduce another owner are not clean
passes. The four Russian final-answer controls supplied no complete correct pair;
the four feminine-voice controls retained feminine forms but contained other
language defects.

The bank-off arm changed 11/16 English replies relative to late experts, including
9/12 recall replies. This contrast includes the bank's influence throughout the
working-state history. Local bank replacements keep that earlier working state
fixed:

| Intervention | Full replies changed | Scored value-token logits changed |
| --- | ---: | ---: |
| Alternate owner bank before question | 0/6 | 0/36 |
| Alternate keeper bank before question | 0/6 | 0/36 |
| Alternate owner bank after question | 2/6 | 1/36 |
| Alternate keeper bank after question | 3/6 | 2/36 |
| Hold writes during question | 3/6 | 14/36 |

The 36 positions per row include both roles and both candidate values. Before
the question, replacements differed from the original bank by 0.148–0.780 in
maximum absolute coordinate difference. During the question this difference
contracted by 97.28–99.95%. The writer was still receiving isolated Qwen features
of the original complete user history, including the question. This directly
describes convergence under the original source; it is the relevant condition
when interpreting the pre-question replacement.

After-question replacement changed every measured FP32 projected contribution
to a value logit, and every contribution measured after hidden-state rounding.
At the actual BF16 head output, only 1/36 owner-source and 2/36 keeper-source
value logits changed. Log probabilities can still change when other vocabulary
logits change. The relevant mean role-value margin moved toward the donor value
in 4/6 owner replacements and 5/6 keeper replacements after the question.
Neither replacement changed the sign of a mean role-value margin.
The altered free replies did not explicitly adopt the replaced role. For the
late baseline, mean working-state saturation at these value positions was 16.43%;
6/72 proposals hit the output bound. Record these separately from rounding.

Holding writes preserved one clean pair in the six probes, the same case as the
ordinary late arm. Explicit role-question paraphrases changed all six replies
but produced no fully clean answer: one supplied the current pair while inventing
an incorrect former owner. The current evidence therefore keeps normal writes
and stage 7 as the selected model.

The contextual reading found concrete callbacks and image development across
the base and expert arms, including returns to the model's own earlier images.
Several strong endings coexist with wrong roles, and several accurate short
role statements later contradict themselves. Read both the factual anchor and
the continuation before accepting a candidate.

## Next narrow experiment

Target the connection between bank content and role-valued output. Start with
the Qwen body and the bank writer frozen, and a short pilot of the memory reader,
memory routing and expert output. Use paired ownership and custody changes with
supervision on the actual name/value positions, plus authored-voice replay.
Measure matched-prefix bank replacement direction alongside the free replies.
Compare the existing BF16 output with a diagnostic FP32 projection to locate
the numeric loss before choosing a production precision change.

Preserve early points and select by fresh English conversations: both current
roles, internally consistent answers, explicit returns after topic changes,
development of the character's own images and a complete ending. Keep a small
Russian retention control. This is the proposed next experiment; no such
training or precision change was performed in stage 27.

Concept and direction: Oleg Ataeff. Implementation and reading: Astra.
