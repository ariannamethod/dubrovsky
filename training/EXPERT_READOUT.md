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

Execution status (2026-11-17): prepared, GPU diagnostic pending. All 120 tests
passed locally on CPU (PyTorch 2.8.0+cpu, Transformers 5.22.1), including seven
new intervention checks. The 20 frozen prefixes and their reference token IDs
match the archived stage-26 records exactly. No new pod was created and no
diagnostic generations are reported; exact GPU replay remains a run-time gate.

Concept and direction: Oleg Ataeff. Implementation and reading: Astra.
