# Training and evaluation audit — 2026-11-02

The response-repair trial keeps stage 7 as the selected primary. Two further
epochs reduce the old reference loss, but fresh language and conversation
outputs do not justify promotion. This audit distinguishes implementation
defects, sampling limits and hypotheses that were checked and corrected.

## DPO conversation context

The previous `dpo.py` encoded only the current question and the two answers.
A supplied `history` field was silently omitted. This would train a contextual
preference without the facts needed to understand that preference. Both
alternatives now use the same complete history through the SFT encoder. A
configurable maximum length rejects oversized conversations without truncation.
Run metadata records the number of contextual pairs and maximum encoded length.

The four saved preference datasets inspected contain 24, 17, 20 and 2 pairs;
all have zero history-bearing pairs. The defect therefore does not change the
encoding of those earlier datasets. Forty newly prepared repair preferences
include 19 history-bearing pairs. Their native tokenization passed identical
prefix and masking checks, with a maximum length of 1,568 tokens. No new DPO
optimizer run was performed in this trial.

## Language loss and conversation selection

The carried-forward validation set has only four Russian and four Hebrew
reference answers. Its loss is a narrow diagnostic. In stage 10 all six losses
improve after one epoch, while fresh Russian still contains agreement errors
and malformed words. After epoch 2 English loss worsens relative to epoch 1.
Neither loss trajectory determines the model selection.

The earlier successful English 20-turn conversation was one scene sequence.
On the new sequence, stage 7 changes the opening character's name at the
requested callback. Epoch 1 recovers the name but imports an unrelated later
character; epoch 2 confuses two spellings and scene ownership. The new Russian
conversation also remains impaired. Long-conversation retention must be tested
across new scene sequences rather than inferred from one favorable transcript.

## Training exposure and evaluation retirement

The 40 new repairs use prior measured outputs and their questions. Those case
identifiers are explicitly retired from held-out use. The earlier English
conversation is also retired: eight reviewed replies are now training replay.
Original raw histories and corrected-history variants have distinct provenance.
Substantive literary rewrites are labelled as such rather than described as
mechanical grammar fixes or verbatim model output.

Selection uses 20 new answer-free questions with two seeds and two new 20-turn
RU/EN conversations. Training questions, including prior user turns, have no
normalized exact overlap with these probes or the retained validation prompts.
Earlier teacher generations saw the original full literary corpus, including
validation material; their development loss must not be described as a fully
blind test. New answer-free probes supply additional evidence.

The stage-10 mixture has 236 EN, 123 RU and ten examples for each of FR/DE/ES/HE.
The latter consist of eight replayed dialogue turns and two repairs per
language. This is a small retention slice, not completed language polishing.
Two fresh questions per each of those languages also remain a small sample.

## Raw output and the corrected diagnostic hypothesis

Probe records now preserve generated text both with and without special-token
stripping. Marker checks use the raw form, while word-copy statistics continue
to use readable text. The initial suspicion was that native thinking markers
were being stripped before inspection. Inspection of the pinned tokenizer
showed that `<think>` and `</think>` are not classified as special tokens there.
Rechecking 93 old control records found zero previously missed markers. The
code change makes evidence independent of that classification; it does not
retroactively invalidate the earlier marker observations.

The current GPU comparison used the unchanged pre-audit probe code throughout.
Its saved token IDs were decoded and checked separately after completion; all
240 records reached EOS, had nonempty readable text and contained no thinking
markers. At epoch 1, a Russian turn nearly repeats an earlier answer: 90.9% of
its word 4-grams occur in prior answers. It is not an exact full-string repeat.
None of the 240 outputs overlaps the 399 current training answers by a word
8-gram. That copy check covers this trial's targets, not every historical corpus.

## What remains to resolve

Frozen embedding/head hashes and all 153 vision tensors remain identical after
both epochs. The updated language body still determines language and image
responses; tensor integrity alone does not establish behavioral retention.
No new image-generation probe was run for these rejected candidates.

Stage 10 changes the data mixture and uses the masked, closed thinking prefix.
It is a candidate-selection experiment, not an isolated causal comparison with
stage 9. Before another general continuation, compare the original huihui base
and earlier selected stages on the same new language probes, expand the
independent language reference set, and separate ordinary grammar failures
from authored wordplay. The prepared contextual preferences remain available
for a separately measured DPO trial. The selected model and decoding profile
remain unchanged.

## Follow-up: matched lineage and serialization audit

The [three-stage comparison](LINEAGE_AUDIT.md) now measures the base, early
English-only SFT and stage 7 on identical fresh scenes, conversations and new
word-choice prompts. It observes language switching after English SFT and
recovery of predominant response language in stage 7, alongside remaining
agreement and callback errors. It also identifies and fixes duplicated tied
weights in SFT exports, with an actual Qwen tensor round trip.
