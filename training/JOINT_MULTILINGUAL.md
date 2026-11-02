# Matched multilingual reconstruction — 2026-11-02

This trial asks whether starting from the original huihui with all six languages
is preferable to placing the English examples first. A third arm learns to
close the native thinking block immediately. Stage 7 remains the primary. The mixed empty-closure arm is preserved as a
research parent for the next Russian-focused continuation, not a final release.

## Historical evidence

The first English reconstruction used 436 training records for three epochs,
84 optimizer updates, peak learning rate 1e-5, and frozen embeddings/head/vision.
English reference loss at steps 0, 28, 56 and 84 was 3.16668, 2.21248, 2.21929
and 2.33834. Fourteen saved generation probes were run after **each** epoch,
including EN/RU/HE/FR/DE/ES. Epoch 3 was deliberately selected for the English
voice and more EOS completions (4/6 versus 2/6 at epoch 2), while the selection
note already acknowledged other languages switching to English. The higher
reference loss alone was not the selection criterion.

The later [lineage audit](LINEAGE_AUDIT.md) confirms that language choice recovered
in stage 7, but grammar and scene ownership remained weak. Freezing the token
interface did not prevent the language body from changing output-language
behavior. This is an observed tradeoff in the historical sequence, not evidence
that multilingual generations were never checked.

## Controlled design

All three arms start independently from
`huihui-ai/Huihui-Qwen3.5-0.8B-abliterated` at
`4813135658fe51b2e535b8e77927d2f23909ce42`. The already reviewed stage-7 mixture
contains 642 unique records: EN 228, RU 131, HE 127 and FR/DE/ES 52 each. No new
training answers are authored for this experiment. Every record appears three
times in each prepared schedule, making 1,926 presentations and 121 updates
with accumulation 16. One outer epoch consumes that complete schedule.

| Arm | Language order | Thinking supervision |
| --- | --- | --- |
| mixed | Seeded mixture; all six languages in the first update | Closed prefix masked; answer and turn end supervised |
| english-first | The same rows, stable-partitioned into 684 English then 1,242 other-language presentations | Same as mixed |
| mixed-closure | Exactly the mixed order | Native immediate empty closure, answer and turn end supervised |

The first two arms isolate ordering within this corpus. They do not reproduce
the entire historical English-first training chain, its larger English slice,
learning rates, DPO runs or intervening curricula. This is one training seed and
one exposure/LR setting, not a general proof of an optimal schedule. Language
order also determines which examples encounter each point of the shared LR curve.

All arms use full language-body SFT, not LoRA: 498,113,344 trainable parameters
out of 852,985,920 total, FP32 trainable parameters,
BF16 autocast, AdamW, peak LR 3e-6, weight decay 0.01, clipping 1, 10% warmup and
cosine decay ending at 10% of peak. Embeddings, tied output head and the visual
encoder are frozen. Context is 3,072; the longest prepared native sequence is
1,362 tokens, so no row is truncated. Answer-only projection retains full
attention history and the preceding prediction position.

`--ordered-training` consumes prepared rows exactly and records their IDs at
every optimizer update. The operator checks this consumed order against the
input schedule. `--eval-every-steps 40` adds within-epoch validation at steps
40, 80 and 120, alongside steps 0 and 121. `validation_by_group` separates new
references from the legacy development references without changing the latter's
loss calculation. Defaults retain the previous shuffled training behavior.

## Independent new development scenes

There are 36 new Astra-authored and self-reviewed literary reference answers:
six scene concepts rendered in six languages. They are correlated translations
of six scenes, not 36 independent topics. Combined with 79 legacy references,
they make 115 validation records. Legacy teacher generations had seen original
validation material; the legacy loss is not a fully blind test.

Fresh sampling uses those 36 new questions and seeds 42/137: 72 responses per
model. Two new RU/EN conversations each contain 20 turns, with topic changes,
an opening-scene callback at turn 17 and a final return at turn 20. Each branch
uses its own actual preceding answers. The 76 new user questions have zero
normalized exact overlap with the available saved training questions and
histories; this is not a semantic-overlap guarantee.

Twelve fresh questions also test both native closed and open thinking prefixes,
seed 42: 24 generations per model. The original base receives 72 fresh probes;
stage 7 and each new arm receive all 136, for 616 planned generations. Closed
thinking controls overlap with the fresh suite and are not independent examples.
All use temperature 0.8, top-p 1, top-k 20, generated-token presence penalty 2,
repetition penalty 1, min-p 0, a 192-token cap and native stop IDs 248044/248046.

Learning an immediate `</think>` transition is behavioral training. It does not
remove an architectural reasoning module or prove the model cannot produce
reasoning under other prompts. Both open-prefix closure and the quality of the
answer after it must be measured.

## Sources and interpretation

The official [Qwen3.5-0.8B card](https://huggingface.co/Qwen/Qwen3.5-0.8B)
defaults to non-thinking and warns that enabling thinking in this small model
can increase repetition. [Unsloth's implementation guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
supports full fine-tuning of the architecture. General multilingual instruction
research, including [Shaham et al.](https://research.google/pubs/multilingual-instruction-tuning-with-just-a-pinch-of-multilinguality/),
motivates testing mixed-language examples early; it does not establish results
for this 0.8B character model. The measured trial, not that analogy, determines
selection here. A lower loss on one literary reference measures greater likelihood
of that specific response; it is not a correctness score for an open-ended scene.

Private schedules, reference texts, exact generations, hashes and recovery
operators are stored in the private `ataeff/Dubrovsky` dataset under
`additions/2026-11-02-joint-multilingual/`. Raw authored training data stays out of git.

## Follow-up to the ordinary mixed arm

The ordinary mixed arm closed all 12 native open thinking prefixes even though
its supervised targets excluded the closure. Four of those continuations had
no final answer. This prompted 24 additional open/closed-prefix generations on
the original base, after the planned comparison, to distinguish inherited
behavior from a new SFT effect. The planned 616 generations and follow-up 24
are recorded separately; their total is 640. A separate read-only likelihood
check uses the actual BF16 exports, loaded into FP32 with BF16 autocast, for
stage 7 and all three new arms. Training-time validation uses live pre-export
FP32 weights and can differ slightly from that export comparison.

## Completed measurements

All three runs completed 121 updates and exactly matched their prepared consumed
row order: 363 optimizer updates and 5,778 presentations in total. All frozen
embedding/head hashes matched before and after each run. The 153 visual tensors
matched the original snapshot after canonical BF16 conversion. This checks
weight integrity, not image-response behavior; no new image probe was run.
All 32 unit tests passed on the H100. Locally, 29 passed and three PyTorch-only
export tests were explicitly skipped because PyTorch is absent.

The live overall reference losses fell from 3.35536 to 2.60440 (mixed), 2.63404
(English-first) and 2.61910 (mixed-closure). All three had a lower overall loss
at step 80 than at step 121, so simply repeating the same examples longer did
not keep improving this diagnostic. The new references on the actual exports
show the following answer-only losses; each language has only six new references:

| Language | Stage 7 | Mixed | English-first | Mixed + empty closure |
| --- | ---: | ---: | ---: | ---: |
| EN | 4.975 | 3.376 | 3.299 | 3.358 |
| RU | 3.914 | 3.178 | 3.179 | 3.194 |
| HE | 2.929 | 2.545 | 2.540 | 2.555 |
| FR | 3.329 | 2.867 | 2.857 | 2.895 |
| DE | 3.475 | 3.111 | 3.089 | 3.129 |
| ES | 3.502 | 3.115 | 3.064 | 3.111 |

The English-first arm is slightly better on five of the six fresh reference
sets, while mixed has the lower overall legacy-dominated loss. At step 40,
before English-first reaches its non-English rows, mixed is better on all five
non-English fresh reference sets. By step 80 English-first catches up. This does
not establish a universally better curriculum or show that the historical
English-first choice was itself an implementation error.

All 72 fresh plus 40 closed-prefix dialogue responses from each trained snapshot
were nonempty and reached EOS. Original huihui reached EOS on 19/72 fresh scenes;
its other outputs met the same 192-token cap. A capped output is not by itself
a detected loop. Grammar and scene ownership remain impaired despite the loss
reduction: mixed contains Russian “Дубровский представило” and “зрительную
атаки”; mixed-closure includes “обслуживание часы” and “другой шляпу”. Hebrew
and German remain substantially malformed. These are agreement/inflection
examples, not a penalty for authored neologisms.

In the mixed English conversation, turn 17 restores Nell, brass and the
weather-lie trigger. English-first instead brings in a copper tube. The
mixed-closure closed-prefix conversation changes the trigger, while its
open-prefix follow-up retrieves all three facts. Russian callbacks remain
unreliable: mixed omits the compass color, and both English-first and
mixed-closure identify Zoya as the compass. A successful English conversation
therefore cannot establish reliable memory across languages and modes.

## Learned closure, measured separately from language quality

| Snapshot | Immediate empty closure, native open prefix | Nonempty final answer after closure | EOS within 192 tokens |
| --- | ---: | ---: | ---: |
| Original huihui | 0/12 | 0/12 | 0/12 |
| Stage 7 | 0/12 | 0/12 | 12/12 |
| Mixed | 12/12 | 8/12 | 12/12 |
| English-first | 4/12 | 4/12 | 12/12 |
| Mixed + empty closure | 12/12 | 12/12 | 12/12 |

The base did not close within this token budget; these probes do not establish
that every base output was looping. Stage 7 ended its text inside the unclosed
block. The ordinary mixed arm's closures were not present in the base's matched
probes, so they must not be described simply as observed inherited closure.
Only the matched mixed-closure arm consistently supplied both immediate closure
and a nonempty answer on this sample.

A follow-up ran both complete 20-turn conversations with the mixed-closure arm,
now using the native open prefix at every turn and retaining only each actual
final answer in history. All 40 turns closed with whitespace only before
`</think>`, generated nonempty answers, contained no extra thinking tags and
reached EOS. Together with the 12 single-turn probes, that is **52/52 tested
open-prefix continuations**. It is evidence of learned weight behavior; it is
not proof that every possible prompt can no longer elicit reasoning. Closed
prefixes also remain valid. Total student generations including both follow-ups:
**680**, with overlapping controls explicitly retained as controls.

The mixed-closure candidate is backed up privately at
`ataeff/Dubrovsky/working/joint-multilingual-empty-closure-stage12`, revision
`a88b6374dbf496f3c4c6cea93004cdebefeb4251`. Its 1,706,030,528-byte weight file
has SHA-256 `74a27581a4cba0eff0170d5b373340c59791a37e8fd8e8d994687871b78758e1`.
It is a temporary research parent for Russian continuation. Stage 7 remains
available as the selected primary and comparison reference; there is no new
final or quantized release. The other two measured exports remain on the stopped
training volume, with their hashes in the private evidence.

## Russian-first follow-up material

Equal language volume is not required. The next continuation prioritizes Russian
while replaying English character and a smaller multilingual retention slice.
The current corpus has approximately 10,028 unique Russian supervised tokens,
11,673 Hebrew, 20,489 English, and 4,600–4,931 per FR/DE/ES. Repeating those
examples three times does not provide three times as many independent scenes.
The intended next improvement is additional reviewed Russian material, not a
larger number of identical passes alone.

A direct authenticated OpenAI API call now succeeds. One `gpt-6-astra` literary
pilot received the recovered Russian source excerpts and 359 EN/RU records from
the existing training mixture, including their histories. It received no stage-12
held-out prompts or references. The full request, response, model identifier,
response ID, hashes and usage are retained privately. Usage: 165,567 input tokens,
5,505 output tokens (including 33 reasoning tokens), status completed. Historical
contexts repeat across input rows; future large batches should deduplicate that
presentation while retaining all unique authored material.

The completed response contains 12 standalone scenes and two full 20-turn Russian
conversations. Astra reviewed all 52 target answers and recorded three small
continuity/pronoun edits, preserving the untouched raw response. Both callbacks
retain their three opening facts. Native open/closed-target tokenization passes;
the longest full-history record is 1,819 tokens. New questions have zero normalized
exact overlap with the 76 stage-12 user questions. These are new GPT-authored
works with an explicit editorial record, not recovered original quotations.
They are prepared for the next student trial and **have not yet been trained**.
