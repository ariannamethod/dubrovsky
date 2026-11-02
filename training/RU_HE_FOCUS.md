# Russian and Hebrew literary focus

This continuation prioritizes Russian and Hebrew while retaining the complete
228-row English slice from the selected stage-7 corpus. The preceding
[Russian continuation](RUSSIAN_CONTINUATION.md) improved some scene callbacks,
but nine presentations of each new Russian answer did not produce sustained
language improvement. This stage adds distinct authored material and recovery
from actual student histories.

## New material and editorial review

Three completed GPT-6 Astra API requests generated 240 new responses:

| Material | Russian | Hebrew |
| --- | ---: | ---: |
| Standalone character replies | 24 | 24 |
| Two 20-turn and two 16-turn conversations | 72 | 72 |
| Additional Russian registers | 48 | — |
| Total | 144 | 96 |

Each conversation changes scenes, revisits earlier facts at different turns,
accepts a local user correction and finishes with a concrete action. New people
enter new scenes; callbacks and explicit invitations determine when earlier
characters return. First- and third-person speech remain part of the voice.

The additional Russian set covers conversational speech, oral storytelling,
sudden ceremonial diction, short dry replies, interrupted comic dialogue,
negotiations between objects, productive coinages/diminutives and broken lyrical
rhythm. For example, spokes receive tutoring in mathematics after someone says
they need tightening; dough takes an instruction to rise literally. Wordplay
produces actions and consequences in the scene.

The teacher received the recovered Russian origins and 538 distinct EN/RU/HE
training pairs, presented without repeated history copies. Development prompts
were withheld. Two initial synchronous requests returned HTTP 503 without an
output or response ID. Background requests completed with saved IDs and usage.
Their raw requests, raw responses and reviewed versions remain private.

Astra read all 240 responses. Six field edits in Hebrew correct scissors
agreement, repair a malformed question, fix agreement with a red indicator lamp,
and replace two editorial-sounding endings with scene actions. The latter two
are recorded as literary rewrites. Intentional word formation and absurdist
transformations are preserved. Editorial notes are metadata; the trainer
supervises character answers and the native empty-thinking transition.

## Parent and initial mixture

The training parent is `working/joint-multilingual-empty-closure-stage12` in
the private `ataeff/Dubrovsky` model repository, revision
`a88b6374dbf496f3c4c6cea93004cdebefeb4251`, weight SHA-256
`74a27581a4cba0eff0170d5b373340c59791a37e8fd8e8d994687871b78758e1`.
The selected stage-7 model is a separate comparison reference. This starts from
the same parent as stage 13, before its Russian-only continuation updates.

Before student-history repairs, the prepared mixture has 696 distinct rows:
EN 228, RU 260, HE 160, and FR/DE/ES 16 each. Russian includes the 52 previously
reviewed teacher responses once, alongside 64 older Russian rows and 144 new
ones. Hebrew includes 64 older rows and 96 new ones. Replay selection is seeded
with `20261104`.

Native tokenization checks all prepared examples without truncation. The final
mixture after repairs has 728 distinct full-history examples: EN 228, RU 276,
HE 176, and FR/DE/ES 16 each. Its 63,031 training target tokens include the
native empty closure: EN 21,173, RU 21,081, HE 16,190, FR 1,618, DE 1,664 and
ES 1,305. Answer/turn-end tokens alone total 60,847. Russian and Hebrew account
for 59.1% of training target tokens, English 33.6%. The longest sequence has
2,356 tokens. Row shares and token shares are different; accumulation weights
target tokens within each update.

The 32 repairs use the parent's actual histories from two separate 16-turn
training scenes. GPT received those histories and the corresponding clean
teacher scenes. Astra read every raw answer and repair, then removed an
unsupported fence-owner assertion and a ribbon location invented from the
teacher's history. Original drafts and editorial changes remain available.
The repaired response is supervised; earlier student mistakes remain masked
context. These rollouts are training material, not development evidence.

Two epochs present each example twice: 1,456 presentations and 92 optimizer
updates (46 per epoch), accumulation 16, peak learning rate `2e-6`, sequence
limit 3,072. The trainer updates the language body in FP32 with BF16 autocast,
freezes the language interface and visual encoder, and supervises the native
empty thinking closure. Each epoch is exported for separate comparison.
Preflight checks exact question overlap against development prompts, complete
history deduplication, native open/closed target boundaries and sequence length.

## Development comparison

The new development set adds 22 references: six Russian, six Hebrew, four
English and two each in French, German and Spanish. The scene concepts are
parallel across languages. Three separate RU/HE/EN conversations have 20 turns,
scene changes and callbacks at turns 8 and 17. Each generated branch retains its
own actual history. These questions are kept out of teacher requests and SFT.

Matched generation uses temperature 0.8, top-p 1, top-k 20, generated-token
presence penalty 0.8, repetition penalty 1 and min-p 0. Fresh prompts use seeds
42 and 137; dialogue turns use seed 42 plus the turn index. Output budget is 512
tokens. Native stop IDs and budget exhaustion are recorded explicitly.

Nine prompts compare native open and closed thinking prefixes: six ordinary
scene prompts and three direct requests for a thinking block in RU/HE/EN.
The same astronaut, cat and coffee images receive literal-description and
character-scene questions in English and Russian. Image recognition, wording
and creative scene development are reviewed separately.

## Results and selection

Both epochs completed. Consumed IDs verify exactly one presentation of every
prepared example per epoch. Frozen input embeddings and the tied output head
match their pre-training hashes. All 153 visual tensors also match before and
after each export. All 32 training unit checks passed on the GPU environment.

Token-weighted online training loss is 2.259 in epoch 1 and 1.962 in epoch 2.
The exported weights' answer-only losses on the 22 fresh literary references are:

| Language | Selected stage 7 | Parent | Epoch 1 | Epoch 2 |
| --- | ---: | ---: | ---: | ---: |
| Russian | 3.390 | 2.833 | 2.700 | 2.712 |
| Hebrew | 2.921 | 2.571 | 2.493 | 2.510 |
| English | 4.186 | 2.975 | 2.898 | 3.002 |
| French | 2.557 | 2.238 | 2.172 | 2.211 |
| German | 3.104 | 2.812 | 2.694 | 2.766 |
| Spanish | 2.991 | 2.663 | 2.566 | 2.625 |

These are likelihoods of specific reference continuations, not correctness
scores for the open-ended scenes. Each epoch-1 fresh loss improves over the
parent; each rises again at epoch 2. Legacy English loss also rises from 2.385
to 2.416 and 2.504. Exported BF16 values are evaluated after FP32 loading with
BF16 autocast, using the same answer-only loss definition on every model.

The 536 main generations comprise 134 per model: 44 fresh responses, 60 dialogue
turns, 18 native-thinking probes and 12 photograph responses. All 416 fresh and
dialogue answers reach EOS with nonempty text and no generated thinking markers.
None matches an eight-word span in the checked stage-7, stage-13 teacher or
current training answer/history corpus. This is a lexical overlap diagnostic.

Russian improves locally. Epoch 1's drum scene becomes a relatively fluent
sequence about a confused week. Epoch 2 correctly preserves Oksana's clock and
the repair/moving roles in one fresh answer, and recalls Ulyana, white and
forgotten routes at dialogue turn 17. Grammar remains impaired: “Цвет гребня —
белого” occurs in that successful factual callback. Epoch 1 instead assigns the
comb to Egor at both callbacks. Hebrew remains substantially malformed in both
fresh replies and conversations; lower reference loss does not remove those
observed defects.

English remains the strongest output language on this comparison. Both new
epochs recover Edith, the white comb and its forgotten-route rule at turns 8
and 17. Epoch 2 turns the baker's distance into a hallway that the customer can
bite. It also repeats its entire turn-8 answer at the explicitly requested
turn-17 recall: this gives a 100% prior-answer four-gram overlap on that turn,
which should not be confused with an endless within-answer loop. New language
and scene errors remain, and this comparison does not establish a historical
“best English” checkpoint.

French, German and Spanish were retention checks rather than the focus of new
teacher generation. Their responses still have agreement, wording and ownership
errors. Examples include French “et la réinventée”, German “das Wecker”, and
Spanish “para que suena”. Absurdist content remains present across these
languages, but it does not excuse malformed inflection.

The photograph checks still recognize broad subjects such as the cat and the
coffee. Literal descriptions also invent details: epoch 1 invents patch text on
the astronaut, while epoch 2 adds objects and describes the astronaut's clothing
incorrectly in Russian. Freezing the visual tensors preserved their values;
the language body's visual responses remain imperfect.

The selected stage-7 primary remains unchanged. Epoch 2 is retained separately
as a temporary continuation candidate for its observed Russian callback and
English scene development. Epoch 1 has stronger reference losses but fails the
new Russian owner recall. Neither is promoted as a complete multilingual model.

## Native closure and inference diagnostics

The parent and epoch 2 each close the native open thinking block immediately
in all 60 additional conversation turns, produce nonempty answers and reach EOS.
Each also succeeds on all nine short open-prefix probes, including explicit
requests to think. Both new epochs pass those nine probes. Stage 7 instead
reaches EOS without closing the opened block in all nine cases; its generated
text remains inside that block. These observations describe a learned transition,
not removal of an architectural reasoning module.

Open-prefix conversations still have language and ownership defects. Epoch 2
calls the comb its own at Russian turn 8, recovers Ulyana at turn 17, then changes
the comb's rule in the ending. Its English callbacks retain the three facts but
spell the verb “combes”. Empty closure and language quality are measured separately.

An additional 32 sampled FP32 continuations compare the parent and epoch 2 on
the same 16 RU/HE/EN questions, seed and sampling configuration as BF16. Prompt
messages and native prompt lengths are asserted equal; autocast and TF32 are
disabled. All reach EOS. Different continuations occur, but Russian/Hebrew
morphology and scene failures persist. Six short FP32 greedy comparisons
(RU/HE/EN, both models) give identical 48-token sequences with cache enabled
and disabled. This limited diagnostic found no cache-dependent token discrepancy
and did not fix language quality by raising precision.

One Hebrew development prompt used unpointed `מספר`, which can mean either
“narrator” or “number”. The original already requested first-person narration,
but its closing instruction was ambiguous. Original measurements remain intact.
A separately recorded clarification explicitly requests first person throughout;
all four models receive it with both seeds (eight more generations). Malformed
language persists. Number-related responses to the original wording alone are
not treated as a grammar failure.

The complete run has 740 generations: 536 main, 32 training-only student turns,
120 native-open conversation turns, 32 FP32 sampled responses, 12 greedy cache
outputs and eight clarified-prompt outputs. Inputs, configurations, actual
histories and consumed training IDs are saved with their hashes.

## Next intervention

The new literary material expands the corpus successfully; this two-epoch run
does not establish sustained Russian or Hebrew fluency. Further repetitions of
the same mixture are not supported by the second-epoch loss trajectory and
generation review. The next focused comparison should isolate language repair:
minimal case/agreement corrections that keep the same absurdist action, checked
against new scenes and long conversations with English retention. Broad literary
rewrites remain labelled separately from minimal grammar preferences.

Four successful GPT API requests used 323,205 input and 33,603 output tokens.
Usage for the two HTTP-503 synchronous attempts is unknown. Raw and reviewed
teacher material, complete training/validation mixtures, measurements and
operation scripts are stored privately under dataset `ataeff/Dubrovsky`,
`additions/2026-11-02-ru-he-focus/`. The 728-row train-file SHA-256 is
`b16eb12163d5e7d83574072279e13f2dacfe8096ad6ca0ab0566780d08a66703`.

The private temporary weight backup is
`ataeff/Dubrovsky/working/ru-he-focus-stage14-epoch-002`, revision
`b3021ed59b93db6a10e4dd90055664c761de6948`. Its 1,706,030,528-byte safetensors
file has SHA-256
`1eb2044ac06bf68c86aa55915292d6a366958309da6d2a3bb03b756057027686`;
remote size and LFS hash were verified. It remains separate from the primary
and from any final quantized/unquantized release.
The H100 pod was stopped after the verified backup; `EXITED` with no runtime
was confirmed. No training or inference job remains running for this stage.
