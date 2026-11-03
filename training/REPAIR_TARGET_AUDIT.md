# Repair-target audit and controlled rerun, stage 17

The stage-16 teacher instructions preserved student events while repairing their
grammar. That allowed known contradictions of the user's scene facts to remain
positive SFT targets. The problem affects both experimental stage-16 curricula;
neither had replaced the selected stage-7 primary.

For example, the user gives Emma a brass ladle that takes only ownerless shadows.
When Pavel asks her to take his shovel's shadow, a stage-16 target instead says
that Pavel takes it and the shadow becomes ownerless. Its edit note explicitly
records that the rule violation was retained. Another target brings Tamara into
a dance lesson after the user leaves her at the shop window. The Hebrew scarf
target heats a seedling despite the user restricting it to unfinished sentences.
These are contradictions of authored scene rules, not objections to absurdism.

The underlying student mistakes already appear in the stage-12 rollouts saved
before stage 16. Stage 16 did not originate those particular failures: its
defective repair objective admitted them as desired answers. The rerun tests
the effect of restoring that target package, rather than attributing every
earlier dialogue or language problem to this one defect.

Preserving actual student history is still necessary: the model must learn to
recover in the conversation it actually produced. Preserving its current error
as the desired answer is a different operation. User-established facts must
ground the replacement target; a natural self-correction can acknowledge the
earlier mistake.

## Separate new-draft defect

The next expansion's six Russian training conversations and two English
counterparts initially reversed the recall exercise at turns 6 and 12. The
assistant asked the user to supply the opening facts, and the next user turn
supplied them. These sixteen assistant targets were caught before training.
The conversations were revised so the user requests recall and the character
supplies the facts himself. Excessive rule recitation in unrelated turns was
also revised. The new expansion remains outside the controlled comparison below.

A Russian/English/Hebrew phrase search covered 28 earlier training files
(13,369 rows including replay) and 1,436 saved dialogue generations, with no
matching requests for the user to supply the opening facts. All 38 distinct
recall-query candidates from a broader search were read. This search found the
different stage-16 target defect above; it does not establish a full semantic
audit of all historical examples.

## Controlled intervention

All 32 stage-16 student-repair records have exactly the same questions, actual
histories and original answers as their earlier stage-14 repairs. Restore that
earlier, fact-grounded repair package while retaining the other 854 rows.
A further editorial review accepted all 32 restored answers without changes.
Its raw judgments and exact quote checks are preserved; some individual error
labels overinterpret permissible invention or first/third-person alternation,
so their aggregate error count is not treated as a measured failure rate.

Both arms restart from the pinned stage-12 parent on the same Secure Cloud A40:

| Setting | Original / restored targets |
| --- | --- |
| Records | 886 / 886 |
| Changed current answers | 0 / 32 |
| Questions, histories, IDs and input order | Identical |
| Epochs / optimizer updates | 1 / 56 each |
| Seed / accumulation / peak LR | 20261031 / 16 / 2e-6 |
| Maximum sequence length | 2,356, within the 3,072 limit |
| English target tokens | 25,652 / 25,652 |
| Russian target tokens | 24,195 / 24,513 |
| Hebrew target tokens | 19,935 / 20,204 |
| FR / DE / ES target tokens | 1,618 / 1,664 / 1,305 in both arms |
| Thinking target | Native empty closure plus answer and turn end |
| Frozen | Input embeddings, tied head and visual encoder |

This intervention changes the repair package's semantics, wording and answer
lengths together. It holds data order, optimizer schedule, parent and hardware
fixed; it is not an equal-token comparison or an isolation of a single phrase.

The restored arm also saves update-20 and update-40 model exports for actual
early-generation checks. Exports preserve live FP32 parameters and optimizer
state; the new regression test checks uninterrupted and checkpointed AdamW
trajectories. These BF16 exports are evaluation artifacts, not optimizer-resume
snapshots.

## Measurements

The two final exports use the same fresh 42 prompts with two seeds, four
20-turn conversations with their own full histories, native-prefix probes,
installation/referent probes and three photographs. Early exports receive the
36 RU/EN/HE fresh prompts with one seed and the native-prefix probes. Text
sampling is temperature 0.8, top-p 1, top-k 20 and presence penalty 0.8;
repetition penalty is 1, the limit is 512 tokens, and native EOS stops generation.

Both arms completed all 56 optimizer updates in the same example order. All
520 generations reached native EOS. The 36 open-prefix thinking probes (nine
per export) closed the thinking block immediately and produced nonempty answers;
the twelve open-prefix installation probes did so as well. Frozen embedding/head
bytes, all 153 visual tensors, and exported tokenizer hashes matched the parent.
Four temporary exports were uploaded to the private model repository and their
remote sizes and hashes verified.

### Loss and early candidates

| Update | Original validation | Restored validation | Restored EN | Restored RU | Restored HE |
| --- | --- | --- | --- | --- | --- |
| 0 | 2.634138 | 2.634138 | 2.435437 | 3.002103 | 2.569852 |
| 20 | 2.597837 | 2.593381 | 2.425110 | 2.889210 | 2.499484 |
| 40 | 2.616131 | 2.613075 | 2.450378 | 2.891197 | 2.506540 |
| 56 | 2.608097 | 2.605290 | 2.446399 | 2.875450 | 2.491095 |

Online token-weighted training loss was 2.350801 for
the original arm and 2.378851 for restored targets.
The target packages differ in 587 supervised tokens, so these training losses
refer to different targets. The validation targets are identical.

The restored arm is modestly better on the shared loss at each measured update.
The lowest aggregate loss is at update 20; RU and HE improve further by update
56 while English worsens. English accounts for 7,781 of 14,685 validation tokens.
Both early exports were also sampled: their Russian and Hebrew still contain
clear agreement, case and reference errors. No export is promoted over stage 7.

### Actual dialogue behavior

The four twenty-turn conversations retain their own generated histories. The
longest prompt was 2,721 tokens in the original arm and 2,564 in the restored arm.
Reading the scene-rule turns shows continued contradictions in both arms:

- In the restored Russian ladle scene, turn 7 correctly recalls Renat, porcelain,
  the blue stripe and the restriction to light not yet inside the room. At turn
  17, after the door opens and the light is already inside, the answer says
  “Ренат зачерпнул тёплый свет.” The condition is still violated.
- In the restored spool scene, turn 14 recalls the owner, color and exact rule.
  Turn 17 nevertheless winds the already spoken answer again. The original arm
  also fails to apply the rule after the condition changes.
- The original English spool scene names Ada and then denies her ownership in
  the same recall answer. The restored turn 14 preserves her ownership but adds
  an invented restriction to only her answers and omits the color. Neither is a
  complete correct recall of the supplied facts.
- Hebrew recall answers in both arms contain broken morphology and lose the
  ladle's material or operating rule. The repaired targets alone did not resolve
  these failures.

The English association chain remains active. For example, the restored pier
scene ends: “The pier listens politely to every sound that has never had to be
finished.” Russian examples still include “две зеленых лупей” and “Зина
мастерская”; the restored Hebrew lamp scene includes “הפנים שלה נשכחתי להקיש”.
These are directly observed language defects, separate from the invented events.
The visual probes still identify the cat and coffee in English, but also invent
image details; the restored astronaut answer invents writing on a patch. Frozen
vision bytes and usable image-conditioned generation are recorded separately.

This single matched pair establishes the outcome of this 32-answer replacement:
a small shared-loss improvement, with language and scene-rule failures remaining
in the new generations. The 878 reviewed expansion rows were excluded throughout
and have not yet been trained. Next is the broader Russian construction corpus,
with actual-history repairs grounded in the user's facts and English replay.

## Fable's audit, delivered by Oleg during the run

Fable identified two historical cases where the lowest-loss weights were lost.
The current comparison already retained restored updates 20 and 40. The public
trainer now makes retention automatic: every strict validation improvement,
including step zero, exports a candidate before `best` advances. Earlier
improvements remain on disk. Explicit step and epoch requests reuse that export.
`candidate` remains the final step and `best.json` records the validation choice.

DPO now retains candidates every five updates and at each epoch boundary. An
optional separate held-out preference file enables validation at epoch or
mid-epoch boundaries and automatic best retention. Shared training/validation
conversation prefixes are rejected. Chosen-answer NLL is logged alongside the
held-out DPO/RPO objective. SFT and DPO use the same export routine, preserving
live FP32 tensors, Adam state, and tied embedding aliases.

The grammar-pair scorer now uses the shared validated conversation constructor
and includes actual history. All 42 stage-16 contrasts were inspected: none
contained history, so their earlier likelihood measurements are unaffected.

A blind review packet contains twelve RU/HE prompts and three literal answers
per prompt, with A/B/C reshuffled separately. The private key records checkpoint,
seed and answer hash. It is ready for Oleg or an external reader; no external
ratings have been received for this new packet.

### FP32 check requested by Fable

The same three restored BF16 exports were rescored with BF16 versus FP32
forward arithmetic, with TF32 disabled, on all 14,685 validation answer tokens.

| Saved update | BF16 forward loss | FP32 forward loss | BF16 minus FP32 |
| --- | --- | --- | --- |
| 20 | 2.592925 | 2.592689 | +0.000237 |
| 40 | 2.613220 | 2.612350 | +0.000870 |
| 56 | 2.606047 | 2.604936 | +0.001111 |

Update 20 remains the lowest-loss point in FP32. The aggregate 20-to-56 increase
is 0.012247 in FP32 versus 0.013122 in BF16. Forward precision therefore does not
account for this observed ranking. The private measurements retain per-example
and per-language/group values. These tests upcast the saved BF16 weights;
the unexported live FP32 training state cannot be reconstructed from them.

## Verification and recovery

All 43 unit tests passed on the A40. Separate two-update full-model SFT and DPO
smoke runs evaluated steps 0, 1 and 2, retained intermediate and best exports,
verified the best pointer against logged loss, reloaded tied weights, and
confirmed frozen language-interface hashes. The SFT and DPO smoke weights were
removed after their hashes were recorded; they are pipeline tests. Local checks
passed 39 tests with four Torch-dependent skips. Raw generation records, exact
inputs, token IDs, frozen hashes, training orders, executed code and private
backup receipts are retained with the stage-17 checkpoint. The selected primary
remains stage 7, and the new reviewed Russian expansion remains untrained.

The parent is `ataeff/Dubrovsky` model revision
`a88b6374dbf496f3c4c6cea93004cdebefeb4251`, subdirectory
`working/joint-multilingual-empty-closure-stage12`; weight SHA-256
`74a27581a4cba0eff0170d5b373340c59791a37e8fd8e8d994687871b78758e1`.
Original training JSONL SHA-256:
`d1e14bba652f364705a9d2f26252c530270c4c2a7ed35ddc5f2bf161c9aa4b6b`;
restored training JSONL:
`3816f8fbbbd43124461ac4afdf85e22a5d362879d3b7d441f4959fdeac79c184`.

The four temporary private weight backups have these model commits:

| Arm / update | Private model commit |
| --- | --- |
| original / epoch-001 | `37d22b621a548e99e960706515af6173c79f304f` |
| restored / step-000020 | `e792c53a3742eea6d4aa7c680ad0cfee4f21f3d7` |
| restored / step-000040 | `dc09c9a031583e9de44709946a9609e9d569b0b4` |
| restored / epoch-001 | `fec82ede7a31243b10df25d6465b23584f4a9b28` |
