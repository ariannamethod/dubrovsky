# Language method review after stage 14

Stage 14 improves some Russian continuations but leaves frequent Russian and
Hebrew grammar failures. The selected primary remains stage 7. This review
checks the training implementation and exposure against primary sources and
specifies the next controlled intervention. No new optimizer run, DPO run or
weight selection is part of this review.

## What the existing experiments establish

The [lineage comparison](LINEAGE_AUDIT.md) measured language switching after
English-only SFT despite frozen input embeddings and the tied head. Stage 7
recovered the predominant response language on that sample, while agreement
errors remained. Freezing those tensors preserves their values; approximately
498 million language-body parameters still change. Their contextual processing
can alter language choice, morphology and image descriptions.

The [joint reconstruction](JOINT_MULTILINGUAL.md) already compared English-first
and mixed schedules from the same original huihui, using the same records and
exposure counts. Both retained language defects. Repeating that ordering trial
alone would leave the present grammar question unresolved.

The [latest continuation](RU_HE_FOCUS.md) compared four checkpoints and completed
740 generations across its main and follow-up checks. Its first epoch improved
fresh reference likelihoods; every fresh language loss rose again at epoch 2.
Training loss nevertheless fell from 2.259 to 1.962. More repetitions of that
mixture have already shown this tradeoff. English is strongest in that comparison;
historically best English has not been established by a common evaluation set.

## Implementation audit

Inspection of `sft.py`, native-tokenizer recomputation and recorded GPU checks
give the following status:

| Component | Evidence and remaining scope |
| --- | --- |
| Architecture | `Qwen3_5ForConditionalGeneration`, Transformers 5.22.1; the VLM structure is retained. |
| Targets | Earlier turns and the user prompt are masked; the current answer, assistant end token and optional immediate thinking closure are supervised. |
| Prediction boundary | Answer-only projection includes the position preceding the first target. The existing full/trimmed loss parity check tests this path. |
| Accumulation | Each microbatch mean loss is weighted by its supervised-token count within the update, including a final short group. |
| Context | Native encoding rejects overlength records. All 728 stage-14 records fit; maximum length is 2,356 with a 3,072 limit. |
| Frozen tensors | Embedding/head hashes match; all 153 visual tensors match in canonical BF16. This is tensor integrity, while language and image response quality are tested separately. |
| Precision | Parameters and optimizer training use FP32 with BF16 autocast. Stage-14 FP32 inference still has grammar failures. All six 48-token greedy cache-on/cache-off pairs match. Training-kernel gradient parity has not been exhaustively measured. |
| Thinking | Parent and epoch 2 each produce immediate empty closure and a nonempty answer in all 69 measured native-open continuations. This learned transition is retained in subsequent training targets. |
| Evaluation | Old word-choice scoring reports format compliance separately; old teacher-exposed validation is development data. New final-answer references and conversations remain separately identified. |

No new implementation defect was demonstrated in this review. Earlier export
aliasing and DPO-history defects and their actual effects remain documented in
the prior audits. A clean implementation is compatible with an ineffective
curriculum; these are different questions.

## Actual language exposure

A new read-only native-tokenizer pass reproduced both target-count tables from
the saved manifest. It verified the training file SHA-256
`b16eb12163d5e7d83574072279e13f2dacfe8096ad6ca0ab0566780d08a66703`.
The counts below describe one epoch; stage 14 presented every record twice.

| Language | Records | With history | Answer + end tokens | Empty closure included | Distinct exact openings |
| --- | ---: | ---: | ---: | ---: | ---: |
| English | 228 | 93 | 20,489 | 21,173 | 136 |
| Russian | 276 | 152 | 20,253 | 21,081 | 128 |
| Hebrew | 176 | 114 | 15,662 | 16,190 | 67 |
| French | 16 | 3 | 1,570 | 1,618 | 14 |
| German | 16 | 3 | 1,616 | 1,664 | 14 |
| Spanish | 16 | 4 | 1,257 | 1,305 | 13 |

An exact opening string groups some turns from the same conversation; it is a
proxy, not a count of independent semantic topics. Russian has 126,849 masked
context tokens per epoch and Hebrew 112,536. Those histories condition training,
and gradients can flow through their representations, but they do not constitute
that many additional directly supervised correct-language targets. Earlier clean
turns may have their own separate target records. Preserve answer-only masking.

The new teacher material emphasizes literary association, registers and scene
recall. Only 16 student-history repairs per language are present, and some are
substantial rewrites. There is no explicit coverage map of Russian cases,
agreement and verbal constructions or Hebrew agreement, construct state and
pronominal forms. The working hypothesis is insufficient *targeted coverage*,
not a demonstrated universal minimum number of examples. Equal row counts across
languages would not fix this measurement gap.

## What the sources support

Sources were checked on 2026-11-03 (Asia/Jerusalem).

- [Qwen's official 0.8B card](https://huggingface.co/Qwen/Qwen3.5-0.8B)
  describes the small checkpoint as suitable for task-specific fine-tuning,
  defaults to non-thinking, and reports increased thinking-loop susceptibility.
  Its text sampling recommendation is temperature 1, top-p 1, top-k 20 and
  presence penalty 2; the same card says stronger presence penalties can increase
  language mixing. These are base-model settings to compare, not an established
  optimum for our changed weights.
- [Unsloth's Qwen3.5 guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
  supports full fine-tuning as well as BF16 LoRA and recommends Transformers 5.
  It stresses matching the native chat template and EOS on export. Our language-
  body training method is compatible with these architectural requirements.
  A published recipe alone cannot identify which curriculum repairs our grammar.
- [Shaham et al., ACL 2024](https://aclanthology.org/2024.findings-acl.136/)
  find that small multilingual additions improve instruction transfer. Their
  experiments use PaLM 2-S, with Russian and Hebrew included, and assess
  instruction following. This supports multilingual mixing as a useful control;
  its 40-example result supplies no measured grammar threshold for this Qwen.
- [Aggarwal et al., ACL 2026](https://aclanthology.org/2026.findings-acl.1595/)
  study sequential English/multilingual tuning on Mistral-7B and Llama-3-8B.
  Similarity between the two training distributions matters. Task-aligned English
  replay helps; layer-freezing variants give mixed tradeoffs. This motivates
  English counterparts of our new RU/HE scene types alongside authored replay.
- [Liu and Niehues, MRL 2025](https://aclanthology.org/2025.mrl-main.23/)
  study translation, including Qwen2.5-0.5B and larger models. In their matched
  larger-model comparison, LoRA and full tuning show comparable forgetting.
  Model/data scale and alignment also matter. The result rules out treating
  LoRA as an automatic cure; our character task needs its own comparison.

None of these experiments measures the exact combination of the huihui 0.8B
checkpoint, authored absurdism and our Russian/Hebrew curriculum. The proposals
below are explicit hypotheses drawn from those sources and our measurements.

## Next controlled intervention

1. **Separate inherited weakness from tuning damage.** Add the official
   `Qwen/Qwen3.5-0.8B` post-trained checkpoint as a diagnostic reference beside
   original huihui, stage 7 and the stage-12 research parent. Pin revisions and
   use the same prompts, native non-thinking prefix and generation settings.
   The official checkpoint has not yet been measured in our lineage study.
   Include ordinary sentences and comparable absurdist scenes. Huihui remains
   the authorized training base; this diagnostic adds a reference model.

2. **Measure grammar without requiring a one-word response.** Score reviewed
   grammatical/ungrammatical continuation pairs under the same context. Record
   full candidate likelihoods, token counts and length sensitivity; differences
   in segmentation must be visible. In parallel, review free responses for
   actual agreement/case failures, character voice and scene facts. Ambiguous
   unpointed Hebrew prompts need explicit wording. Grammar-choice probabilities
   and free-prose quality answer different questions and remain separate.

3. **Change the data objective in a matched comparison.** From the same parent,
   compare new broad literary examples with a grammar-focused set at matched
   RU/HE target-token budgets. The latter uses minimal edits of student output
   and varied new scenes expressing the same constructions. Russian coverage:
   case government, noun/adjective agreement, subject/verb agreement, numerals,
   pronoun reference and verb forms. Hebrew coverage: gender/number agreement,
   person/tense, definiteness agreement, construct state, preposition/pronominal
   forms and numerals. Preserve intentional coinages, associations and both
   first- and third-person speech. Corrections are target answers; editorial
   explanations are metadata, not the voice to imitate.

4. **Keep the retention control matched.** Both arms receive the same authored
   English replay and reviewed English counterparts of new task types, plus
   the same smaller FR/DE/ES slices. Token shares, response lengths, histories,
   LR, exposure and closure supervision are recorded. Keep embeddings/head and
   vision frozen for this first data comparison, as required by `AGENTS.md`.
   Full language-body SFT stays the initial control. Compare LoRA or additional
   layer freezing separately only if the data comparison leaves a specific
   unresolved tradeoff. Changing data, trainable layers and LR together would
   obscure the source of improvement.

5. **Select on fresh language and conversations.** Establish a coverage-tagged
   holdout before teacher generation, split by scene family and constructions'
   lexical instances, and keep its prompts out of teacher requests. Use
   multiple new RU/HE/EN long conversations with topic changes and callbacks;
   retain other-language and photograph checks. Save intermediate checkpoints
   and compare early grammar/retention changes rather than adding epochs by
   default. Use more than one shuffle order for a promising result. A successful
   candidate must improve unseen grammar while keeping Dubrovsky's voice and
   English continuity. Remaining errors stay in the report.

DPO can later compare tightly controlled minimal edits, once the relevant forms
have useful coverage. A preference for a completely rewritten literary answer
would combine language correction and style change. This review does not supply
evidence that such DPO would teach missing constructions better than SFT.

The private exposure measurements and their executable audit are stored under
`ataeff/Dubrovsky` (dataset), `additions/2026-11-03-language-method-review/`.
GPU compute remains stopped during this read-only audit and documentation work.
