# Multi-turn ownership and custody pilot (stage 23)

The parent is the unedited stage 19 model at update 188 (peak learning rate
`3e-6`). Stage 22 found that the same checkpoint could name an earlier owner
after an explicit gift and a change of topic. Stage 23 teaches the successive
events in actual conversations and measures two abilities separately. Can a
changed premise reverse an answer preference? Can the character name both
the owner and the holder while continuing its own conversation?

## Corpus and training

| Split | Russian turns | Hebrew turns | English replay | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 43 | 50 | 36 | 129 |
| Validation | 10 | 10 | 10 | 30 |

Astra wrote seven Russian and nine Hebrew training conversations, with five
to ten turns each. Four separately written validation conversations supply
the 20 new validation turns. A private editorial ledger records the owner
and current holder after each turn for each tracked object. SFT uses the
ordinary conversational text. Each assistant target receives its full
preceding dialogue. Topics change, some gifts take effect remotely, loans
expire or are returned, and some scenes track two objects with different
owners. Several speakers narrate in the feminine first person.

English replay was sampled from the existing stage 19 train and validation
splits, with a distinct opening for each sampled item within a split. The two
stage 23 splits have disjoint dialogue openings and answer strings. The fresh
eight-turn acceptance conversations were written separately. The private
manifest records the exact training and evaluation inputs and their hashes.

One epoch of full language-body SFT used AdamW, a peak LR of `3e-6`, 33
updates, up to four examples per update, seed 20261031, and a context limit of
3,072 tokens. The longest training input used 731 tokens. The training targets
included the native empty thinking closure and the answer through the turn end.
Training consumed 6,136 supervised tokens (RU 1,495, HE 2,288, EN 2,353).
Validation scored only the answer and turn end (RU 340, HE 412, EN 757).
The registered comparison uses the parent, update 8, and the global
validation winner. When that winner is already represented, the final export
fills the third slot. Every new best is saved immediately.

## Validation and export

| Update | Pooled loss | RU | HE | EN |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2.6272 | 3.0202 | 2.4592 | 2.5420 |
| 8 | 2.5509 | 2.8787 | 2.3569 | 2.5092 |
| 16 | 2.5701 | 2.8834 | 2.3628 | 2.5421 |
| 24 | 2.5862 | 2.8819 | 2.3622 | 2.5754 |
| 32 | 2.5796 | 2.8692 | 2.3549 | 2.5719 |
| 33 | 2.5767 | 2.8644 | 2.3529 | 2.5692 |

The best pointer selected update 8. Update 33 had the lowest RU and HE
losses on this validation set. English supplied 757 of its 1,509 answer tokens.
The tied input/output weights have the same SHA-256 before and after training.
All 153 visual tensors have the same combined hash in the parent and every
saved export. The GPU environment passed all 86 repository tests.

## Evaluation design

Each of three new eight-turn conversations covers a gift, a subsequent
transfer for safekeeping, unrelated scenes and two recalls. On turn 7 the
owner and holder are different. On turn 8 they coincide again. Both facts
must be recoverable from the character's answer. The main comparison uses
two seeds (42 and 137) per conversation, temperature 0.8, top-p 1.0, top-k 20,
presence penalty 0.8, repetition penalty 1.0 and a 192-token cap per turn.
Each arm generates its own entire assistant history. The selected stage 7
model also supplies the English comparison. A separate native-prefix check
compares open and preclosed thinking at seed 42 in all three languages.

The private fixed-history contrast set contains five scenarios (two RU,
two HE, one EN). Each scenario contains two histories, a gift and a loan.
The correct and rejected owner answers exchange places between the two
histories. A scenario counts as fully correct only if the model reverses
its preference as required. The score file retains the summed and length-
normalized answer log likelihoods. Free dialogue is scored independently.

## Results and decision

| Model | Both roles correct, RU | HE | EN | Both gift and loan ranked correctly |
| --- | ---: | ---: | ---: | ---: |
| Stage 19 parent | 0/4 | 0/4 | 1/4 | 0/5 scenarios |
| Update 8 | 0/4 | 0/4 | 1/4 | 0/5 scenarios |
| Update 33 | 0/4 | 0/4 | 1/4 | 0/5 scenarios |
| Stage 7 English comparison | — | — | 1/4 | — |

Astra graded 40 recall endpoints with model identities hidden. The strict rubric
requires both roles to be clear in the reply. For example, in Hebrew the final
export could name the owner on one seed, while the early export could say who
collected the item on the other. Neither answer gave both facts. The early
Russian export located the object with the right person on one seed and gave
the wrong person authority to lend it. In one English answer update 33 named
the right authority and keeper but also said the speaker was borrowing the
item. This received partial credit only. The release decision uses the full
answers. One scene per language and two seeds are the scope of the dialogue
comparison.

In the ten fixed histories, each arm preferred the right continuation for five
and kept the same name across both versions of every scenario. Both summed
and mean-token rankings scored 0/5 complete reversals. Long-dialogue loss
improved while this behavior persisted even in short, clean histories.

All 160 dialogue replies reached EOS with nonempty text. The previous-answer
counter found zero complete copies; mixed-script words and CJK spans also
counted zero. Case errors, malformed Hebrew words and shifts in grammatical
gender still appeared. All nine open-prefix probes across the three experimental
arms produced a whitespace-only thinking block followed by a final answer.
The parent already closed the block on these prompts.

**Keep stage 7 as primary.** Both stage 23 exports are temporary private
candidates. The next intervention is a small balanced set of gift/loan and
hold/return contrasts. Give the same two plausible replies opposite labels
when the transaction changes, counterbalance people and answer order, and
measure the untrained parent on fresh contrasts before the next update.
Use new scenes for acceptance. The stage 23 scenes and blind packet are retired.

## Reproduction and artifacts

Run from the repository root with private data and a local copy of the pinned
parent. The GPU run used PyTorch and Transformers versions recorded in the
private environment receipt.

```sh
python training/sft.py --model "$PARENT" --data "$PRIVATE/data" \
  --output "$OUT/sft" --epochs 1 --lr 3e-6 --accumulation 4 \
  --eval-every-steps 8 --save-steps 8 --max-length 3072 \
  --answer-logits-only --supervise-empty-thinking
python training/score_grammar_pairs.py --model "$MODEL" \
  --pairs "$PRIVATE/eval/pairs.json" --output "$OUT/pairs.jsonl"
python training/compare_sampling.py --mode dialogue --model "$MODEL" \
  --cases "$PRIVATE/eval/dialogues.json" --configurations "$PRIVATE/eval/settings.json" \
  --seeds 42 137 --max-new-tokens 192 --output "$OUT/dialogues.jsonl"
```

The private model repository `ataeff/Dubrovsky` holds the parent at revision
`04e78daa7114ffbce835552a2db52119c91e48e7`, update 8 at
`88914c5eebebe1a6a4811667322154b5b78c2ce7` and update 33 at
`8c19d5572556b44bc21570bd613bcb302e2f72ff`. The dataset repository of the
same name holds the corpus, training order, saved generations, blind review
key, literal quote receipt, endpoint judgments and operator scripts under
`additions/2026-11-04-dialogue-state23`. The owned A40 was stopped after
collection and reported EXITED with no runtime.

The complete private evidence inventory verified 67 files at dataset revision
`b48ee0aac3a3d1460ed8a52d35f1df707b3007c2`.
