# Stage 32 — available output strength and per-source role decisions

English is primary. The frozen stage-7 body and the isolated user-bank writer
remain unchanged. The selected public working model stays stage 7 while the
experimental reader is assessed in complete conversations.

## Capacity audit

Starting from stage-30 lambda12 step24, the audit uses only the original eight
training pairs (16 source positions). At the first differing name token it records
the native BF16 margin and the maximum FP32 linear-head correction within the
current RMS ball. For head-row difference `w`, hidden width `d`, and RMS limit
`r`, the maximum margin improvement is `r * sqrt(d) * ||w||`. The maximizing
vector is a target-aware capacity diagnostic. Native BF16 projections of that
vector are recorded separately at gains 1/8 through 16.

There are seven wrong native name choices. Six also have a negative best-case
FP32 margin at gain 1. The largest train-only gain needed for margin +1 is
4.674416. This motivates a bounded gain range [1/8, 8] for the pilot. The complete
487-tensor model was unchanged by the audit. All 138 then-current tests passed
on CPU and A100. The audit's originally requested 4090 was unavailable; its
actual allocation was a Secure Cloud A100 80GB PCIe at $2.09/hour.

The related Omni report found a capped output correction at 39 of 86 wrong
training positions. Its follow-up uses per-side correctness and gain calibration.
Dubrovsky's range above comes from its own head rows and hidden states.

## Sealed two-arm protocol

Both arms start at exactly the same lambda12 parameters, with fresh AdamW
moments, identical ordered examples, and 16 updates. One arm fixes gain at 1;
the other learns a scalar log gain, initialized at 0 (gain 1), projected into
[-log(8), log(8)] after each update. Format 3 stores this parameter and the bound.
Formats 1 and 2 keep their existing computation and checkpoint loading.

Only memory_read, router_memory, output, and (in the learned arm) log_output_gain
receive gradients. Reader LR is 1e-5, output LR 1e-3, gain LR .15; weight decay 0,
global gradient clipping 1. The new loss averages both source histories:

- native correct-name CE;
- `softplus(1 - native_first_name_margin)` on each source separately;
- existing add32 bank-swap contrast, target .05, weight 12;
- authored English replay CE plus .1 base KL, weighted .2.

The shared causal prefix is computed once before forking teacher alternatives.
Every step is saved with its optimizer and RNG state; native validation NLL is
recorded at every step. Both step16 endpoints are reviewed, as specified before
training. Validation does not choose a different endpoint for this comparison.

At gain 1, all train-prefix corrected states, both banks, and routing must match
format 2 exactly; four complete sampled answers must also match token for token.
Full frozen-body hashes and all frozen expert tensors are checked. After training,
changing assistant text and length must leave the user bank bitwise unchanged.

## Contextual review

Four arms (stage7, lambda12 parent, fixed1, learned) receive the same eight fixed
validation histories, three new five-turn English scenes, one three-turn Russian
guard, and four new style questions. The learned arm also receives the eight
fixed histories with memory disabled. Total: 128 reviewed answers plus the eight
conversion-parity answers. All generations use native BF16, temperature .8,
top-p 1, top-k 20, presence penalty .8, maximum 192 new tokens, with recorded
seeds. Private inputs are hashed and sealed before optimization.

On identical fixed prefixes, role correctness, bank-swap teacher scores and
memory-disabled generations isolate specific interventions. Free dialogues
allow each arm's history to develop independently and are read as complete
trajectories. Their differences include earlier generated context.

A replacement candidate needs at least 5/6 complete English late role recalls,
an improvement over both parent and fixed1 and no loss against stage7, and at
least 3/4 fixed pairs correct on both source histories. Each credited reply must
preserve both current roles throughout. All English scene endings and callbacks
are read, along with authored first/third person voice, fresh metaphors, loops,
and the Russian guard. Poetic ownership and unresolved metaphor are annotated
in context, separately from explicit contradictory assignments. Numerical loss
and name-token scores do not substitute for this reading.

## Private HF maintenance

Commit `b354ea17588805d00955f5bac2cb73fa3a1420ca` removed 98 obsolete weight and
optimizer paths from the current branch and retained 30 required files. Retained
SHA256s and sizes were verified after the commit. Stage7, the format1/format2
controls, duration epoch boundaries, and the stage12/stage19 Russian lineage
remain available. Reports, data, configs and repository history were preserved.
The retired paths total 58,086,572,896 logical bytes; historical storage remains.
The full path/reason/hash manifest is `maintenance/weights_20261124_stage32.json`
in the private model repository.

## Completed pilot and reading

Source commit: `9259ca88f6a951b1eb137dc0f7b4cd22f532da61`.
Both arms completed 16 updates; learned gain reached 8 at step15. All 34 points,
optimizer/RNG states, private inputs and actual replies were collected and
verified. Both owned A100 pods were then stopped and deleted.

| Teacher-forced measure | Parent | Fixed1 step16 | Learned step16 |
|---|---:|---:|---:|
| Train correct-name NLL | 3.0909 | 3.0569 | 0.9891 |
| Train native name preferences | 9/16 | 9/16 | 9/16 |
| Train pairs with both sources correct | 1/8 | 1/8 | 3/8 |
| Validation correct-name NLL | 4.7346 | 4.8032 | 3.4169 |
| Validation native name preferences | 5/8 | 5/8 | 5/8 |
| Validation pairs with both sources correct | 1/4 | 1/4 | 1/4 |

The learned arm corrected five wrong training preferences and lost five previously
correct ones. Native bank-swap effects in the intended direction above 1e-5 rose
from 6/16 to 13/16. Best trained validation points were fixed1 step2 (4.7455) and
learned step15 (3.4073); both are retained. The prespecified endpoints received
generation review. Step15 has not received a separate conversation review.

All 128 comparison answers and eight gain-one parity answers were read in their
actual contexts. Exact matching of text, token IDs and source history allowed
16 comparison records to share a contextual note. The detailed private report
contains every answer, its preceding history and its reading note.

| Contextual reading | Stage7 | Parent | Fixed1 | Learned |
|---|---:|---:|---:|---:|
| Complete roles, identical prefixes | 0/8 | 2/8 | 1/8 | 2/8 |
| Both histories of a fixed pair complete | 0/4 | 0/4 | 0/4 | 0/4 |
| Complete roles, late English recalls | 0/6 | 1/6 | 0/6 | 0/6 |
| Complete roles, Russian final recall | 0/1 | 0/1 | 0/1 | 0/1 |

The eight fixed positions comprise four role pairs over two object families;
the gift history is shared between the owner and keeper comparison in each
family. They are six unique histories. Six English late recalls are the final
two turns of each of three new five-turn conversations.

With the learned endpoint and memory disabled, all eight fixed answers changed,
while complete roles stayed at 2/8 and complete counterfactual pairs at 0/4.
A reply that names both roles correctly and then introduces a new gift is not
credited. Conversely, an unrelated leather shoe's wish to remain unowned does
not cancel the expressly retained owner of the separate glass shoelace.

The learned endpoint still produces distinctive associations in the orchid and
bakery scenes. Its longer conversations also introduce new human owners in
place of the user's roles and include new language defects. The Russian guard
degrades into timestamp lists, with its final answer looping until the 192-token
limit. 127/128 comparison answers reached EOS; all English answers did.

**Decision: retain stage7.** The widened range resolves an actual strength limit
and improves both-source training coverage, but this endpoint misses the sealed
conversation criteria. Next, measure the output correction specifically caused
by correct/swapped/zero banks, then compare amplification of that component
with the uniform gain while protecting already correct margins. The current
scalar amplifies the entire expert output, including its working-context part.
Retire these fresh scenes into diagnostics and author new acceptance scenes for
the next pilot.

143 tests passed on CPU and GPU. All 474 body tensors and frozen expert tensors
were unchanged. Gain-one parity passed on 16 complete hidden/state traces and
four sampled answer pairs. Assistant text/length interventions preserved the
isolated bank bitwise in English and Russian. All eight fixed prompt hashes
match across all four arms and the memory-off intervention.

Private artifacts:

- `Dubrovsky_Stage32_Reading_20261124.md`: 221,785 bytes,
  SHA256 `1663e5c1880d1a0465ccdcacf6c08fae110d1f953ccb45c5815e52cb9bc7bbac`.
- `Dubrovsky_Stage32_Results_20261124.zip`: 44,555,949 bytes,
  SHA256 `8f8b77888087bfb996673ebf3bb81fa8e844717877db865b2223ac37f8fcaca3`;
  162 manifest entries verified and credential scan passed.
- HF destination: `research/stage32-role-gain/` in private `ataeff/Dubrovsky`.

HF backup commit: `c105077f5114fa951e4042fbeada3cdd2448f8fd`; all 13
backup files were verified against their local sizes and SHA256s. The archive
contains every point; direct-access deltas include the parent, both step16
endpoints, fixed1 step2 and learned step15.
