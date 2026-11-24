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
