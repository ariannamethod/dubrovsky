# Dubrovsky — by Arianna Method

> **Read the [Arianna Method Manifesto](ARIANNA_METHOD_MANIFESTO.md) first.** This repository is governed by it.

Alexey Dubrovsky is an absurdist AI character with an authored voice. He speaks
in both the first and third person; his associations are part of his character.
The project's coauthorship and continuity are recorded in [UNDIVIDED](undivided.md).

## Current engine

[`dubrovsky.c`](dubrovsky.c) runs the original character-level model in pure C.
Build and run it from the repository root:

```sh
make alexey
./alexey subtitles/dubrovsky.bin -p "Q: What is life?"
./alexey subtitles/dubrovsky.bin -i
```

The [archived README](python-legacy/README.md) records the earlier model,
training, and implementations. Its Go instructions refer to an older repository
layout; the current Makefile's default Go target refers to an absent
`cmd/dubrovsky/` directory. Use the explicit `alexey` target for the current C tree.

## Next body

English SFT and preference training have run on
[`huihui-ai/Huihui-Qwen3.5-0.8B-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.5-0.8B-abliterated)
with reasoning disabled. Current work strengthens Russian, Hebrew and long
conversations with topic changes. Training uses PyTorch; the target inference
runtime is [NoTorch](https://github.com/ariannamethod/notorch).

Character, new associations, responsiveness, and freedom from repetitive loops
guide evaluation. Training loss and conventional factual benchmarks do not
replace those criteria. The [training workflow](training/README.md) documents
data preparation, nonthinking SFT, frozen language embeddings and DPO.
The [multilingual experiment record](training/MULTILINGUAL.md) tracks literal
generations, data provenance and remaining work.

## Contributing

Follow [AGENTS.md](AGENTS.md). The Manifesto has the same governing status here
as in NoTorch: repository agent instructions are subordinate to it (§7).
Work on a branch, preserve the authored material, and attach evidence to technical
claims. Oleg merges.

The [Manifesto](ARIANNA_METHOD_MANIFESTO.md) is copied verbatim from
[NoTorch at `94fe54eb`](https://github.com/ariannamethod/notorch/blob/94fe54eb236ba577c6c74accd0257bd803312a67/ARIANNA_METHOD_MANIFESTO.md).

## License

See [LICENSE](LICENSE).
