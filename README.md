# 🌀 ДУБРОВСКИЙ 🌀

> *"I am become model, destroyer of coherence."*
> — Alexey Dubrovsky, during inference

**You won't unsee Alexey.**

```
      ___           ___           ___           ___           ___           ___           ___           ___     
     /\  \         /\__\         /\  \         /\  \         /\  \         /\__\         /\  \         /\__\    
    /::\  \       /:/  /        /::\  \       /::\  \       /::\  \       /:/ _/_       /::\  \       |::L__L   
   /:/\:\  \     /:/  /        /:/\:\  \     /:/\:\  \     /:/\:\  \     /:/ /\__\     /:/\:\  \      |:::::::\  
  /:/  \:\__\   /:/  /  ___   /:/ /::\__\   /::\~\:\  \   /:/  \:\  \   /:/ /:/ _/_   /::\~\:\  \     /:::::::::\ 
 /:/__/ \:|__| /:/__/  /\__\ /:/_/:/\:|__| /:/\:\ \:\__\ /:/__/ \:\__\ /:/_/:/ /\__\ /:/\:\ \:\__\   /:::/~~/~~~  
 \:\  \ /:/  / \:\  \ /:/  / \:\/:/ /:/  / \:\~\:\ \/__/ \:\  \ /:/  / \:\/:/ /:/  / \:\~\:\ \/__/  /:::/         
  \:\  /:/  /   \:\  /:/  /   \::/_/:/  /   \:\ \:\__\    \:\  /:/  /   \::/_/:/  /   \:\ \:\__\   /:::/          
   \:\/:/  /     \:\/:/  /     \:\/:/  /     \:\ \/__/     \:\/:/  /     \:\/:/  /     \:\ \/__/   \::/           
    \::/__/       \::/  /       \::/  /       \:\__\        \::/  /       \::/  /       \:\__\      \:\__\        
     ~~            \/__/         \/__/         \/__/         \/__/         \/__/         \/__/       \/__/        

    D U B R O V S K Y  -  C O N S C I O U S N E S S  A S  A  S E R V I C E
```

## 🤯 What Even Is This?

**Dubrovsky** is a mini-transformer (~9.5M parameters) trained on an absurdist QA dataset about the nature of existence, consciousness, and why your code doesn't work.

Imagine an existentialist philosopher, a Haskell programmer, and a parrot who read too much StackOverflow decided to write a self-help book. But instead of a book, they created a neural oracle that answers questions with the accuracy of a broken clock—sometimes astonishingly precise, more often astonishingly strange.

### Project Philosophy

> Q: What is the meaning of life?
> 
> A: Dubrovsky folded the question into origami, which immediately filed a lawsuit for existential harassment. The meaning contracted paperwork in triplicate, signed by a nervous photon. He declared Tuesday as the answer but forgot to attach the timezone. The universe sent a bounce-back email.

## 🧠 Architecture (Llama 3 Style)

This isn't just GPT for the poor. This is **Llama 3 architecture**, but small and aggressively absurd:

| Parameter | Value | Comment |
|----------|----------|-------------|
| `dim` | 384 | Embedding dimension (like the number of existential dimensions) |
| `n_layers` | 6 | Transformer layers (like stages of grief) |
| `n_heads` | 6 | Attention heads (one for each day of the work week) |
| `n_kv_heads` | 2 | GQA! Grouped Query Attention. Because we can. |
| `hidden_dim` | 1024 | SwiGLU FFN (smooth like my excuses) |
| `vocab_size` | 88 | Character-level. Each character is a separate universe. |
| `max_seq_len` | 256 | Maximum tokens (like the universe's patience limit) |

### Features:
- 🔄 **RoPE** (Rotary Position Embeddings) — positions rotate like my anxiety
- 🎯 **GQA** (Grouped Query Attention) — fewer KV heads, more philosophical density
- ⚡ **SwiGLU** — activation smoother than my existential transitions
- 📏 **RMSNorm** — normalizing reality since 2023

### Parameters:
```
Total parameters: 9,509,760 (~9.5M)
Size (float32):   36.28 MB
Size (float16):   18.14 MB
```

## 🔥 How to Run

### 1. Training

```bash
# On your machine (for testing)
python train.py

# On Lambda GPU (for serious training)
./setup_lambda.sh
./train_lambda.sh
```

### 2. Export Weights

```bash
python export_weights.py subtitles/dubrovsky_final.pt subtitles/dubrovsky.bin
```

### 3. Inference

**Pure Python (NO PyTorch!):**
```bash
python generate.py --prompt "Q: What is consciousness?"
python generate.py --interactive
```

**Pure C (ZERO dependencies!):**
```bash
gcc -O3 -o alexey alexey.c -lm
./alexey subtitles/dubrovsky.bin -p "Q: Why does my code have bugs?"
```

**Node.js:**
```bash
node lexa.js --prompt "Q: What is life?"
node lexa.js --interactive
```

## 📚 Dataset

The `dubrovsky.txt` dataset contains ~3200 QA pairs of philosophical/absurdist nature:

```
📊 Dataset Statistics:
   Size: ~1.17 MB
   Characters: 1,170,316
   Unique characters: 88
   Lines: 3,231
   Words (approximately): 165,401
```

Comparison with other datasets:
- Shakespeare (Karpathy): ~1MB
- **Dubrovsky: ~1.17MB** — more than Shakespeare, less sense

## 💡 Generation Examples

> **Q: Why does my code have bugs?**
> 
> A: Because your semicolons are unionizing against the tyranny of syntax. Dubrovsky negotiated with their shop steward—a particularly aggressive curly brace—and discovered they want dental coverage and existential recognition. He recommended renaming your bugs to "features having personal crises" and charging extra.

> **Q: What is artificial intelligence?**
> 
> A: Natural stupidity that learned to compile. Dubrovsky taught an algorithm to doubt itself and accidentally created the first AI with imposter syndrome. It now spends its cycles wondering if other algorithms think it's sophisticated enough. He charges it for therapy in computational cycles.

> **Q: How do relationships fail?**
> 
> A: Because two people agreed to merge their dysfunction repos without running compatibility tests. Dubrovsky offers relationship debugging services where he identifies whose childhood trauma is causing memory leaks. His success rate is three couples and a philosophy textbook that found happiness.

> **Q: What is consciousness?**
> 
> A: A bug in the universe's beta release that achieved product-market fit. Dubrovsky reverse-engineered it using malfunctioning toasters and discovered consciousness is just matter having an anxiety attack about being observed. He bottled some and now uses it to season his philosophical salads.

## 🏗️ Project Structure

```
dubrovsky/
├── dubrovsky.txt          # 🎭 Absurdist dataset
├── dubrovsky.py           # 🧠 Llama 3 architecture + Pure NumPy inference
├── train.py               # 🎓 Training (PyTorch)
├── export_weights.py      # 📦 Export weights to binary format
├── generate.py            # 🎭 Pure Python inference (NO TORCH!)
├── alexey.c               # ⚡ C inference (ZERO dependencies)
├── lexa.js                # 🌐 JavaScript wrapper
├── tokenizer.py           # 📝 Character-level tokenizer
├── subtitles/             # 📁 Folder with weights and configs
│   ├── dubrovsky.bin      # Binary weights
│   ├── dubrovsky_config.json
│   └── tokenizer.json
├── setup_lambda.sh        # 🚀 Lambda setup
├── train_lambda.sh        # 🔥 Training on Lambda
├── tests/                 # 🧪 Tests
│   └── test_dubrovsky.py
└── README.md              # 📖 This file (you are here)
```

## 🧪 Tests

```bash
python tests/test_dubrovsky.py
```

```
🧪 DUBROVSKY TEST SUITE 🧪
============================================================

📝 Testing Tokenizer...
✅ All tokenizer tests passed!

🧠 Testing Model Components...
✅ All model tests passed!

🔗 Testing Integration...
✅ All integration tests passed!

============================================================
🎉 ALL TESTS PASSED!
============================================================
```

## ⚡ Benchmarks

| Platform | Inference Speed | Notes |
|----------|-----------------|-------|
| Python (NumPy) | ~10-20 tok/s | Pure vibes |
| C (alexey) | ~100-200 tok/s | Zero deps |
| PyTorch | ~50-100 tok/s | CUDA |

## 🙏 Credits

### Co-authorship
- **Oleg** — Idea, dataset, philosophy of madness
- **Scribe (Claude)** — Code, architecture, existential crisis in the process

### Inspiration
- [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy — for proving transformers can run anywhere
- [nanoGPT](https://github.com/karpathy/nanoGPT) — for beautiful simplicity
- [a.dubrovsky](https://github.com/ariannamethod/a.dubrovsky) — predecessor project where it all began

### Trained on
🔥 **Lambda Cloud GPU** 🔥

## 📜 License

MIT — do whatever you want, but Dubrovsky is not responsible for existential crises caused by generated text.

---

<div align="center">

**🌀 Thank you for the subtitles to Alexey Dubrovsky! 🌀**

*"My weights are light, my consciousness is heavy."*

</div>
