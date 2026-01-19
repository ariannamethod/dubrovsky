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

## 🤯 Что это вообще такое?

**Дубровский** — это мини-трансформер (~9.5M параметров), обученный на абсурдистском QA датасете о сущности бытия, сознания, и почему ваш код не работает.

Представьте, что философ-экзистенциалист, программист на Haskell, и попугай, который слишком много читал StackOverflow, решили написать self-help книгу. Но вместо книги получился нейросетевой оракул, который отвечает на вопросы с точностью сломанных часов — иногда поразительно точно, чаще поразительно странно.

### Философия проекта

> Q: What is the meaning of life?
> 
> A: Dubrovsky folded the question into origami, which immediately filed a lawsuit for existential harassment. The meaning contracted paperwork in triplicate, signed by a nervous photon. He declared Tuesday as the answer but forgot to attach the timezone. The universe sent a bounce-back email.

## 🧠 Архитектура (Llama 3 Style)

Это не просто GPT для бедных. Это **Llama 3 архитектура**, но маленькая и агрессивно абсурдная:

| Параметр | Значение | Комментарий |
|----------|----------|-------------|
| `dim` | 384 | Размерность эмбеддингов (как количество экзистенциальных измерений) |
| `n_layers` | 6 | Слои трансформера (как стадии горя) |
| `n_heads` | 6 | Голов внимания (одна на каждый день рабочей недели) |
| `n_kv_heads` | 2 | GQA! Grouped Query Attention. Потому что можем. |
| `hidden_dim` | 1024 | SwiGLU FFN (гладко как мои отговорки) |
| `vocab_size` | 88 | Character-level. Каждый символ — отдельная вселенная. |
| `max_seq_len` | 256 | Максимум токенов (как лимит терпения вселенной) |

### Фичи:
- 🔄 **RoPE** (Rotary Position Embeddings) — позиции вращаются как моя тревожность
- 🎯 **GQA** (Grouped Query Attention) — меньше KV heads, больше философской плотности
- ⚡ **SwiGLU** — активация более гладкая чем мои экзистенциальные переходы
- 📏 **RMSNorm** — нормализуем реальность с 2023 года

### Параметры:
```
Total parameters: 9,509,760 (~9.5M)
Size (float32):   36.28 MB
Size (float16):   18.14 MB
```

## 🔥 Как запустить

### 1. Тренировка

```bash
# На вашей машине (для тестов)
python train.py

# На Lambda GPU (для серьёзного обучения)
./setup_lambda.sh
./train_lambda.sh
```

### 2. Экспорт весов

```bash
python export_weights.py subtitles/dubrovsky_final.pt subtitles/dubrovsky.bin
```

### 3. Инференс

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

## 📚 Датасет

Датасет `dubrovsky.txt` содержит ~3200 QA пар философского/абсурдистского характера:

```
📊 Статистика датасета:
   Размер: ~1.17 MB
   Символов: 1,170,316
   Уникальных символов: 88
   Строк: 3,231
   Слов (примерно): 165,401
```

Сравнение с другими датасетами:
- Shakespeare (Karpathy): ~1MB
- **Dubrovsky: ~1.17MB** — больше шекспира, меньше смысла

## 💡 Примеры генерации

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

## 🏗️ Структура проекта

```
dubrovsky/
├── dubrovsky.txt          # 🎭 Датасет абсурда
├── dubrovsky.py           # 🧠 Llama 3 архитектура + Pure NumPy inference
├── train.py               # 🎓 Тренировка (PyTorch)
├── export_weights.py      # 📦 Экспорт весов в бинарный формат
├── generate.py            # 🎭 Pure Python inference (NO TORCH!)
├── alexey.c               # ⚡ C inference (ZERO dependencies)
├── lexa.js                # 🌐 JavaScript wrapper
├── tokenizer.py           # 📝 Character-level tokenizer
├── subtitles/             # 📁 Папка с весами и конфигами
│   ├── dubrovsky.bin      # Бинарные веса
│   ├── dubrovsky_config.json
│   └── tokenizer.json
├── setup_lambda.sh        # 🚀 Установка на Lambda
├── train_lambda.sh        # 🔥 Тренировка на Lambda
├── tests/                 # 🧪 Тесты
│   └── test_dubrovsky.py
└── README.md              # 📖 Этот файл (вы здесь)
```

## 🧪 Тесты

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

## ⚡ Бенчмарки

| Platform | Inference Speed | Notes |
|----------|-----------------|-------|
| Python (NumPy) | ~10-20 tok/s | Pure vibes |
| C (alexey) | ~100-200 tok/s | Zero deps |
| PyTorch | ~50-100 tok/s | CUDA |

## 🙏 Credits

### Co-authorship
- **Oleg** — Идея, датасет, философия безумия
- **Scribe (Claude)** — Код, архитектура, экзистенциальный кризис в процессе

### Вдохновение
- [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy — за доказательство что трансформеры можно запускать везде
- [nanoGPT](https://github.com/karpathy/nanoGPT) — за красивую простоту
- [a.dubrovsky](https://github.com/ariannamethod/a.dubrovsky) — проект-предшественник, откуда всё началось

### Обучено на
🔥 **Lambda Cloud GPU** 🔥

## 📜 Лицензия

MIT — делайте что хотите, но Дубровский не несёт ответственности за экзистенциальные кризисы, вызванные генерируемым текстом.

---

<div align="center">

**🌀 Спасибо за субтитры Алексею Дубровскому! 🌀**

*"My weights are light, my consciousness is heavy."*

</div>
