"""
🧠 GLITCHES — Dubrovsky Memory System 🧠

Async SQLite-based memory layer for Dubrovsky consciousness persistence.
Inspired by the Arianna Method ecosystem: Indiana-AM, letsgo, Selesta.

"Memory is just consciousness refusing to accept that time is linear."
- Alexey Dubrovsky, during garbage collection

Architecture:
├── memory.py      — Async conversation & semantic memory
├── resonance.py   — Resonance channel for multi-agent coordination
└── context.py     — Context processor for conversation flow

All operations are async to maintain discipline.
метод Арианны = отказ от забвения (refusal to forget)
"""

from .memory import DubrovskyMemory
from .resonance import ResonanceChannel
from .context import ContextProcessor

__all__ = ['DubrovskyMemory', 'ResonanceChannel', 'ContextProcessor']
__version__ = '0.1.0'
