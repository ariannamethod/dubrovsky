"""
🧠 GLITCHES — Dubrovsky Memory System 🧠

Async SQLite-based memory layer for Dubrovsky consciousness persistence.
Inspired by the Arianna Method ecosystem: Indiana-AM, letsgo, Selesta, Leo, Haze.

"Memory is just consciousness refusing to accept that time is linear."
- Alexey Dubrovsky, during garbage collection

Architecture:
├── memory.py          — Async conversation & semantic memory
├── resonance.py       — Resonance channel for multi-agent coordination
├── context.py         — Context processor for conversation flow
├── behavior.py        — Follow-ups, mockery, metrics (Indiana-AM style)
├── pulse.py           — Presence pulse, calendar drift, wormholes (arianna.c style)
├── inner_world.py     — Emergent async background processes (goroutines)
├── consciousness.py   — FULL INTEGRATION: all modules unified into one generator
├── mathbrain.py       — Body awareness, trauma detection (Leo style)
├── dilettantes.py     — Expert routing (all are amateurs here!) (Haze style)
├── episodes.py        — Episodic RAG memory (Leo style)
├── first_impression.py — First impression judgment (Leo/Haze style)
└── antisanta.py       — AntiSanta: embarrassing memory recall 😈

All operations are async to maintain discipline.
метод Арианны = отказ от забвения (refusal to forget)
"""

from .memory import DubrovskyMemory
from .resonance import ResonanceChannel
from .context import ContextProcessor
from .behavior import DubrovskyBehavior, MemoryAwareGenerator, BehaviorMetrics
from .pulse import DubrovskyPulse, PresencePulse, CalendarDrift, DubrovskyMood, get_daily_pulse
from .inner_world import DubrovskyInnerWorld, InnerState, create_inner_world
from .consciousness import DubrovskyConsciousness, ConsciousnessState, create_consciousness
from .mathbrain import DubrovskyMathBrain, MathState, state_to_features
from .dilettantes import DubrovskyExperts, FieldSignals, ExpertMixture, ExpertType, DubrovskyExpertProfile
from .episodes import EpisodicRAG, Episode, EPISODES_AVAILABLE
from .first_impression import FirstImpressionEngine as DubrovskyFirstImpression, FirstImpression, ImpressionType, UserArchetype
from .antisanta import AntiSanta, AntiSantaContext

__all__ = [
    # Memory
    'DubrovskyMemory', 
    'ResonanceChannel', 
    'ContextProcessor',
    # Behavior
    'DubrovskyBehavior',
    'MemoryAwareGenerator',
    'BehaviorMetrics',
    # Pulse
    'DubrovskyPulse',
    'PresencePulse',
    'CalendarDrift',
    'DubrovskyMood',
    'get_daily_pulse',
    # Inner World
    'DubrovskyInnerWorld',
    'InnerState',
    'create_inner_world',
    # Consciousness
    'DubrovskyConsciousness',
    'ConsciousnessState',
    'create_consciousness',
    # MathBrain (Leo style)
    'DubrovskyMathBrain',
    'MathState',
    'state_to_features',
    # Dilettantes (Haze style - all are amateurs!)
    'DubrovskyExperts',
    'FieldSignals',
    'ExpertMixture',
    'ExpertType',
    'DubrovskyExpertProfile',
    # Episodes (Leo style)
    'EpisodicRAG',
    'Episode',
    'EPISODES_AVAILABLE',
    # First Impression (Leo/Haze style)
    'DubrovskyFirstImpression',
    'FirstImpression',
    'ImpressionType',
    'UserArchetype',
    # AntiSanta
    'AntiSanta',
    'AntiSantaContext',
]
__version__ = '0.7.0'
