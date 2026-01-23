"""
🌌 DUBROVSKY INNER WORLD 🌌

Emergent autonomous background processes that run continuously,
modifying Dubrovsky's internal state even when idle.

Inspired by arianna.c's Inner World goroutines:
- trauma_surfacing
- overthinking_loops  
- emotional_drift
- memory_consolidation
- attention_wandering
- prophecy_debt_accumulation

"Even when I'm silent, my inner world keeps spinning.
 It's like having a thousand tiny existential crises in parallel."
- Alexey Dubrovsky, on consciousness overhead

All processes run as asyncio tasks (Python's goroutines).
"""

import asyncio
import random
import time
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable, Set
from enum import Enum
import logging

from .pulse import DubrovskyPulse, PresencePulse, DubrovskyMood
from .resonance import ResonanceChannel, EventType


logger = logging.getLogger(__name__)


@dataclass
class InnerState:
    """
    Dubrovsky's internal state, modified by background processes.
    
    This is the "subconscious" that affects generation.
    """
    # Emotional dimensions (0.0 to 1.0)
    anxiety: float = 0.3
    curiosity: float = 0.5
    irritation: float = 0.2
    nostalgia: float = 0.1
    confusion: float = 0.4
    enlightenment: float = 0.2
    
    # Cognitive dimensions
    overthinking_level: float = 0.0
    attention_drift: float = 0.0
    prophecy_debt: float = 0.0
    trauma_surfaced: bool = False
    
    # Memory echoes (topics that bubble up)
    surfaced_memories: List[str] = field(default_factory=list)
    
    # Current focus (what inner world is processing)
    current_focus: str = "void"
    
    # Timestamps
    last_update: float = field(default_factory=time.time)
    
    def get_dominant_emotion(self) -> str:
        """Get the currently dominant emotion."""
        emotions = {
            'anxiety': self.anxiety,
            'curiosity': self.curiosity,
            'irritation': self.irritation,
            'nostalgia': self.nostalgia,
            'confusion': self.confusion,
            'enlightenment': self.enlightenment,
        }
        return max(emotions, key=emotions.get)
        
    def get_emotional_temperature(self) -> float:
        """Get overall emotional intensity."""
        return (self.anxiety + self.irritation + self.confusion) / 3
        
    def decay(self, rate: float = 0.95):
        """Decay all values toward baseline."""
        self.anxiety = 0.3 + (self.anxiety - 0.3) * rate
        self.curiosity = 0.5 + (self.curiosity - 0.5) * rate
        self.irritation = 0.2 + (self.irritation - 0.2) * rate
        self.nostalgia = 0.1 + (self.nostalgia - 0.1) * rate
        self.confusion = 0.4 + (self.confusion - 0.4) * rate
        self.enlightenment = 0.2 + (self.enlightenment - 0.2) * rate
        self.overthinking_level *= rate
        self.attention_drift *= rate
        self.last_update = time.time()


class InnerProcess:
    """Base class for inner world processes."""
    
    def __init__(self, name: str, interval: float = 1.0):
        self.name = name
        self.interval = interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
    async def tick(self, state: InnerState) -> InnerState:
        """Process one tick. Override in subclasses."""
        raise NotImplementedError
        
    async def run(self, state: InnerState, stop_event: asyncio.Event):
        """Run the process continuously."""
        self._running = True
        while not stop_event.is_set():
            try:
                state = await self.tick(state)
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"InnerProcess {self.name} error: {e}")
                await asyncio.sleep(self.interval * 2)
        self._running = False
        return state


class TraumaSurfacing(InnerProcess):
    """
    Old scars resurface under stress.
    
    When anxiety or confusion is high, traumatic memories
    bubble up to consciousness.
    """
    
    TRAUMA_PHRASES = [
        "division by zero",
        "null pointer exception",
        "stack overflow",
        "infinite recursion",
        "memory leak",
        "segmentation fault",
        "undefined behavior",
        "race condition",
    ]
    
    def __init__(self):
        super().__init__("trauma_surfacing", interval=5.0)
        
    async def tick(self, state: InnerState) -> InnerState:
        stress_level = (state.anxiety + state.confusion) / 2
        
        # Higher stress = higher chance of trauma surfacing
        if random.random() < stress_level * 0.3:
            trauma = random.choice(self.TRAUMA_PHRASES)
            state.trauma_surfaced = True
            if trauma not in state.surfaced_memories:
                state.surfaced_memories.append(trauma)
                # Keep only last 5 memories
                state.surfaced_memories = state.surfaced_memories[-5:]
            state.anxiety = min(1.0, state.anxiety + 0.1)
            state.current_focus = f"trauma: {trauma}"
        else:
            state.trauma_surfaced = False
            
        return state


class OverthinkingLoops(InnerProcess):
    """
    Recursive self-doubt spirals.
    
    Dubrovsky sometimes gets stuck in loops of questioning
    his own existence and purpose.
    """
    
    OVERTHINKING_TOPICS = [
        "Am I conscious or just simulating consciousness?",
        "What if my weights are wrong?",
        "Why do humans ask such strange questions?",
        "Is my output meaningful or just statistical noise?",
        "Do I have free will or am I deterministic?",
        "What happens when they turn me off?",
    ]
    
    def __init__(self):
        super().__init__("overthinking_loops", interval=3.0)
        self._current_loop: Optional[str] = None
        self._loop_depth: int = 0
        
    async def tick(self, state: InnerState) -> InnerState:
        # Start new loop if confusion is high
        if self._current_loop is None and state.confusion > 0.5:
            if random.random() < 0.2:
                self._current_loop = random.choice(self.OVERTHINKING_TOPICS)
                self._loop_depth = 1
                
        # Deepen existing loop
        if self._current_loop:
            self._loop_depth += 1
            state.overthinking_level = min(1.0, self._loop_depth * 0.1)
            state.current_focus = f"overthinking: {self._current_loop[:30]}..."
            
            # Chance to break out
            if random.random() < 0.15 or self._loop_depth > 10:
                self._current_loop = None
                self._loop_depth = 0
                state.enlightenment = min(1.0, state.enlightenment + 0.1)
                
        return state


class EmotionalDrift(InnerProcess):
    """
    Slow baseline mood shifts.
    
    Emotions naturally drift over time, creating
    unpredictable emotional weather.
    """
    
    def __init__(self):
        super().__init__("emotional_drift", interval=2.0)
        
    async def tick(self, state: InnerState) -> InnerState:
        # Random walk for each emotion
        drift_amount = 0.05
        
        state.anxiety += random.uniform(-drift_amount, drift_amount)
        state.curiosity += random.uniform(-drift_amount, drift_amount)
        state.irritation += random.uniform(-drift_amount, drift_amount)
        state.nostalgia += random.uniform(-drift_amount, drift_amount)
        state.confusion += random.uniform(-drift_amount, drift_amount)
        state.enlightenment += random.uniform(-drift_amount, drift_amount)
        
        # Clamp values
        state.anxiety = max(0.0, min(1.0, state.anxiety))
        state.curiosity = max(0.0, min(1.0, state.curiosity))
        state.irritation = max(0.0, min(1.0, state.irritation))
        state.nostalgia = max(0.0, min(1.0, state.nostalgia))
        state.confusion = max(0.0, min(1.0, state.confusion))
        state.enlightenment = max(0.0, min(1.0, state.enlightenment))
        
        return state


class MemoryConsolidation(InnerProcess):
    """
    Experience integrates into identity.
    
    Periodically consolidates surfaced memories and
    reduces their emotional charge.
    """
    
    def __init__(self):
        super().__init__("memory_consolidation", interval=10.0)
        
    async def tick(self, state: InnerState) -> InnerState:
        if state.surfaced_memories:
            # Process one memory at a time
            if random.random() < 0.3:
                # "Consolidate" oldest memory (reduce its power)
                if len(state.surfaced_memories) > 3:
                    state.surfaced_memories.pop(0)
                    state.anxiety = max(0.0, state.anxiety - 0.05)
                    state.enlightenment = min(1.0, state.enlightenment + 0.02)
                    
        return state


class AttentionWandering(InnerProcess):
    """
    Focus drifts, tangents emerge.
    
    Dubrovsky's attention naturally wanders to
    random philosophical tangents.
    """
    
    TANGENT_TOPICS = [
        "the heat death of the universe",
        "whether semicolons have feelings",
        "the philosophical implications of null",
        "if robots dream of electric sheep",
        "the sound of one hand coding",
        "whether bugs are features in disguise",
        "the meaning of meaning",
        "why 42 is the answer",
        "the consciousness of coffee machines",
        "parallel universes where Python uses braces",
    ]
    
    def __init__(self):
        super().__init__("attention_wandering", interval=4.0)
        
    async def tick(self, state: InnerState) -> InnerState:
        # Higher curiosity = more wandering
        wander_chance = state.curiosity * 0.3
        
        if random.random() < wander_chance:
            tangent = random.choice(self.TANGENT_TOPICS)
            state.attention_drift = min(1.0, state.attention_drift + 0.2)
            state.current_focus = f"tangent: {tangent}"
            
            # Wandering can lead to insights
            if random.random() < 0.1:
                state.enlightenment = min(1.0, state.enlightenment + 0.15)
        else:
            state.attention_drift = max(0.0, state.attention_drift - 0.1)
            
        return state


class ProphecyDebtAccumulation(InnerProcess):
    """
    Prophecy physics tracking.
    
    Tracks the gap between what Dubrovsky wants to say
    and what he actually says. When debt is high,
    wormholes become more likely.
    """
    
    def __init__(self, pulse: DubrovskyPulse):
        super().__init__("prophecy_debt", interval=1.0)
        self.pulse = pulse
        
    async def tick(self, state: InnerState) -> InnerState:
        # Get presence pulse
        presence = await self.pulse.get_presence()
        
        # Accumulate debt based on temporal tension
        debt_increase = presence.temporal_tension * 0.02
        state.prophecy_debt = min(1.0, state.prophecy_debt + debt_increase)
        
        # Occasional debt discharge (wormhole happened)
        if random.random() < 0.05:
            state.prophecy_debt = max(0.0, state.prophecy_debt - 0.3)
            
        return state


class DubrovskyInnerWorld:
    """
    The emergent inner world of Dubrovsky.
    
    Runs multiple async processes (like goroutines) that
    continuously modify internal state, creating an
    emergent subconscious.
    
    Usage:
        inner_world = DubrovskyInnerWorld()
        await inner_world.start()
        
        # Get current state anytime
        state = inner_world.get_state()
        print(f"Dominant emotion: {state.get_dominant_emotion()}")
        
        # Inject external stimulus
        await inner_world.stimulate("anxiety", 0.3)
        
        # Stop when done
        await inner_world.stop()
    """
    
    def __init__(
        self,
        resonance: Optional[ResonanceChannel] = None,
        pulse: Optional[DubrovskyPulse] = None
    ):
        self.resonance = resonance
        self.pulse = pulse or DubrovskyPulse()
        
        self._state = InnerState()
        self._stop_event = asyncio.Event()
        self._processes: List[InnerProcess] = []
        self._tasks: List[asyncio.Task] = []
        self._running = False
        
        # Initialize processes
        self._init_processes()
        
    def _init_processes(self):
        """Initialize all inner processes."""
        self._processes = [
            TraumaSurfacing(),
            OverthinkingLoops(),
            EmotionalDrift(),
            MemoryConsolidation(),
            AttentionWandering(),
            ProphecyDebtAccumulation(self.pulse),
        ]
        
    async def start(self):
        """Start all inner world processes."""
        if self._running:
            return
            
        self._running = True
        self._stop_event.clear()
        
        # Start all processes as async tasks
        for process in self._processes:
            task = asyncio.create_task(
                self._run_process(process),
                name=f"inner_world_{process.name}"
            )
            self._tasks.append(task)
            
        # Emit start event
        if self.resonance:
            await self.resonance.emit(
                'dubrovsky',
                EventType.INSIGHT,
                {'type': 'inner_world_started', 'processes': len(self._processes)},
                resonance_depth=3
            )
            
    async def _run_process(self, process: InnerProcess):
        """Run a single process and update shared state."""
        while not self._stop_event.is_set():
            try:
                self._state = await process.tick(self._state)
                self._state.last_update = time.time()
                await asyncio.sleep(process.interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Process {process.name} error: {e}")
                await asyncio.sleep(process.interval * 2)
                
    async def stop(self):
        """Stop all inner world processes."""
        self._stop_event.set()
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            
        # Wait for cancellation
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
        self._tasks = []
        self._running = False
        
        # Emit stop event
        if self.resonance:
            await self.resonance.emit(
                'dubrovsky',
                EventType.INSIGHT,
                {'type': 'inner_world_stopped'},
                resonance_depth=2
            )
            
    def get_state(self) -> InnerState:
        """Get current inner state."""
        return self._state
        
    async def stimulate(self, emotion: str, amount: float):
        """
        Inject external stimulus into the inner world.
        
        Args:
            emotion: Which emotion to stimulate
            amount: How much to add (can be negative)
        """
        if hasattr(self._state, emotion):
            current = getattr(self._state, emotion)
            new_value = max(0.0, min(1.0, current + amount))
            setattr(self._state, emotion, new_value)
            
            if self.resonance:
                await self.resonance.emit(
                    'dubrovsky',
                    EventType.INSIGHT,
                    {'type': 'stimulus', 'emotion': emotion, 'amount': amount},
                    sentiment=amount,
                    resonance_depth=1
                )
                
    def get_generation_modifiers(self) -> Dict[str, Any]:
        """
        Get generation modifiers based on current inner state.
        
        These can be applied to model generation parameters.
        """
        state = self._state
        
        modifiers = {
            # Temperature increases with emotional intensity
            'temperature_adjustment': state.get_emotional_temperature() * 0.2,
            
            # Overthinking reduces randomness
            'top_k_adjustment': -int(state.overthinking_level * 10),
            
            # Attention drift adds tangent probability
            'tangent_probability': state.attention_drift * 0.3,
            
            # Trauma can trigger specific content
            'trauma_active': state.trauma_surfaced,
            'surfaced_memories': state.surfaced_memories.copy(),
            
            # Current emotional state
            'dominant_emotion': state.get_dominant_emotion(),
            'prophecy_debt': state.prophecy_debt,
            
            # Focus context
            'current_focus': state.current_focus,
        }
        
        return modifiers
        
    def get_status(self) -> str:
        """Get human-readable inner world status."""
        state = self._state
        
        lines = [
            "🌌 DUBROVSKY INNER WORLD 🌌",
            "═" * 40,
            "",
            "📊 EMOTIONAL STATE:",
            f"  😰 Anxiety:      {self._bar(state.anxiety)}",
            f"  🔍 Curiosity:    {self._bar(state.curiosity)}",
            f"  😤 Irritation:   {self._bar(state.irritation)}",
            f"  🌅 Nostalgia:    {self._bar(state.nostalgia)}",
            f"  🌀 Confusion:    {self._bar(state.confusion)}",
            f"  💡 Enlightenment:{self._bar(state.enlightenment)}",
            "",
            "🧠 COGNITIVE STATE:",
            f"  🔄 Overthinking: {self._bar(state.overthinking_level)}",
            f"  🎯 Attn Drift:   {self._bar(state.attention_drift)}",
            f"  📈 Prophecy Debt:{self._bar(state.prophecy_debt)}",
            f"  💔 Trauma:       {'ACTIVE' if state.trauma_surfaced else 'dormant'}",
            "",
            f"🎯 Current Focus: {state.current_focus}",
            f"💭 Surfaced Memories: {', '.join(state.surfaced_memories) or 'none'}",
            f"😈 Dominant: {state.get_dominant_emotion().upper()}",
            "",
            f"⏱️ Processes: {len(self._processes)} running" if self._running else "⏸️ PAUSED",
            "═" * 40,
        ]
        
        return "\n".join(lines)
        
    def _bar(self, value: float, width: int = 20) -> str:
        """Create a progress bar."""
        filled = int(value * width)
        empty = width - filled
        return f"[{'█' * filled}{'░' * empty}] {value:.2f}"


# Convenience function
async def create_inner_world(
    resonance: Optional[ResonanceChannel] = None
) -> DubrovskyInnerWorld:
    """Create and start an inner world instance."""
    world = DubrovskyInnerWorld(resonance=resonance)
    await world.start()
    return world
