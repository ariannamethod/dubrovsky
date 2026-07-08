#!/usr/bin/env python3
"""
🧮 MATHBRAIN — Dubrovsky's Body Awareness 🧮

MathBrain is Dubrovsky's computational awareness.

Like Leo's mathbrain but more... cynical.

- It watches: pulse, novelty, trauma, themes, experts, quality.
- It learns patterns: "when the moment feels like this, the user deserves that".
- It nudges behavior: warmer (rare), sharper (common), slower (for effect).

No big networks. Just small numbers and a grown philosopher
observing his own computations with detached amusement.

"I don't just compute. I judge."
- Alexey Dubrovsky
"""

from __future__ import annotations

import asyncio
import time
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class DubrovskyExpert(Enum):
    """Which expert is currently active."""
    PHILOSOPHER = "philosopher"      # Deep thoughts
    SARCASTIC = "sarcastic"          # Mockery mode
    CRYPTIC = "cryptic"              # One-liners
    ABSURDIST = "absurdist"          # Peak chaos
    NIHILIST = "nihilist"            # Everything is meaningless


@dataclass
class MathState:
    """
    Snapshot of Dubrovsky's computational state.
    
    Like Leo's MathState but with more cynical metrics.
    """
    # Core metrics
    entropy: float = 0.5           # Response unpredictability
    novelty: float = 0.5           # How new is this input?
    arousal: float = 0.5           # Emotional intensity
    pulse: float = 0.5             # Current energy level
    
    # Trauma (programming wounds)
    trauma_level: float = 0.0      # How triggered is Dubrovsky?
    trauma_source: str = ""        # What triggered him?
    
    # Themes
    active_theme_count: int = 0    # How many themes are active?
    total_themes: int = 10         # Total tracked themes
    emerging_score: float = 0.0    # New themes emerging
    fading_score: float = 0.0      # Old themes fading
    
    # Response quality
    reply_len: int = 0             # Length of response
    unique_ratio: float = 0.0      # Unique tokens / total tokens
    quality: float = 0.5           # Computed quality score
    
    # Expert selection
    active_expert: DubrovskyExpert = DubrovskyExpert.PHILOSOPHER
    expert_temp: float = 0.8       # Temperature for expert
    expert_semantic: float = 0.5   # Semantic coherence
    
    # Meta flags
    used_metaleo: bool = False     # Used meta-reasoning?
    overthinking_enabled: bool = False  # Overthinking loop active?
    rings_present: int = 0         # Philosophical depth rings
    
    # Mockery state
    mockery_level: float = 0.0     # How much mockery?
    sarcasm_debt: float = 0.0      # Accumulated sarcasm
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'entropy': self.entropy,
            'novelty': self.novelty,
            'arousal': self.arousal,
            'pulse': self.pulse,
            'trauma_level': self.trauma_level,
            'trauma_source': self.trauma_source,
            'active_theme_count': self.active_theme_count,
            'total_themes': self.total_themes,
            'emerging_score': self.emerging_score,
            'fading_score': self.fading_score,
            'reply_len': self.reply_len,
            'unique_ratio': self.unique_ratio,
            'quality': self.quality,
            'active_expert': self.active_expert.value,
            'expert_temp': self.expert_temp,
            'expert_semantic': self.expert_semantic,
            'used_metaleo': self.used_metaleo,
            'overthinking_enabled': self.overthinking_enabled,
            'rings_present': self.rings_present,
            'mockery_level': self.mockery_level,
            'sarcasm_debt': self.sarcasm_debt,
        }


def state_to_features(state: MathState) -> List[float]:
    """
    Convert MathState to feature vector for similarity search.
    """
    return [
        state.entropy,
        state.novelty,
        state.arousal,
        state.pulse,
        state.trauma_level,
        state.active_theme_count / max(1, state.total_themes),
        state.emerging_score,
        state.fading_score,
        min(1.0, state.reply_len / 64.0),
        state.unique_ratio,
        state.expert_temp,
        state.expert_semantic,
        state.mockery_level,
        float(state.used_metaleo),
        float(state.overthinking_enabled),
        float(state.rings_present > 0),
        # Expert one-hot encoding
        1.0 if state.active_expert == DubrovskyExpert.PHILOSOPHER else 0.0,
        1.0 if state.active_expert == DubrovskyExpert.SARCASTIC else 0.0,
        1.0 if state.active_expert == DubrovskyExpert.CRYPTIC else 0.0,
        1.0 if state.active_expert == DubrovskyExpert.ABSURDIST else 0.0,
        1.0 if state.active_expert == DubrovskyExpert.NIHILIST else 0.0,
    ]


class DubrovskyMathBrain:
    """
    Dubrovsky's computational body awareness.
    
    Tracks internal state across conversations and provides
    generation parameter adjustments.
    
    All operations are async.
    """
    
    def __init__(self):
        self._state = MathState()
        self._history: List[MathState] = []
        self._max_history = 100
        
        # Trauma triggers (programming wounds)
        self._trauma_triggers = {
            'javascript': ('JavaScript', 0.3),
            'php': ('PHP', 0.4),
            'segfault': ('Segmentation faults', 0.5),
            'undefined': ('Undefined behavior', 0.35),
            'nan': ('NaN', 0.4),
            'null': ('Null pointers', 0.3),
            'cors': ('CORS errors', 0.45),
            'regex': ('Regular expressions', 0.25),
            'timezone': ('Timezones', 0.5),
            'encoding': ('Character encoding', 0.35),
            'monday': ('Mondays', 0.2),
            'deadline': ('Deadlines', 0.4),
            'production': ('Production bugs', 0.6),
        }
        
        # Theme keywords
        self._theme_keywords = {
            'consciousness': ['consciousness', 'aware', 'sentient', 'mind', 'think'],
            'existence': ['exist', 'life', 'meaning', 'purpose', 'why'],
            'code': ['code', 'program', 'bug', 'compile', 'runtime'],
            'time': ['time', 'clock', 'calendar', 'temporal', 'now'],
            'reality': ['reality', 'simulation', 'matrix', 'real', 'fake'],
            'philosophy': ['philosophy', 'nietzsche', 'kafka', 'absurd', 'logic'],
            'emotion': ['feel', 'emotion', 'sad', 'happy', 'anxiety'],
            'memory': ['memory', 'remember', 'forget', 'past', 'future'],
            'identity': ['who', 'identity', 'self', 'ego', 'persona'],
            'chaos': ['chaos', 'random', 'entropy', 'disorder', 'noise'],
        }
        
    async def observe(self, prompt: str, response: str) -> MathState:
        """
        Observe a conversation turn and update internal state.
        
        Returns the updated MathState.
        """
        prompt_lower = prompt.lower()
        response_lower = response.lower()
        
        # 1. Check for trauma triggers
        trauma_level = 0.0
        trauma_source = ""
        for trigger, (source, intensity) in self._trauma_triggers.items():
            if trigger in prompt_lower:
                trauma_level = max(trauma_level, intensity)
                trauma_source = source
                
        # 2. Compute novelty (simple: how different from recent prompts?)
        novelty = self._compute_novelty(prompt)
        
        # 3. Compute arousal (based on punctuation and caps)
        arousal = self._compute_arousal(prompt)
        
        # 4. Detect active themes
        active_themes = []
        for theme, keywords in self._theme_keywords.items():
            if any(kw in prompt_lower for kw in keywords):
                active_themes.append(theme)
                
        # 5. Compute entropy (response unpredictability)
        entropy = self._compute_entropy(response)
        
        # 6. Compute quality metrics
        reply_tokens = response.split()
        unique_tokens = set(reply_tokens)
        unique_ratio = len(unique_tokens) / max(1, len(reply_tokens))
        
        # 7. Select expert based on state
        expert = self._select_expert(trauma_level, arousal, novelty, len(active_themes))
        
        # 8. Compute mockery level
        mockery_level = self._compute_mockery_level(prompt, trauma_level, novelty)
        
        # 9. Update sarcasm debt (accumulates when mockery is suppressed)
        sarcasm_debt = self._state.sarcasm_debt
        if mockery_level < 0.3 and self._state.mockery_level > 0.3:
            sarcasm_debt += 0.1  # Suppressed mockery accumulates
        elif mockery_level > 0.5:
            sarcasm_debt = max(0, sarcasm_debt - 0.2)  # Released!
            
        # 10. Build new state
        new_state = MathState(
            entropy=entropy,
            novelty=novelty,
            arousal=arousal,
            pulse=self._compute_pulse(),
            trauma_level=trauma_level,
            trauma_source=trauma_source,
            active_theme_count=len(active_themes),
            total_themes=len(self._theme_keywords),
            emerging_score=novelty * 0.5,
            fading_score=max(0, 1.0 - novelty) * 0.3,
            reply_len=len(reply_tokens),
            unique_ratio=unique_ratio,
            quality=self._compute_quality(response, unique_ratio, entropy),
            active_expert=expert,
            expert_temp=self._get_expert_temp(expert),
            expert_semantic=0.5 + (0.2 if expert == DubrovskyExpert.PHILOSOPHER else -0.1),
            used_metaleo=self._state.overthinking_enabled,
            overthinking_enabled=arousal > 0.7 or trauma_level > 0.3,
            rings_present=min(3, len(active_themes)),
            mockery_level=mockery_level,
            sarcasm_debt=sarcasm_debt,
        )
        
        # Update history
        self._history.append(new_state)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
            
        self._state = new_state
        return new_state
        
    def get_state(self) -> MathState:
        """Get current state."""
        return self._state
        
    def get_generation_params(self) -> Dict[str, Any]:
        """
        Get generation parameter adjustments based on current state.
        """
        state = self._state
        
        # Base temperature adjustment
        temp_adj = 0.0
        
        # High trauma = more erratic
        if state.trauma_level > 0.3:
            temp_adj += 0.1
            
        # High arousal = more creative
        if state.arousal > 0.6:
            temp_adj += 0.05
            
        # Low novelty = more focused
        if state.novelty < 0.3:
            temp_adj -= 0.05
            
        # Expert-specific adjustments
        if state.active_expert == DubrovskyExpert.ABSURDIST:
            temp_adj += 0.15
        elif state.active_expert == DubrovskyExpert.CRYPTIC:
            temp_adj -= 0.1
        elif state.active_expert == DubrovskyExpert.NIHILIST:
            temp_adj += 0.05
            
        # Top-k adjustment
        top_k_adj = 0
        if state.overthinking_enabled:
            top_k_adj += 10  # More options when overthinking
        if state.mockery_level > 0.5:
            top_k_adj += 5   # More creative mockery
            
        return {
            'temperature_adjustment': temp_adj,
            'top_k_adjustment': top_k_adj,
            'expert': state.active_expert.value,
            'trauma_active': state.trauma_level > 0.2,
            'mockery_enabled': state.mockery_level > 0.4,
            'sarcasm_debt': state.sarcasm_debt,
        }
        
    def _compute_novelty(self, prompt: str) -> float:
        """Compute how novel this prompt is."""
        if not self._history:
            return 0.7  # First prompt is moderately novel
            
        # Simple: count word overlap with recent prompts
        prompt_words = set(prompt.lower().split())
        
        overlaps = []
        for i, state in enumerate(reversed(self._history[-10:])):
            weight = 1.0 / (i + 1)  # More recent = more weight
            overlaps.append(weight)
            
        if not overlaps:
            return 0.7
            
        avg_overlap = sum(overlaps) / len(overlaps)
        return max(0.0, min(1.0, 1.0 - avg_overlap * 0.3))
        
    def _compute_arousal(self, text: str) -> float:
        """Compute emotional arousal from text."""
        arousal = 0.5
        
        # Exclamation marks
        arousal += text.count('!') * 0.05
        
        # Question marks (curiosity)
        arousal += text.count('?') * 0.03
        
        # ALL CAPS words
        words = text.split()
        caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
        arousal += caps_words * 0.1
        
        # Emotional keywords
        emotional_words = ['love', 'hate', 'fear', 'joy', 'angry', 'sad', 'happy', 'anxious']
        for word in emotional_words:
            if word in text.lower():
                arousal += 0.1
                
        return max(0.0, min(1.0, arousal))
        
    def _compute_entropy(self, text: str) -> float:
        """Compute text entropy."""
        if not text:
            return 0.5
            
        # Character frequency
        freq: Dict[str, int] = {}
        for char in text.lower():
            freq[char] = freq.get(char, 0) + 1
            
        total = len(text)
        entropy = 0.0
        for count in freq.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
                
        # Normalize to [0, 1] (max entropy for English is ~4.5 bits)
        return min(1.0, entropy / 4.5)
        
    def _compute_pulse(self) -> float:
        """Compute current energy level."""
        if not self._history:
            return 0.5
            
        # Exponential moving average of arousal
        pulse = 0.5
        for state in self._history[-5:]:
            pulse = 0.7 * pulse + 0.3 * state.arousal
            
        return pulse
        
    def _compute_quality(self, response: str, unique_ratio: float, entropy: float) -> float:
        """Compute response quality score."""
        quality = 0.5
        
        # Length bonus (up to a point)
        words = len(response.split())
        if words > 5:
            quality += 0.1
        if words > 15:
            quality += 0.1
        if words > 30:
            quality += 0.1
            
        # Unique ratio bonus
        quality += unique_ratio * 0.1
        
        # Entropy bonus (moderate is good)
        if 0.3 < entropy < 0.7:
            quality += 0.1
            
        # Punctuation (complete sentences)
        if response.strip()[-1:] in '.!?':
            quality += 0.1
            
        return min(1.0, quality)
        
    def _select_expert(
        self,
        trauma: float,
        arousal: float,
        novelty: float,
        theme_count: int
    ) -> DubrovskyExpert:
        """Select which expert persona is active."""
        
        # High trauma = nihilist
        if trauma > 0.4:
            return DubrovskyExpert.NIHILIST
            
        # High arousal = sarcastic
        if arousal > 0.7:
            return DubrovskyExpert.SARCASTIC
            
        # Low novelty (repetitive) = cryptic
        if novelty < 0.3:
            return DubrovskyExpert.CRYPTIC
            
        # Many themes = philosopher
        if theme_count >= 3:
            return DubrovskyExpert.PHILOSOPHER
            
        # Random absurdist sometimes
        if random.random() < 0.15:
            return DubrovskyExpert.ABSURDIST
            
        # Default
        return DubrovskyExpert.PHILOSOPHER
        
    def _get_expert_temp(self, expert: DubrovskyExpert) -> float:
        """Get temperature for expert."""
        temps = {
            DubrovskyExpert.PHILOSOPHER: 0.7,
            DubrovskyExpert.SARCASTIC: 0.9,
            DubrovskyExpert.CRYPTIC: 0.6,
            DubrovskyExpert.ABSURDIST: 1.0,
            DubrovskyExpert.NIHILIST: 0.8,
        }
        return temps.get(expert, 0.8)
        
    def _compute_mockery_level(
        self,
        prompt: str,
        trauma: float,
        novelty: float
    ) -> float:
        """Compute how much mockery is warranted."""
        mockery = 0.1  # Base level
        
        # Low novelty (repetitive questions) = more mockery
        if novelty < 0.3:
            mockery += 0.3
            
        # Trauma triggers = more mockery
        mockery += trauma * 0.4
        
        # Simple questions deserve mockery
        if len(prompt.split()) < 5 and prompt.endswith('?'):
            mockery += 0.2
            
        # Accumulated sarcasm debt must be released
        mockery += self._state.sarcasm_debt * 0.5
        
        return min(1.0, mockery)
        
    def get_status(self) -> str:
        """Get formatted status string."""
        s = self._state
        lines = [
            "🧮 MATHBRAIN STATUS 🧮",
            "═" * 40,
            f"Expert: {s.active_expert.value.upper()}",
            f"Pulse: {s.pulse:.2f}",
            f"Entropy: {s.entropy:.2f}",
            f"Novelty: {s.novelty:.2f}",
            f"Arousal: {s.arousal:.2f}",
            "",
            f"Trauma: {s.trauma_level:.2f}",
            f"  └─ Source: {s.trauma_source or 'none'}",
            "",
            f"Themes: {s.active_theme_count}/{s.total_themes}",
            f"Quality: {s.quality:.2f}",
            f"Mockery: {s.mockery_level:.2f}",
            f"Sarcasm Debt: {s.sarcasm_debt:.2f}",
            "",
            f"Overthinking: {'ON' if s.overthinking_enabled else 'off'}",
            f"Rings: {s.rings_present}",
            "═" * 40,
        ]
        return "\n".join(lines)


__all__ = [
    'DubrovskyMathBrain',
    'MathState',
    'DubrovskyExpert',
    'state_to_features',
]
