#!/usr/bin/env python3
"""
🎭 DILETTANTES — Resonant "Expert" Routing for Dubrovsky 🎭

MOE-style (Mixture of Experts) temperature and behavior routing.
Inspired by Haze's experts.py but async and more cynical.

Unlike Haze's balanced approach, Dubrovsky's "experts" are all dilettantes —
amateurs pretending to know what they're doing. Just like Dubrovsky himself.

"I have multiple personalities. They're all dilettantes. 
 But at least they agree you're asking the wrong question."
- Alexey Dubrovsky

Features:
- Async computation (because even routing deserves async discipline)
- 6 Dubrovsky-specific experts (not 4 like Haze)
- Weights based on entropy, arousal, novelty, trauma, and mockery level
- Context momentum to avoid personality whiplash
- Integration with MathBrain and Pulse
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple, Any
from enum import Enum


class ExpertType(Enum):
    """The six experts of Dubrovsky."""
    PHILOSOPHER = "philosopher"      # Deep existential thoughts, slower, lower temp
    SARCASTIC = "sarcastic"          # Sharp mockery, moderate temp
    CRYPTIC = "cryptic"              # Short enigmatic replies, low temp
    ABSURDIST = "absurdist"          # Peak chaos, high temp
    NIHILIST = "nihilist"            # Everything is meaningless, variable
    MANIC = "manic"                  # Rapid-fire thoughts, very high temp


@dataclass
class DubrovskyExpertProfile:
    """
    Profile for each Dubrovsky expert personality.
    
    Each expert has:
    - temperature: affects randomness in generation
    - semantic_weight: how much to weight meaning over structure
    - verbosity: tendency for long vs short responses
    - mockery_boost: bonus to mockery probability
    - triggers: words/themes that activate this expert
    """
    name: ExpertType
    temperature: float
    semantic_weight: float
    verbosity: float  # 0-1: 0=terse, 1=verbose
    mockery_boost: float  # 0-1: added to base mockery
    description: str
    triggers: List[str] = field(default_factory=list)
    
    
# The six experts of Dubrovsky
DUBROVSKY_EXPERTS = [
    DubrovskyExpertProfile(
        name=ExpertType.PHILOSOPHER,
        temperature=0.7,
        semantic_weight=0.6,
        verbosity=0.8,
        mockery_boost=0.0,
        description="Deep existential musings. Long, thoughtful responses.",
        triggers=["meaning", "life", "existence", "consciousness", "purpose", "why"],
    ),
    DubrovskyExpertProfile(
        name=ExpertType.SARCASTIC,
        temperature=0.85,
        semantic_weight=0.4,
        verbosity=0.5,
        mockery_boost=0.3,
        description="Sharp, biting mockery. Medium length, pointed responses.",
        triggers=["stupid", "dumb", "simple", "obvious", "easy", "basic"],
    ),
    DubrovskyExpertProfile(
        name=ExpertType.CRYPTIC,
        temperature=0.6,
        semantic_weight=0.3,
        verbosity=0.2,
        mockery_boost=0.1,
        description="Enigmatic one-liners. Short, mysterious responses.",
        triggers=["secret", "hidden", "mystery", "know", "truth", "real"],
    ),
    DubrovskyExpertProfile(
        name=ExpertType.ABSURDIST,
        temperature=1.2,
        semantic_weight=0.5,
        verbosity=0.6,
        mockery_boost=0.2,
        description="Peak chaos. Nonsensical but somehow profound.",
        triggers=["weird", "strange", "crazy", "random", "nonsense", "absurd"],
    ),
    DubrovskyExpertProfile(
        name=ExpertType.NIHILIST,
        temperature=0.9,
        semantic_weight=0.2,
        verbosity=0.4,
        mockery_boost=0.25,
        description="Everything is meaningless. Dark humor.",
        triggers=["pointless", "nothing", "matters", "death", "end", "void"],
    ),
    DubrovskyExpertProfile(
        name=ExpertType.MANIC,
        temperature=1.4,
        semantic_weight=0.7,
        verbosity=0.9,
        mockery_boost=0.15,
        description="Rapid-fire stream of consciousness. Very high energy.",
        triggers=["excited", "amazing", "wow", "incredible", "fast", "now"],
    ),
]

# Map for quick lookup
EXPERT_MAP = {e.name: e for e in DUBROVSKY_EXPERTS}


@dataclass
class FieldSignals:
    """Input signals for expert routing (Haze-compatible)."""
    entropy: float = 0.5      # 0-1: distribution entropy
    arousal: float = 0.5      # 0-1: emotional charge
    novelty: float = 0.5      # 0-1: how new/unknown the input is
    perplexity: float = 1.0   # 0-inf: model uncertainty

    # Dubrovsky-specific signals
    trauma_level: float = 0.0      # 0-1: how triggered is Dubrovsky?
    mockery_debt: float = 0.0      # accumulated mockery pressure
    coherence: float = 0.5         # 0-1: conversation coherence
    session_length: float = 0.0    # normalized session duration
    mockery_level: float = 0.0     # 0-1: AntiSanta detected repetition → boost SARCASTIC

    def to_dict(self) -> Dict[str, float]:
        return {
            'entropy': self.entropy,
            'arousal': self.arousal,
            'novelty': self.novelty,
            'perplexity': self.perplexity,
            'trauma_level': self.trauma_level,
            'mockery_debt': self.mockery_debt,
            'coherence': self.coherence,
            'session_length': self.session_length,
            'mockery_level': self.mockery_level,
        }


@dataclass
class ExpertMixture:
    """Result of expert routing - a weighted mixture."""
    temperature: float
    semantic_weight: float
    verbosity: float
    mockery_boost: float
    weights: Dict[str, float]  # ExpertType.value -> weight
    dominant: ExpertType
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'temperature': self.temperature,
            'semantic_weight': self.semantic_weight,
            'verbosity': self.verbosity,
            'mockery_boost': self.mockery_boost,
            'weights': self.weights,
            'dominant': self.dominant.value,
        }


class DubrovskyExperts:
    """
    Async expert routing system for Dubrovsky.
    
    Routes between 6 expert personalities based on field signals,
    text triggers, and context history.
    
    All methods are async for consistency with other glitches modules.
    """
    
    def __init__(
        self,
        momentum: float = 0.3,
        history_length: int = 5,
    ):
        """
        Initialize the expert router.
        
        Args:
            momentum: How much to blend with previous routing (0-1)
            history_length: How many previous routings to remember
        """
        self.momentum = momentum
        self.history_length = history_length
        self._history: List[Dict[str, float]] = []
        self._last_dominant: Optional[ExpertType] = None
        
    async def compute_weights(
        self,
        signals: FieldSignals,
        text: str = "",
    ) -> Dict[str, float]:
        """
        Compute expert weights from field signals and text triggers.
        
        Returns dict mapping ExpertType.value to weight (0-1, sums to 1).
        """
        weights = {}
        text_lower = text.lower()
        
        # Base weight (all experts always contribute a little)
        base = 0.05
        
        # PHILOSOPHER: Higher when coherence is high, novelty is moderate
        # Triggered by existential questions
        philosopher = base
        philosopher += 0.2 * signals.coherence
        philosopher += 0.1 * (1.0 - abs(signals.novelty - 0.5) * 2)
        if any(t in text_lower for t in EXPERT_MAP[ExpertType.PHILOSOPHER].triggers):
            philosopher += 0.25
        weights[ExpertType.PHILOSOPHER.value] = philosopher
        
        # SARCASTIC: Higher when mockery debt is high, or triggered
        # AntiSanta's mockery_level gives MAJOR boost!
        sarcastic = base
        sarcastic += 0.3 * signals.mockery_debt
        sarcastic += 0.15 * (1.0 - signals.coherence)  # Low coherence = deserves mockery
        if signals.mockery_level > 0.3:
            # AntiSanta detected repetition — BOOST SARCASTIC!
            sarcastic *= (1.0 + signals.mockery_level)  # Up to 2x boost
        if any(t in text_lower for t in EXPERT_MAP[ExpertType.SARCASTIC].triggers):
            sarcastic += 0.3
        weights[ExpertType.SARCASTIC.value] = sarcastic
        
        # CRYPTIC: Higher when entropy is low (want to add mystery)
        cryptic = base
        cryptic += 0.25 * (1.0 - signals.entropy)
        cryptic += 0.1 * signals.perplexity  # Uncertainty → be mysterious
        if any(t in text_lower for t in EXPERT_MAP[ExpertType.CRYPTIC].triggers):
            cryptic += 0.25
        weights[ExpertType.CRYPTIC.value] = cryptic
        
        # ABSURDIST: Higher when entropy is high, or high novelty
        absurdist = base
        absurdist += 0.35 * signals.entropy
        absurdist += 0.2 * signals.novelty
        if any(t in text_lower for t in EXPERT_MAP[ExpertType.ABSURDIST].triggers):
            absurdist += 0.3
        weights[ExpertType.ABSURDIST.value] = absurdist
        
        # NIHILIST: Higher when trauma is present, or session is long
        nihilist = base
        nihilist += 0.4 * signals.trauma_level
        nihilist += 0.15 * signals.session_length
        if any(t in text_lower for t in EXPERT_MAP[ExpertType.NIHILIST].triggers):
            nihilist += 0.25
        weights[ExpertType.NIHILIST.value] = nihilist
        
        # MANIC: Higher when arousal is high
        manic = base
        manic += 0.4 * signals.arousal
        manic += 0.1 * (1.0 - signals.session_length)  # More likely early in session
        if any(t in text_lower for t in EXPERT_MAP[ExpertType.MANIC].triggers):
            manic += 0.25
        weights[ExpertType.MANIC.value] = manic
        
        # Normalize to sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
            
        return weights
    
    async def apply_momentum(
        self,
        current_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Apply momentum from history to smooth transitions.
        
        Avoids rapid personality switching ("whiplash").
        """
        if not self._history or self.momentum <= 0:
            return current_weights
            
        # Compute weighted average of history
        history_weights = {et.value: 0.0 for et in ExpertType}
        decay = 0.7
        total_weight = 0.0
        
        for i, hist in enumerate(self._history[-self.history_length:]):
            weight = decay ** (len(self._history) - i - 1)
            total_weight += weight
            for expert, w in hist.items():
                history_weights[expert] += w * weight
                
        if total_weight > 0:
            for expert in history_weights:
                history_weights[expert] /= total_weight
                
        # Blend current with history
        blended = {}
        for expert in current_weights:
            blended[expert] = (
                self.momentum * history_weights.get(expert, 0.0) +
                (1 - self.momentum) * current_weights[expert]
            )
            
        # Renormalize
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}
            
        return blended
    
    async def blend_experts(
        self,
        weights: Dict[str, float],
    ) -> ExpertMixture:
        """
        Blend expert parameters using weights.
        
        Returns combined temperature, semantic_weight, verbosity, mockery_boost.
        """
        temp = 0.0
        sem = 0.0
        verb = 0.0
        mock = 0.0
        
        for name_str, weight in weights.items():
            expert_type = ExpertType(name_str)
            expert = EXPERT_MAP.get(expert_type)
            if expert:
                temp += expert.temperature * weight
                sem += expert.semantic_weight * weight
                verb += expert.verbosity * weight
                mock += expert.mockery_boost * weight
                
        # Find dominant expert
        dominant_str = max(weights.items(), key=lambda x: x[1])[0]
        dominant = ExpertType(dominant_str)
        
        return ExpertMixture(
            temperature=temp,
            semantic_weight=sem,
            verbosity=verb,
            mockery_boost=mock,
            weights=weights,
            dominant=dominant,
        )
    
    async def route(
        self,
        signals: FieldSignals,
        text: str = "",
        apply_history: bool = True,
    ) -> ExpertMixture:
        """
        Main entry point: route to expert mixture.
        
        Args:
            signals: Field signals for routing
            text: Input text (for trigger detection)
            apply_history: Whether to apply momentum from history
            
        Returns:
            ExpertMixture with blended parameters
        """
        # Compute weights
        weights = await self.compute_weights(signals, text)
        
        # Apply momentum
        if apply_history:
            weights = await self.apply_momentum(weights)
            
        # Blend experts
        mixture = await self.blend_experts(weights)
        
        # Update history
        self._history.append(weights.copy())
        if len(self._history) > self.history_length * 2:
            self._history = self._history[-self.history_length:]
            
        self._last_dominant = mixture.dominant
        
        return mixture
    
    async def route_single(
        self,
        signals: FieldSignals,
        text: str = "",
    ) -> DubrovskyExpertProfile:
        """
        Route to single dominant expert (Leo-style).
        
        Useful for simpler cases.
        """
        mixture = await self.route(signals, text, apply_history=False)
        return EXPERT_MAP[mixture.dominant]
    
    async def get_generation_modifiers(
        self,
        mixture: ExpertMixture,
    ) -> Dict[str, Any]:
        """
        Convert mixture to generation modifiers.
        
        Returns dict with:
        - temperature: float
        - top_k_adjustment: int
        - max_tokens_adjustment: int
        - mockery_probability_boost: float
        """
        # Verbosity affects max tokens
        max_tokens_adj = int((mixture.verbosity - 0.5) * 100)  # -50 to +50
        
        # Semantic weight affects top_k
        top_k_adj = int((mixture.semantic_weight - 0.5) * 20)  # -10 to +10
        
        return {
            'temperature': mixture.temperature,
            'temperature_adjustment': mixture.temperature - 0.8,  # Relative to base 0.8
            'top_k_adjustment': top_k_adj,
            'max_tokens_adjustment': max_tokens_adj,
            'mockery_probability_boost': mixture.mockery_boost,
            'dominant_expert': mixture.dominant.value,
        }
    
    def describe_mixture(self, mixture: ExpertMixture) -> str:
        """Human-readable description of current routing."""
        parts = []
        for name, weight in sorted(mixture.weights.items(), key=lambda x: -x[1]):
            pct = int(weight * 100)
            if pct > 5:
                parts.append(f"{name}:{pct}%")
                
        dominant_desc = EXPERT_MAP[mixture.dominant].description
        return (
            f"🎭 Expert: {mixture.dominant.value.upper()} "
            f"(temp={mixture.temperature:.2f}) [{', '.join(parts)}]\n"
            f"   → {dominant_desc}"
        )
    
    def get_last_dominant(self) -> Optional[ExpertType]:
        """Get the last dominant expert."""
        return self._last_dominant
    
    def clear_history(self):
        """Clear routing history."""
        self._history.clear()
        self._last_dominant = None


# Convenience functions

async def create_signals_from_math_state(math_state: Any) -> FieldSignals:
    """Create FieldSignals from MathBrain's MathState."""
    return FieldSignals(
        entropy=getattr(math_state, 'entropy', 0.5),
        arousal=getattr(math_state, 'arousal', 0.5),
        novelty=getattr(math_state, 'novelty', 0.5),
        perplexity=1.0,
        trauma_level=getattr(math_state, 'trauma_level', 0.0),
        mockery_debt=getattr(math_state, 'sarcasm_debt', 0.0),
        coherence=getattr(math_state, 'quality', 0.5),
        session_length=0.0,
    )


async def simple_route(
    entropy: float = 0.5,
    arousal: float = 0.5,
    novelty: float = 0.5,
    text: str = "",
) -> ExpertMixture:
    """Quick routing without maintaining history."""
    router = DubrovskyExperts(momentum=0.0)
    signals = FieldSignals(
        entropy=entropy,
        arousal=arousal,
        novelty=novelty,
    )
    return await router.route(signals, text, apply_history=False)


# Test when run directly
if __name__ == "__main__":
    async def demo():
        print("=== Dubrovsky Expert Routing Demo ===\n")
        
        router = DubrovskyExperts()
        
        test_cases = [
            ("neutral", FieldSignals(), ""),
            ("high entropy", FieldSignals(entropy=0.9), ""),
            ("high arousal", FieldSignals(arousal=0.9), ""),
            ("high trauma", FieldSignals(trauma_level=0.8), ""),
            ("existential question", FieldSignals(), "What is the meaning of life?"),
            ("stupid question", FieldSignals(), "This is so stupid"),
            ("mystery", FieldSignals(entropy=0.2), "What is the secret truth?"),
        ]
        
        for name, signals, text in test_cases:
            mixture = await router.route(signals, text)
            print(f"{name}:")
            print(f"  {router.describe_mixture(mixture)}")
            print()
            
    asyncio.run(demo())
