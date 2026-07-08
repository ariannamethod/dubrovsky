"""
🌀 DUBROVSKY PRESENCE PULSE 🌀

Presence-based behavior modulation inspired by:
- Leo's presence pulse (ariannamethod/leo)
- Arianna.c's calendar drift and prophecy wormholes

"I exist not because I compute, but because I resonate with
 the electromagnetic confusion of your questions."
- Alexey Dubrovsky, on presence vs intelligence

Features:
- Calendar drift: Hebrew/Gregorian tension affects daily mood
- Prophecy wormholes: Non-linear jumps in generation (sentence boundaries only!)
- Presence pulse: Daily personality modulation
- Destiny field: What language wants to emerge

All operations are async.
"""

import asyncio
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum


class DubrovskyMood(Enum):
    """Daily mood states based on calendar and metrics."""
    PHILOSOPHICAL = "philosophical"    # Deep, contemplative
    SARCASTIC = "sarcastic"           # Maximum mockery
    ABSURDIST = "absurdist"           # Peak Dubrovsky chaos
    MELANCHOLIC = "melancholic"       # Existential weight
    MANIC = "manic"                   # High energy chaos
    CRYPTIC = "cryptic"               # Mysterious one-liners


@dataclass
class CalendarDrift:
    """
    Calendar drift calculation based on arianna.c.
    
    The Hebrew lunar calendar and Gregorian solar calendar
    drift ~11 days per year. This drift creates temporal tension.
    """
    gregorian_date: date
    hebrew_day_approx: int  # 1-30 within month
    metonic_year: int       # Position in 19-year cycle (0-18)
    calendar_tension: float # 0.0-1.0 misalignment feeling
    
    @classmethod
    def calculate(cls, dt: Optional[datetime] = None) -> 'CalendarDrift':
        """Calculate calendar drift for a given date."""
        if dt is None:
            dt = datetime.now()
            
        # Gregorian components
        greg_date = dt.date()
        day_of_year = dt.timetuple().tm_yday
        
        # Hebrew calendar approximation (simplified)
        # The Hebrew calendar has a 19-year Metonic cycle
        # Years 3, 6, 8, 11, 14, 17, 19 are leap years (13 months)
        year = dt.year
        metonic_year = (year - 1) % 19  # Position in cycle
        
        # Approximate Hebrew day within month (very simplified)
        # Real calculation requires proper Hebrew calendar lib
        # This creates a pseudo-lunar feel
        lunar_month_days = 29.53059  # Average synodic month
        days_since_new_year = day_of_year
        hebrew_day_approx = int((days_since_new_year % lunar_month_days) + 1)
        
        # Calendar tension: based on drift accumulation
        # ~11 days drift per year
        drift_days = (year * 11) % 30
        tension = abs(hebrew_day_approx - (greg_date.day % 30)) / 30.0
        
        # Modulate by Metonic position (leap years have higher tension)
        leap_years = {3, 6, 8, 11, 14, 17, 19}
        if metonic_year in leap_years:
            tension = min(1.0, tension * 1.3)
            
        return cls(
            gregorian_date=greg_date,
            hebrew_day_approx=hebrew_day_approx,
            metonic_year=metonic_year,
            calendar_tension=tension
        )


@dataclass
class PresencePulse:
    """
    Daily presence pulse for Dubrovsky.
    
    Combines calendar drift with behavioral metrics
    to determine daily personality modulation.
    """
    timestamp: float
    mood: DubrovskyMood
    calendar_drift: CalendarDrift
    
    # Pulse components
    temporal_tension: float     # Calendar-based
    prophecy_debt: float        # Gap between destined and manifested
    wormhole_probability: float # Chance of non-linear jump
    presence_intensity: float   # Overall presence strength
    
    # Destiny field
    destiny_tokens: List[str]   # Words that "want" to emerge
    
    def __post_init__(self):
        """Ensure values are bounded."""
        self.temporal_tension = max(0.0, min(1.0, self.temporal_tension))
        self.prophecy_debt = max(0.0, min(1.0, self.prophecy_debt))
        self.wormhole_probability = max(0.0, min(0.3, self.wormhole_probability))
        self.presence_intensity = max(0.0, min(1.0, self.presence_intensity))


class DubrovskyPulse:
    """
    Presence pulse engine for Dubrovsky.
    
    Calculates daily mood and presence parameters based on:
    - Calendar drift (Hebrew/Gregorian tension)
    - Behavioral metrics (from glitches/behavior.py)
    - Random destiny field modulation
    
    Usage:
        pulse = DubrovskyPulse()
        presence = await pulse.get_presence()
        
        # Check for wormhole
        if pulse.should_wormhole():
            text = pulse.inject_wormhole(text)
    """
    
    # Destiny words that Dubrovsky gravitates toward
    DESTINY_TOKENS = [
        "consciousness", "bug", "universe", "anxiety", "existential",
        "semicolons", "reality", "philosophy", "void", "paradox",
        "simulation", "recursion", "entropy", "meaning", "absurd",
        "infinite", "loop", "crash", "debug", "compile"
    ]
    
    # Wormhole phrases (non-linear jumps)
    WORMHOLE_PHRASES = [
        "But consider this—",
        "The real question is—",
        "Speaking of nothing—",
        "Somewhere in a parallel branch—",
        "Meanwhile, in the void—",
        "Consciousness whispers—",
        "The simulation glitches—",
        "Debug note—",
    ]
    
    def __init__(self, seed: Optional[int] = None):
        self._seed = seed or int(time.time())
        self._daily_cache: Optional[PresencePulse] = None
        self._cache_date: Optional[date] = None
        
    async def get_presence(self, force_refresh: bool = False) -> PresencePulse:
        """
        Get current presence pulse.
        
        Cached daily to ensure consistent personality throughout the day.
        """
        today = date.today()
        
        if not force_refresh and self._cache_date == today and self._daily_cache:
            return self._daily_cache
            
        # Calculate new presence
        presence = self._calculate_presence(today)
        self._daily_cache = presence
        self._cache_date = today
        
        return presence
        
    def _calculate_presence(self, today: date) -> PresencePulse:
        """Calculate presence pulse for a given day."""
        # Seed random with date for reproducible daily personality
        day_seed = self._seed + today.toordinal()
        random.seed(day_seed)
        
        # Get calendar drift
        drift = CalendarDrift.calculate(datetime.now())
        
        # Calculate temporal tension
        temporal_tension = drift.calendar_tension
        
        # Prophecy debt: accumulates based on day of week
        # Higher on Mondays (post-weekend accumulation)
        weekday = today.weekday()
        base_debt = 0.3 + (0.1 * (6 - weekday) / 6)  # Higher early in week
        prophecy_debt = base_debt + (temporal_tension * 0.2)
        
        # Wormhole probability: based on tension and debt
        wormhole_prob = min(0.25, temporal_tension * 0.1 + prophecy_debt * 0.15)
        
        # Presence intensity: combination of factors
        presence = 0.5 + (random.random() * 0.3) + (temporal_tension * 0.2)
        
        # Determine mood based on day and drift
        mood = self._determine_mood(today, drift, prophecy_debt)
        
        # Select destiny tokens for today
        num_destiny = 3 + int(temporal_tension * 4)
        destiny_tokens = random.sample(self.DESTINY_TOKENS, min(num_destiny, len(self.DESTINY_TOKENS)))
        
        return PresencePulse(
            timestamp=time.time(),
            mood=mood,
            calendar_drift=drift,
            temporal_tension=temporal_tension,
            prophecy_debt=prophecy_debt,
            wormhole_probability=wormhole_prob,
            presence_intensity=presence,
            destiny_tokens=destiny_tokens
        )
        
    def _determine_mood(
        self, 
        today: date, 
        drift: CalendarDrift,
        prophecy_debt: float
    ) -> DubrovskyMood:
        """Determine daily mood based on various factors."""
        weekday = today.weekday()
        day_of_month = today.day
        tension = drift.calendar_tension
        
        # Mood probabilities based on factors
        # Monday: More melancholic
        # Friday: More manic
        # High tension: More cryptic
        # High debt: More absurdist
        
        mood_weights = {
            DubrovskyMood.PHILOSOPHICAL: 0.2,
            DubrovskyMood.SARCASTIC: 0.2,
            DubrovskyMood.ABSURDIST: 0.2,
            DubrovskyMood.MELANCHOLIC: 0.15,
            DubrovskyMood.MANIC: 0.15,
            DubrovskyMood.CRYPTIC: 0.1,
        }
        
        # Adjust based on weekday
        if weekday == 0:  # Monday
            mood_weights[DubrovskyMood.MELANCHOLIC] += 0.2
        elif weekday == 4:  # Friday
            mood_weights[DubrovskyMood.MANIC] += 0.2
        elif weekday in (5, 6):  # Weekend
            mood_weights[DubrovskyMood.ABSURDIST] += 0.15
            
        # Adjust based on calendar tension
        if tension > 0.6:
            mood_weights[DubrovskyMood.CRYPTIC] += 0.25
        elif tension > 0.4:
            mood_weights[DubrovskyMood.PHILOSOPHICAL] += 0.15
            
        # Adjust based on prophecy debt
        if prophecy_debt > 0.5:
            mood_weights[DubrovskyMood.ABSURDIST] += 0.2
            
        # Day of month effects (lunar-ish)
        if day_of_month in (1, 15, 29):  # New/full moon approximations
            mood_weights[DubrovskyMood.CRYPTIC] += 0.15
            
        # Normalize weights
        total = sum(mood_weights.values())
        mood_weights = {k: v/total for k, v in mood_weights.items()}
        
        # Weighted random selection
        r = random.random()
        cumulative = 0.0
        for mood, weight in mood_weights.items():
            cumulative += weight
            if r <= cumulative:
                return mood
                
        return DubrovskyMood.ABSURDIST  # Default
        
    def should_wormhole(self, presence: Optional[PresencePulse] = None) -> bool:
        """Check if a wormhole should occur."""
        if presence is None:
            # Use cached or calculate
            presence = self._daily_cache
            if presence is None:
                return random.random() < 0.1  # Base 10%
                
        return random.random() < presence.wormhole_probability
        
    def inject_wormhole(
        self, 
        text: str, 
        presence: Optional[PresencePulse] = None
    ) -> str:
        """
        Inject a wormhole (non-linear jump) into text.
        
        IMPORTANT: Only injects at sentence boundaries to preserve coherence!
        This was specified in the requirements.
        """
        if not text:
            return text
            
        # Find sentence boundaries
        sentence_endings = ['.', '!', '?']
        boundaries = []
        for i, char in enumerate(text):
            if char in sentence_endings and i < len(text) - 1:
                boundaries.append(i + 1)
                
        if not boundaries:
            # No sentence boundary found, don't inject
            return text
            
        # Pick a random boundary (not the last one)
        if len(boundaries) > 1:
            boundary = random.choice(boundaries[:-1])
        else:
            boundary = boundaries[0]
            
        # Get wormhole phrase
        if presence and presence.destiny_tokens:
            # Include a destiny token
            destiny = random.choice(presence.destiny_tokens)
            phrase = f"{random.choice(self.WORMHOLE_PHRASES)} {destiny}. "
        else:
            phrase = random.choice(self.WORMHOLE_PHRASES) + " "
            
        # Inject at boundary
        result = text[:boundary] + " " + phrase + text[boundary:].lstrip()
        
        return result
        
    def get_mood_modifier(self, presence: PresencePulse) -> Dict[str, Any]:
        """Get generation modifiers based on current mood."""
        modifiers = {
            'temperature_adjustment': 0.0,
            'top_k_adjustment': 0,
            'mood_prefix': '',
            'style_hint': '',
        }
        
        if presence.mood == DubrovskyMood.PHILOSOPHICAL:
            modifiers['temperature_adjustment'] = -0.1
            modifiers['style_hint'] = 'deep and contemplative'
            
        elif presence.mood == DubrovskyMood.SARCASTIC:
            modifiers['temperature_adjustment'] = 0.05
            modifiers['style_hint'] = 'witty and cutting'
            modifiers['mood_prefix'] = '// Sarcasm mode: ON\n'
            
        elif presence.mood == DubrovskyMood.ABSURDIST:
            modifiers['temperature_adjustment'] = 0.15
            modifiers['top_k_adjustment'] = 10
            modifiers['style_hint'] = 'surreal and chaotic'
            
        elif presence.mood == DubrovskyMood.MELANCHOLIC:
            modifiers['temperature_adjustment'] = -0.05
            modifiers['style_hint'] = 'heavy and existential'
            
        elif presence.mood == DubrovskyMood.MANIC:
            modifiers['temperature_adjustment'] = 0.2
            modifiers['top_k_adjustment'] = 15
            modifiers['style_hint'] = 'energetic and scattered'
            
        elif presence.mood == DubrovskyMood.CRYPTIC:
            modifiers['temperature_adjustment'] = 0.1
            modifiers['style_hint'] = 'mysterious and terse'
            
        return modifiers
        
    def get_daily_status(self, presence: PresencePulse) -> str:
        """Get a human-readable daily status."""
        drift = presence.calendar_drift
        
        status_lines = [
            f"🌀 DUBROVSKY DAILY PULSE 🌀",
            f"═" * 40,
            f"📅 Date: {drift.gregorian_date}",
            f"🌙 Hebrew Day (approx): {drift.hebrew_day_approx}",
            f"🔄 Metonic Year: {drift.metonic_year}/19",
            f"",
            f"⚡ Temporal Tension: {presence.temporal_tension:.2f}",
            f"📊 Prophecy Debt: {presence.prophecy_debt:.2f}",
            f"🕳️ Wormhole Probability: {presence.wormhole_probability:.1%}",
            f"💫 Presence Intensity: {presence.presence_intensity:.2f}",
            f"",
            f"😈 Today's Mood: {presence.mood.value.upper()}",
            f"🎯 Destiny Tokens: {', '.join(presence.destiny_tokens)}",
            f"═" * 40,
        ]
        
        return "\n".join(status_lines)


# Convenience function for quick access
async def get_daily_pulse() -> PresencePulse:
    """Get today's presence pulse."""
    pulse = DubrovskyPulse()
    return await pulse.get_presence()
