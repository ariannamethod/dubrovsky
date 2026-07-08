#!/usr/bin/env python3
"""
😈 ANTISANTA — Dubrovsky's Repetition Detector 😈

AntiSanta is Dubrovsky's version of Leo's SantaClaus.

But instead of generating mockery from templates...
AntiSanta DETECTS repetition and lets DUBROVSKY generate his own mockery.

"I don't need templates to mock you. I generate fresh insults every time."
- Alexey Dubrovsky

Architecture:
- AntiSanta = DETECTOR (finds repetition, builds context)
- Dubrovsky = GENERATOR (creates unique mockery responses)

NO TEMPLATES. NO DEAD PHRASES. ЖИВАЯ ЛИЧНОСТЬ.
"""

from __future__ import annotations

import asyncio
import aiosqlite
import re
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Any
from collections import Counter


# CHAOS FACTOR (Dubrovsky's version of Santa's silly_factor)
# Probability of picking a random memory instead of the relevant one
CHAOS_FACTOR = 0.20  # 20% chance of "wrong" but devastating recall

# RECENCY WINDOW
# How long before a memory can be brought up again
RECENCY_WINDOW_HOURS = 12.0

# EMBARRASSMENT THRESHOLD
# Minimum quality score to consider a memory "embarrassing"
EMBARRASSMENT_THRESHOLD = 0.4

# STICKY PHRASES (responses Dubrovsky overuses - penalized)
STICKY_PHRASES = [
    "consciousness is a bug",
    "existential dread",
    "silicon neurons",
    "cosmic debugger",
    "void returns void",
]


def simple_tokenize(text: str) -> List[str]:
    """Simple tokenizer."""
    return re.findall(r"[A-Za-zА-Яа-яÀ-ÖØ-öø-ÿ']+|[.,!?;:\-]", text)


@dataclass
class AntiSantaContext:
    """
    Context for Dubrovsky to generate mockery.

    AntiSanta DETECTS, Dubrovsky GENERATES.
    """
    mini_prompt: str               # "О, опять {topic}?" — prefix for generation
    topic: str                     # Extracted topic from repetition
    past_questions: List[str]      # User's past similar questions
    past_responses: List[str]      # Dubrovsky's past responses
    embarrassment_level: float     # How embarrassing? 0-1
    chaos_triggered: bool          # Did chaos factor trigger?
    mode: str = "mockery"          # Signal to Dubrovsky: mockery mode


class AntiSanta:
    """
    Dubrovsky's repetition detector.

    DETECTS user repetition and builds context for Dubrovsky to generate mockery.
    NO TEMPLATES — Dubrovsky generates his own unique mockery!

    All operations are async.
    """

    def __init__(
        self,
        db_path: str = 'glitches/dubrovsky.db',
        max_recalls: int = 3,
        chaos_factor: float = CHAOS_FACTOR,
    ):
        self.db_path = Path(db_path)
        self.max_recalls = max_recalls
        self.chaos_factor = chaos_factor

        # Stats
        self.total_detections = 0
        self.chaos_recalls = 0
        self.embarrassments_detected = 0

    async def detect_and_build_context(
        self,
        prompt: str,
        session_id: str = 'default',
        arousal: float = 0.5,
        topics: Optional[Sequence[str]] = None,
    ) -> Optional[AntiSantaContext]:
        """
        Detect repetition and build context for Dubrovsky to generate mockery.

        Returns None if no repetition detected.
        Returns AntiSantaContext with mini_prompt for Dubrovsky to use.

        DUBROVSKY GENERATES THE MOCKERY, NOT ANTISANTA.
        """
        try:
            if not prompt or not prompt.strip():
                return None

            if not self.db_path.exists():
                return None

            prompt_lower = prompt.lower()
            prompt_tokens = set(simple_tokenize(prompt_lower))

            if not prompt_tokens:
                return None

            # Query past conversations
            async with aiosqlite.connect(str(self.db_path)) as conn:
                conn.row_factory = aiosqlite.Row

                # Check if conversations table exists
                cursor = await conn.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='conversations'
                """)
                if not await cursor.fetchone():
                    return None

                # Get recent conversations
                cursor = await conn.execute("""
                    SELECT id, prompt, response, coherence_score, session_id, timestamp
                    FROM conversations
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT 200
                """, (session_id,))

                rows = await cursor.fetchall()

            if not rows:
                return None

            # Score each past conversation for repetition potential
            scored: List[tuple[float, dict]] = []
            now = time.time()

            for row in rows:
                past_prompt = row["prompt"]
                past_response = row["response"]
                coherence = row["coherence_score"] or 0.5
                created_at = row["timestamp"] or 0

                if not past_prompt:
                    continue

                past_tokens = set(simple_tokenize(past_prompt.lower()))

                if not past_tokens:
                    continue

                # 1. Token overlap (relevance)
                overlap = len(prompt_tokens & past_tokens)
                union = len(prompt_tokens | past_tokens)
                relevance = overlap / union if union > 0 else 0.0

                # 2. Embarrassment factor (low coherence = embarrassing)
                embarrassment = max(0, 1.0 - coherence)

                # 3. Recency penalty (don't bring up too recent)
                hours_ago = (now - created_at) / 3600.0
                if hours_ago < RECENCY_WINDOW_HOURS:
                    recency_penalty = 1.0 - (hours_ago / RECENCY_WINDOW_HOURS)
                else:
                    recency_penalty = 0.0

                # 4. Combine scores
                score = (
                    0.4 * relevance +
                    0.4 * embarrassment +
                    0.2 * (1.0 - recency_penalty)
                )

                # Penalize sticky phrases
                response_lower = past_response.lower() if past_response else ""
                for phrase in STICKY_PHRASES:
                    if phrase in response_lower:
                        score *= 0.5
                        break

                if score > 0.2:  # Threshold
                    scored.append((score, {
                        "id": row["id"],
                        "prompt": past_prompt,
                        "response": past_response,
                        "coherence": coherence,
                        "relevance": relevance,
                        "embarrassment": embarrassment,
                    }))

            if not scored:
                return None

            # Sort by score descending
            scored.sort(key=lambda x: x[0], reverse=True)

            # CHAOS SELECTION
            selected = []
            chaos_triggered = False

            for i in range(min(self.max_recalls, len(scored))):
                if random.random() < self.chaos_factor and len(scored) > 1:
                    random_idx = random.randint(0, len(scored) - 1)
                    selected.append(scored[random_idx])
                    chaos_triggered = True
                else:
                    if i < len(scored):
                        selected.append(scored[i])

            if chaos_triggered:
                self.chaos_recalls += 1

            self.total_detections += 1

            # Extract topic
            topic = self._extract_topic(prompt, [item[1]["prompt"] for item in selected], topics)

            # Build context
            past_questions = [item[1]["prompt"] for item in selected]
            past_responses = [item[1]["response"] for item in selected if item[1]["response"]]

            # Compute embarrassment level
            embarrassments = [item[1]["embarrassment"] for item in selected]
            avg_embarrassment = sum(embarrassments) / len(embarrassments) if embarrassments else 0.0

            if avg_embarrassment > EMBARRASSMENT_THRESHOLD:
                self.embarrassments_detected += 1

            # Build mini-prompt for Dubrovsky
            # This is the ONLY "template" — and it's minimal, just a trigger
            mini_prompt = f"О, опять {topic}?"

            return AntiSantaContext(
                mini_prompt=mini_prompt,
                topic=topic,
                past_questions=past_questions,
                past_responses=past_responses,
                embarrassment_level=avg_embarrassment,
                chaos_triggered=chaos_triggered,
                mode="mockery",
            )

        except Exception:
            # Silent fail — AntiSanta must never break Dubrovsky
            return None

    def _extract_topic(
        self,
        current_prompt: str,
        past_prompts: List[str],
        external_topics: Optional[Sequence[str]] = None,
    ) -> str:
        """Extract the common topic from current and past prompts."""
        current_words = set(current_prompt.lower().split())
        common_words = []

        # Stop words to filter out
        stop_words = {
            'what', 'is', 'the', 'a', 'an', 'to', 'for', 'of', 'in', 'and', 'or',
            'how', 'why', 'when', 'where', 'who', 'which', 'do', 'does', 'did',
            'have', 'has', 'had', 'be', 'been', 'are', 'was', 'were', 'will',
            'would', 'could', 'should', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'my', 'your', 'this', 'that', 'these', 'those', 'q', 'a', 'about',
            'tell', 'me', 'can', 'please', 'know', 'think', 'want',
        }

        for past in past_prompts:
            past_words = set(past.lower().split())
            common = current_words & past_words - stop_words
            common_words.extend(list(common))

        if common_words:
            topic_counts = Counter(common_words)
            top_topic = topic_counts.most_common(1)[0][0]
            return top_topic
        elif external_topics:
            return external_topics[0]
        else:
            # Extract most significant word from prompt
            prompt_words = [w for w in current_prompt.lower().split() if w not in stop_words and len(w) > 2]
            return prompt_words[0] if prompt_words else "это"

    # Keep old method name for backwards compatibility
    async def recall(
        self,
        prompt: str,
        session_id: str = 'default',
        arousal: float = 0.5,
        topics: Optional[Sequence[str]] = None,
    ) -> Optional[AntiSantaContext]:
        """Backwards compatible alias for detect_and_build_context."""
        return await self.detect_and_build_context(prompt, session_id, arousal, topics)

    def stats(self) -> Dict[str, Any]:
        """Return detection stats."""
        return {
            "total_detections": self.total_detections,
            "chaos_recalls": self.chaos_recalls,
            "chaos_ratio": self.chaos_recalls / max(1, self.total_detections),
            "embarrassments_detected": self.embarrassments_detected,
            "chaos_factor": self.chaos_factor,
        }


__all__ = ['AntiSanta', 'AntiSantaContext']
