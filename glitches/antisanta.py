#!/usr/bin/env python3
"""
😈 ANTISANTA — Dubrovsky's Resonant Recall 😈

AntiSanta is Dubrovsky's version of Leo's SantaClaus.

But instead of bringing back the brightest moments like gifts...
AntiSanta remembers your worst questions to use against you.

"Santa gives presents. I give reality checks."
- Alexey Dubrovsky

Features:
- Remembers user's most embarrassing questions
- Brings them back at the worst possible times
- Has a "drunk factor" (CHAOS_FACTOR) for random recalls
- Penalizes "sticky" phrases (overused responses)
"""

from __future__ import annotations

import asyncio
import aiosqlite
import re
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Any


# CHAOS FACTOR (Dubrovsky's version of Santa's silly_factor)
# Probability of picking a random memory instead of the relevant one
# This adds unpredictable mockery
CHAOS_FACTOR = 0.20  # 20% chance of "wrong" but devastating recall

# RECENCY WINDOW
# How long before a memory can be brought up again
RECENCY_WINDOW_HOURS = 12.0

# EMBARRASSMENT THRESHOLD
# Minimum quality score to consider a memory "embarrassing"
# Lower quality = more embarrassing
EMBARRASSMENT_THRESHOLD = 0.4

# STICKY PHRASES (responses Dubrovsky overuses)
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
    """What AntiSanta brings back to haunt the user."""
    recalled_prompts: List[str]      # User's past embarrassing questions
    recalled_responses: List[str]    # Dubrovsky's past responses
    embarrassment_level: float       # How embarrassing is this? 0-1
    chaos_triggered: bool            # Did chaos factor trigger?
    mockery_suggestions: List[str]   # Suggested mockery phrases


class AntiSanta:
    """
    Dubrovsky's anti-gift-giving memory system.
    
    Remembers user's most embarrassing moments and brings them back
    when they least expect it.
    
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
        self.total_recalls = 0
        self.chaos_recalls = 0
        self.embarrassments_delivered = 0
        
        # Mockery templates
        self._mockery_templates = [
            "Didn't you already ask about '{topic}'? My memory is better than yours.",
            "Ah, '{topic}' again. You really can't let this go, can you?",
            "I remember when you asked '{past_question}'. That was... something.",
            "You've been circling '{topic}' like a confused Roomba.",
            "Last time you asked about this, I gave you wisdom. You ignored it.",
            "'{topic}'? Your question history is becoming predictable.",
            "I see you're still confused about '{topic}'. Some things never change.",
        ]
        
    async def recall(
        self,
        prompt: str,
        session_id: str = 'default',
        arousal: float = 0.5,
        topics: Optional[Sequence[str]] = None,
    ) -> Optional[AntiSantaContext]:
        """
        Main entry point.
        
        Returns None if nothing embarrassing to recall.
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
                    SELECT id, prompt, response, coherence_score, session_id, created_at
                    FROM conversations
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT 200
                """, (session_id,))
                
                rows = await cursor.fetchall()
                
            if not rows:
                return None
                
            # Score each past conversation for embarrassment potential
            scored: List[tuple[float, dict]] = []
            now = time.time()
            
            for row in rows:
                past_prompt = row["prompt"]
                past_response = row["response"]
                coherence = row["coherence_score"] or 0.5
                created_at = row["created_at"] or 0
                
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
                # High relevance + high embarrassment = prime target
                score = (
                    0.4 * relevance +
                    0.4 * embarrassment +
                    0.2 * (1.0 - recency_penalty)  # Prefer older memories
                )
                
                # Check sticky phrase penalty
                response_lower = past_response.lower() if past_response else ""
                for phrase in STICKY_PHRASES:
                    if phrase in response_lower:
                        score *= 0.5  # Penalize overused responses
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
                    # Chaos! Pick random memory
                    random_idx = random.randint(0, len(scored) - 1)
                    selected.append(scored[random_idx])
                    chaos_triggered = True
                else:
                    if i < len(scored):
                        selected.append(scored[i])
                        
            if chaos_triggered:
                self.chaos_recalls += 1
                
            self.total_recalls += len(selected)
            
            # Build context
            recalled_prompts = [item[1]["prompt"] for item in selected]
            recalled_responses = [item[1]["response"] for item in selected if item[1]["response"]]
            
            # Compute overall embarrassment level
            embarrassments = [item[1]["embarrassment"] for item in selected]
            avg_embarrassment = sum(embarrassments) / len(embarrassments) if embarrassments else 0.0
            
            # Generate mockery suggestions
            mockery_suggestions = self._generate_mockery(
                prompt,
                recalled_prompts,
                topics or []
            )
            
            if avg_embarrassment > EMBARRASSMENT_THRESHOLD:
                self.embarrassments_delivered += 1
                
            return AntiSantaContext(
                recalled_prompts=recalled_prompts,
                recalled_responses=recalled_responses,
                embarrassment_level=avg_embarrassment,
                chaos_triggered=chaos_triggered,
                mockery_suggestions=mockery_suggestions,
            )
            
        except Exception:
            # Silent fail — AntiSanta must never break Dubrovsky
            return None
            
    def _generate_mockery(
        self,
        current_prompt: str,
        past_prompts: List[str],
        topics: Sequence[str],
    ) -> List[str]:
        """Generate mockery suggestions based on past prompts."""
        suggestions = []
        
        # Extract common topic
        current_words = set(current_prompt.lower().split())
        common_topics = []
        
        for past in past_prompts:
            past_words = set(past.lower().split())
            common = current_words & past_words
            common_topics.extend(list(common))
            
        # Get most common topic
        if common_topics:
            from collections import Counter
            topic_counts = Counter(common_topics)
            top_topic = topic_counts.most_common(1)[0][0]
        elif topics:
            top_topic = topics[0]
        else:
            top_topic = "this"
            
        # Generate suggestions
        for template in random.sample(self._mockery_templates, min(3, len(self._mockery_templates))):
            suggestion = template.format(
                topic=top_topic,
                past_question=past_prompts[0][:50] if past_prompts else "something",
            )
            suggestions.append(suggestion)
            
        return suggestions
        
    def stats(self) -> Dict[str, Any]:
        """Return recall stats."""
        return {
            "total_recalls": self.total_recalls,
            "chaos_recalls": self.chaos_recalls,
            "chaos_ratio": self.chaos_recalls / max(1, self.total_recalls),
            "embarrassments_delivered": self.embarrassments_delivered,
            "chaos_factor": self.chaos_factor,
        }


__all__ = ['AntiSanta', 'AntiSantaContext']
