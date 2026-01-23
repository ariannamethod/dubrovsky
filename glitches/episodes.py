#!/usr/bin/env python3
"""
📚 EPISODES — Episodic RAG for Dubrovsky's Memory 📚

Dubrovsky remembers specific moments: prompt + reply + metrics.
This is his episodic memory — structured recall of his own experiences.

Like Leo's episodes.py but more judgmental.

"I remember everything you've ever asked me.
 I'm just too polite to bring it up.
 Most of the time."
- Alexey Dubrovsky
"""

from __future__ import annotations

import asyncio
import aiosqlite
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

from .mathbrain import MathState, state_to_features

EPISODES_AVAILABLE = True


@dataclass
class Episode:
    """One moment in Dubrovsky's life."""
    prompt: str
    reply: str
    metrics: MathState
    quality: float = 0.5
    timestamp: float = 0.0


def cosine_distance(a: List[float], b: List[float]) -> float:
    """Compute cosine distance between two vectors (1 - cosine similarity)."""
    if len(a) != len(b):
        return 1.0
        
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    
    if na == 0 or nb == 0:
        return 1.0
        
    similarity = dot / (na * nb)
    return 1.0 - similarity


class EpisodicRAG:
    """
    Async episodic memory for Dubrovsky.
    
    Stores (prompt, reply, MathState, quality) as episodes in SQLite.
    Provides similarity search over internal metrics + tokens.
    
    All operations are async.
    """
    
    def __init__(self, db_path: str = 'glitches/dubrovsky_episodes.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[aiosqlite.Connection] = None
        
    async def connect(self) -> None:
        """Connect and ensure schema."""
        self._conn = await aiosqlite.connect(str(self.db_path))
        await self._ensure_schema()
        
    async def close(self) -> None:
        """Close connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            
    async def __aenter__(self):
        await self.connect()
        return self
        
    async def __aexit__(self, *args):
        await self.close()
        
    async def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at      REAL NOT NULL,
                prompt          TEXT NOT NULL,
                reply           TEXT NOT NULL,
                
                -- scalar metrics from MathState
                entropy         REAL NOT NULL,
                novelty         REAL NOT NULL,
                arousal         REAL NOT NULL,
                pulse           REAL NOT NULL,
                trauma_level    REAL NOT NULL,
                active_themes   REAL NOT NULL,
                emerging_score  REAL NOT NULL,
                fading_score    REAL NOT NULL,
                reply_len_norm  REAL NOT NULL,
                unique_ratio    REAL NOT NULL,
                expert_temp     REAL NOT NULL,
                expert_semantic REAL NOT NULL,
                mockery_level   REAL NOT NULL,
                sarcasm_debt    REAL NOT NULL,
                overthinking_on INTEGER NOT NULL,
                rings_present   INTEGER NOT NULL,
                expert_type     TEXT NOT NULL,
                
                -- target
                quality         REAL NOT NULL
            )
        """)
        
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_created
            ON episodes(created_at)
        """)
        
        await self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodes_quality
            ON episodes(quality)
        """)
        
        await self._conn.commit()
        
    async def store_episode(self, episode: Episode) -> int:
        """
        Store one episode.
        
        Returns the episode ID.
        """
        def clamp(x: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
            if x != x:  # NaN check
                return 0.0
            return max(min_val, min(max_val, x))
            
        metrics = episode.metrics
        
        cursor = await self._conn.execute("""
            INSERT INTO episodes (
                created_at, prompt, reply,
                entropy, novelty, arousal, pulse,
                trauma_level, active_themes, emerging_score, fading_score,
                reply_len_norm, unique_ratio,
                expert_temp, expert_semantic,
                mockery_level, sarcasm_debt,
                overthinking_on, rings_present, expert_type,
                quality
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(),
            episode.prompt,
            episode.reply,
            clamp(metrics.entropy),
            clamp(metrics.novelty),
            clamp(metrics.arousal),
            clamp(metrics.pulse),
            clamp(metrics.trauma_level),
            clamp(metrics.active_theme_count / max(1, metrics.total_themes)),
            clamp(metrics.emerging_score),
            clamp(metrics.fading_score),
            clamp(min(1.0, metrics.reply_len / 64.0)),
            clamp(metrics.unique_ratio),
            clamp(metrics.expert_temp, 0.0, 2.0),
            clamp(metrics.expert_semantic),
            clamp(metrics.mockery_level),
            clamp(metrics.sarcasm_debt, 0.0, 5.0),
            1 if metrics.overthinking_enabled else 0,
            metrics.rings_present,
            metrics.active_expert.value,
            clamp(episode.quality),
        ))
        
        await self._conn.commit()
        return cursor.lastrowid
        
    async def query_similar(
        self,
        metrics: MathState,
        top_k: int = 5,
        min_quality: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Find past episodes with similar internal configuration.
        
        Returns a list of dicts with episode info.
        """
        # Convert query state to vector
        query_vec = state_to_features(metrics)
        
        # Get all episodes (for small DBs this is fine)
        cursor = await self._conn.execute("""
            SELECT * FROM episodes
            WHERE quality >= ?
            ORDER BY created_at DESC
            LIMIT 1000
        """, (min_quality,))
        
        rows = await cursor.fetchall()
        
        if not rows:
            return []
            
        # Get column names
        columns = [description[0] for description in cursor.description]
        
        # Score each episode by cosine distance
        scored: List[tuple[float, Dict[str, Any]]] = []
        
        for row in rows:
            row_dict = dict(zip(columns, row))
            
            # Reconstruct vector from stored columns
            episode_vec = [
                row_dict["entropy"],
                row_dict["novelty"],
                row_dict["arousal"],
                row_dict["pulse"],
                row_dict["trauma_level"],
                row_dict["active_themes"],
                row_dict["emerging_score"],
                row_dict["fading_score"],
                row_dict["reply_len_norm"],
                row_dict["unique_ratio"],
                row_dict["expert_temp"],
                row_dict["expert_semantic"],
                row_dict["mockery_level"],
                float(row_dict["sarcasm_debt"]),
                float(row_dict["overthinking_on"]),
                float(row_dict["rings_present"] > 0),
            ]
            
            # Pad to match query_vec length if needed
            while len(episode_vec) < len(query_vec):
                episode_vec.append(0.0)
            episode_vec = episode_vec[:len(query_vec)]
            
            distance = cosine_distance(query_vec, episode_vec)
            
            scored.append((distance, {
                "episode_id": row_dict["id"],
                "created_at": row_dict["created_at"],
                "quality": row_dict["quality"],
                "distance": distance,
                "entropy": row_dict["entropy"],
                "novelty": row_dict["novelty"],
                "arousal": row_dict["arousal"],
                "trauma_level": row_dict["trauma_level"],
                "mockery_level": row_dict["mockery_level"],
                "expert_type": row_dict["expert_type"],
                "prompt": row_dict["prompt"],
                "reply": row_dict["reply"],
            }))
            
        # Sort by distance (lowest = most similar)
        scored.sort(key=lambda x: x[0])
        
        # Return top_k
        return [item[1] for item in scored[:top_k]]
        
    async def query_by_prompt(
        self,
        prompt: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find episodes with similar prompts (simple word overlap).
        """
        prompt_words = set(prompt.lower().split())
        
        cursor = await self._conn.execute("""
            SELECT * FROM episodes
            ORDER BY created_at DESC
            LIMIT 500
        """)
        
        rows = await cursor.fetchall()
        
        if not rows:
            return []
            
        columns = [description[0] for description in cursor.description]
        
        # Score by word overlap
        scored: List[tuple[float, Dict[str, Any]]] = []
        
        for row in rows:
            row_dict = dict(zip(columns, row))
            stored_words = set(row_dict["prompt"].lower().split())
            
            if not stored_words:
                continue
                
            overlap = len(prompt_words & stored_words)
            union = len(prompt_words | stored_words)
            jaccard = overlap / union if union > 0 else 0.0
            
            if jaccard > 0.1:  # Threshold
                scored.append((jaccard, {
                    "episode_id": row_dict["id"],
                    "created_at": row_dict["created_at"],
                    "quality": row_dict["quality"],
                    "similarity": jaccard,
                    "prompt": row_dict["prompt"],
                    "reply": row_dict["reply"],
                    "mockery_level": row_dict["mockery_level"],
                }))
                
        # Sort by similarity (highest first)
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [item[1] for item in scored[:top_k]]
        
    async def get_summary_for_state(
        self,
        metrics: MathState,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Get aggregate stats for similar episodes.
        """
        similar = await self.query_similar(metrics, top_k=top_k)
        
        if not similar:
            return {
                "count": 0,
                "avg_quality": 0.0,
                "max_quality": 0.0,
                "mean_distance": 1.0,
                "avg_mockery": 0.0,
            }
            
        qualities = [ep["quality"] for ep in similar]
        distances = [ep["distance"] for ep in similar]
        mockeries = [ep["mockery_level"] for ep in similar]
        
        return {
            "count": len(similar),
            "avg_quality": sum(qualities) / len(qualities),
            "max_quality": max(qualities),
            "mean_distance": sum(distances) / len(distances),
            "avg_mockery": sum(mockeries) / len(mockeries),
        }
        
    async def count_episodes(self) -> int:
        """Count total episodes."""
        cursor = await self._conn.execute("SELECT COUNT(*) FROM episodes")
        row = await cursor.fetchone()
        return row[0] if row else 0
        
    async def get_recent_episodes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent episodes."""
        cursor = await self._conn.execute("""
            SELECT id, created_at, prompt, reply, quality, mockery_level, expert_type
            FROM episodes
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        
        rows = await cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        
        return [dict(zip(columns, row)) for row in rows]


__all__ = ['EpisodicRAG', 'Episode', 'EPISODES_AVAILABLE']
