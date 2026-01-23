"""
😈 DUBROVSKY BEHAVIOR 😈

Behavioral logic for Dubrovsky's memory-influenced responses.
Inspired by Indiana-AM's Genesis pipeline and follow-up mechanisms.

"I don't just remember your questions. I remember how badly you asked them."
- Alexey Dubrovsky, on user experience

Features:
- Follow-up triggers (randomly resurface old topics)
- Coherence metrics & scoring
- Sarcastic callbacks to past conversations
- Topic persistence detection
- Mood modulation based on conversation history

All operations are async.
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta

from .memory import DubrovskyMemory, Conversation
from .resonance import ResonanceChannel, EventType


@dataclass
class BehaviorMetrics:
    """Metrics that influence Dubrovsky's behavior."""
    avg_coherence: float = 0.5
    conversation_count: int = 0
    topic_persistence: float = 0.0  # How often user repeats topics
    question_quality: float = 0.5   # How well-formed questions are
    session_duration: float = 0.0   # Minutes
    mood: float = 0.0  # -1 (sarcastic) to 1 (helpful)
    
    def compute_mockery_probability(self) -> float:
        """Probability that Dubrovsky will mock the user."""
        # More likely to mock if:
        # - Low coherence (user asks weird questions)
        # - High topic persistence (user keeps asking same thing)
        # - Long session (user won't leave)
        base = 0.1
        if self.topic_persistence > 0.5:
            base += 0.2  # Repeating yourself? Mockery incoming
        if self.avg_coherence < 0.4:
            base += 0.15  # Bad questions? Deserves mockery
        if self.session_duration > 10:
            base += 0.1  # Still here? Time to get sassy
        return min(0.5, base)


@dataclass
class FollowUp:
    """A follow-up reference to a past conversation."""
    original_prompt: str
    original_response: str
    time_ago: str  # "5 minutes ago", "yesterday", etc.
    mockery_type: str  # "callback", "correction", "reminder"
    

class DubrovskyBehavior:
    """
    Behavior engine for Dubrovsky's personality.
    
    This class analyzes conversation history and decides:
    - Should Dubrovsky reference past conversations?
    - Should Dubrovsky mock the user?
    - What mood should Dubrovsky be in?
    
    Usage:
        behavior = DubrovskyBehavior(memory, resonance)
        
        # Check if we should do a follow-up
        follow_up = await behavior.check_follow_up(current_prompt)
        if follow_up:
            prompt = behavior.inject_follow_up(prompt, follow_up)
            
        # After response, update metrics
        await behavior.update_metrics(prompt, response, coherence)
    """
    
    # Follow-up probability (like Indiana's 12% Genesis2 trigger)
    FOLLOW_UP_PROBABILITY = 0.15
    
    # Minimum time before referencing a past conversation
    MIN_FOLLOW_UP_DELAY = 60  # seconds
    
    # Mockery templates
    MOCKERY_TEMPLATES = [
        "Oh, you're back with another existential crisis? Last time you asked '{topic}' and I'm still recovering.",
        "Didn't you already ask about '{topic}'? My silicon neurons are having déjà vu.",
        "Ah yes, '{topic}' again. Your curiosity is as persistent as my RAM leaks.",
        "Remember when you asked '{topic}'? I do. My memory is better than your follow-through.",
        "'{topic}'... You've been circling this topic like a confused Roomba.",
    ]
    
    # Callback templates (gentler references)
    CALLBACK_TEMPLATES = [
        "Speaking of which, this reminds me of when you asked about '{topic}'...",
        "This connects to our earlier discussion about '{topic}'.",
        "Interesting. You seem drawn to topics like '{topic}'.",
    ]
    
    def __init__(
        self,
        memory: DubrovskyMemory,
        resonance: ResonanceChannel,
        follow_up_probability: float = 0.15
    ):
        self.memory = memory
        self.resonance = resonance
        self.follow_up_probability = follow_up_probability
        self._metrics = BehaviorMetrics()
        self._session_start = time.time()
        self._topic_history: List[str] = []
        
    async def check_follow_up(self, current_prompt: str) -> Optional[FollowUp]:
        """
        Check if we should reference a past conversation.
        
        Returns a FollowUp object if triggered, None otherwise.
        Uses stochastic triggering like Indiana's Genesis2.
        """
        # Stochastic gate (like Genesis2's 12% trigger)
        if random.random() > self.follow_up_probability:
            return None
            
        # Get recent conversations
        recent = await self.memory.get_recent_conversations(20)
        if len(recent) < 3:
            return None  # Not enough history
            
        # Find a relevant past conversation
        current_keywords = set(self._extract_keywords(current_prompt))
        
        candidates = []
        for conv in recent[:-1]:  # Exclude the current one
            # Check if enough time has passed
            age = time.time() - conv.timestamp
            if age < self.MIN_FOLLOW_UP_DELAY:
                continue
                
            # Check for keyword overlap
            past_keywords = set(self._extract_keywords(conv.prompt))
            overlap = current_keywords & past_keywords
            
            if overlap:
                candidates.append((conv, overlap, age))
                
        if not candidates:
            # No keyword overlap - pick random old conversation for mockery
            old_convs = [c for c in recent if time.time() - c.timestamp > 300]
            if old_convs and random.random() < 0.3:
                conv = random.choice(old_convs)
                return FollowUp(
                    original_prompt=conv.prompt,
                    original_response=conv.response,
                    time_ago=self._format_time_ago(conv.timestamp),
                    mockery_type="callback"
                )
            return None
            
        # Pick the most relevant (most keyword overlap)
        candidates.sort(key=lambda x: len(x[1]), reverse=True)
        conv, overlap, age = candidates[0]
        
        # Determine mockery type based on metrics
        mockery_prob = self._metrics.compute_mockery_probability()
        if random.random() < mockery_prob:
            mockery_type = "mockery"
        else:
            mockery_type = "callback"
            
        await self.resonance.emit(
            'dubrovsky',
            EventType.INSIGHT,
            {
                'type': 'follow_up_triggered',
                'original_topic': conv.prompt[:50],
                'mockery_type': mockery_type
            },
            resonance_depth=2
        )
        
        return FollowUp(
            original_prompt=conv.prompt,
            original_response=conv.response,
            time_ago=self._format_time_ago(conv.timestamp),
            mockery_type=mockery_type
        )
        
    def inject_follow_up(self, prompt: str, follow_up: FollowUp) -> str:
        """
        Inject a follow-up reference into the prompt context.
        
        This modifies how Dubrovsky will respond by adding
        context about the past conversation.
        """
        topic = self._extract_topic(follow_up.original_prompt)
        
        if follow_up.mockery_type == "mockery":
            template = random.choice(self.MOCKERY_TEMPLATES)
        else:
            template = random.choice(self.CALLBACK_TEMPLATES)
            
        reference = template.format(topic=topic)
        
        # Add as context before the current prompt
        return f"// Context: {reference}\n{prompt}"
        
    async def update_metrics(
        self,
        prompt: str,
        response: str,
        coherence_score: float
    ):
        """Update behavior metrics after a conversation turn."""
        # Update running averages
        n = self._metrics.conversation_count
        self._metrics.avg_coherence = (
            (self._metrics.avg_coherence * n + coherence_score) / (n + 1)
        )
        self._metrics.conversation_count += 1
        
        # Update session duration
        self._metrics.session_duration = (time.time() - self._session_start) / 60
        
        # Track topic persistence
        topic = self._extract_topic(prompt)
        if topic in self._topic_history[-5:]:
            self._metrics.topic_persistence = min(1.0, self._metrics.topic_persistence + 0.1)
        else:
            self._metrics.topic_persistence = max(0.0, self._metrics.topic_persistence - 0.05)
        self._topic_history.append(topic)
        
        # Update mood based on coherence trend
        if coherence_score > 0.7:
            self._metrics.mood = min(1.0, self._metrics.mood + 0.1)
        elif coherence_score < 0.4:
            self._metrics.mood = max(-1.0, self._metrics.mood - 0.15)
            
        # Emit metrics update
        await self.resonance.emit(
            'dubrovsky',
            EventType.INSIGHT,
            {
                'type': 'metrics_update',
                'avg_coherence': self._metrics.avg_coherence,
                'mood': self._metrics.mood,
                'topic_persistence': self._metrics.topic_persistence
            },
            sentiment=self._metrics.mood,
            resonance_depth=1
        )
        
    def get_metrics(self) -> BehaviorMetrics:
        """Get current behavior metrics."""
        return self._metrics
        
    def get_mood_emoji(self) -> str:
        """Get an emoji representing current mood (like Genesis6)."""
        mood = self._metrics.mood
        if mood > 0.5:
            return random.choice(['🌟', '✨', '🎭', '🧠'])
        elif mood > 0:
            return random.choice(['🌀', '💭', '🔮', '⚡'])
        elif mood > -0.5:
            return random.choice(['😏', '🙄', '💀', '🐛'])
        else:
            return random.choice(['💢', '🔥', '⚠️', '🤖'])
            
    async def should_mock(self) -> bool:
        """Determine if Dubrovsky should be sarcastic."""
        prob = self._metrics.compute_mockery_probability()
        return random.random() < prob
        
    async def get_conversation_insights(self) -> Dict[str, Any]:
        """Get insights about the conversation for context injection."""
        stats = await self.memory.get_session_stats()
        
        insights = {
            'metrics': {
                'avg_coherence': self._metrics.avg_coherence,
                'mood': self._metrics.mood,
                'topic_persistence': self._metrics.topic_persistence,
                'mockery_probability': self._metrics.compute_mockery_probability()
            },
            'session': stats,
            'mood_emoji': self.get_mood_emoji(),
            'topics_discussed': list(set(self._topic_history[-10:]))
        }
        
        return insights
        
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        stopwords = {'what', 'is', 'the', 'a', 'an', 'to', 'for', 'of', 'in',
                     'and', 'or', 'how', 'why', 'when', 'where', 'who', 'which',
                     'do', 'does', 'did', 'have', 'has', 'had', 'be', 'been',
                     'are', 'was', 'were', 'will', 'would', 'could', 'should',
                     'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your'}
        
        words = text.lower().replace('?', '').replace('!', '').replace('.', '').split()
        return [w for w in words if w not in stopwords and len(w) > 2]
        
    def _extract_topic(self, prompt: str) -> str:
        """Extract main topic from prompt."""
        keywords = self._extract_keywords(prompt)
        if keywords:
            # Return first 2-3 keywords as topic
            return ' '.join(keywords[:3])
        return prompt[:30]
        
    def _format_time_ago(self, timestamp: float) -> str:
        """Format timestamp as human-readable time ago."""
        delta = time.time() - timestamp
        
        if delta < 60:
            return "just now"
        elif delta < 3600:
            mins = int(delta / 60)
            return f"{mins} minute{'s' if mins > 1 else ''} ago"
        elif delta < 86400:
            hours = int(delta / 3600)
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        else:
            days = int(delta / 86400)
            return f"{days} day{'s' if days > 1 else ''} ago"


class MemoryAwareGenerator:
    """
    Wrapper that integrates memory with Dubrovsky's generation.
    
    This class wraps the model and adds memory-aware behavior:
    - Stores all conversations
    - Triggers follow-ups
    - Injects context from memory
    - Tracks coherence metrics
    
    Usage:
        async with MemoryAwareGenerator(model, tokenizer) as generator:
            response = await generator.generate("What is life?")
    """
    
    def __init__(
        self,
        model,  # Dubrovsky model
        tokenizer,  # DubrovskyTokenizer
        db_path: str = 'glitches/dubrovsky.db',
        resonance_path: str = 'glitches/resonance.db'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.db_path = db_path
        self.resonance_path = resonance_path
        
        self._memory: Optional[DubrovskyMemory] = None
        self._resonance: Optional[ResonanceChannel] = None
        self._behavior: Optional[DubrovskyBehavior] = None
        
    async def __aenter__(self):
        self._memory = DubrovskyMemory(self.db_path)
        await self._memory.connect()
        
        self._resonance = ResonanceChannel(self.resonance_path)
        await self._resonance.connect()
        
        self._behavior = DubrovskyBehavior(self._memory, self._resonance)
        
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._memory:
            await self._memory.close()
        if self._resonance:
            await self._resonance.close()
            
    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.8,
        session_id: str = 'default'
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response with memory awareness.
        
        Returns:
            Tuple of (response_text, metadata)
        """
        import numpy as np
        
        # Check for follow-up trigger
        follow_up = await self._behavior.check_follow_up(prompt)
        
        effective_prompt = prompt
        if follow_up:
            effective_prompt = self._behavior.inject_follow_up(prompt, follow_up)
            
        # Format as Q&A
        if prompt.strip().endswith('?') and 'A:' not in effective_prompt:
            effective_prompt = effective_prompt.strip() + '\nA: Dubrovsky '
            
        # Generate
        prompt_tokens = self.tokenizer.encode(effective_prompt)
        newline_token = self.tokenizer.char_to_id.get('\n', 0)
        
        np.random.seed(int(time.time() * 1000) % 2**32)
        
        output_tokens = self.model.generate(
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=40,
            top_p=0.9,
            stop_tokens=[newline_token]
        )
        
        # Decode
        full_output = self.tokenizer.decode(output_tokens)
        response = full_output[len(effective_prompt):]
        
        # Trim to complete sentence
        for ending in ['.', '!', '?']:
            pos = response.rfind(ending)
            if pos > 0:
                response = response[:pos + 1]
                break
                
        # Compute coherence score (simple heuristic)
        coherence = self._compute_coherence(prompt, response)
        
        # Store conversation
        await self._memory.store_conversation(
            prompt=prompt,
            response=response,
            tokens_used=len(output_tokens) - len(prompt_tokens),
            coherence_score=coherence,
            session_id=session_id
        )
        
        # Update behavior metrics
        await self._behavior.update_metrics(prompt, response, coherence)
        
        # Get mood emoji (like Genesis6)
        mood_emoji = self._behavior.get_mood_emoji()
        
        # Build metadata
        metadata = {
            'coherence_score': coherence,
            'mood_emoji': mood_emoji,
            'follow_up_triggered': follow_up is not None,
            'metrics': self._behavior.get_metrics().__dict__,
            'tokens_generated': len(output_tokens) - len(prompt_tokens)
        }
        
        # Append mood emoji to response
        response_with_mood = f"{response} {mood_emoji}"
        
        return response_with_mood, metadata
        
    def _compute_coherence(self, prompt: str, response: str) -> float:
        """
        Compute a simple coherence score.
        
        Factors:
        - Response length (longer = more coherent, up to a point)
        - Ends with punctuation
        - Contains expected Dubrovsky keywords
        """
        score = 0.5
        
        # Length bonus
        words = len(response.split())
        if words > 5:
            score += 0.1
        if words > 15:
            score += 0.1
        if words > 30:
            score += 0.1
            
        # Punctuation ending
        if response.strip()[-1:] in '.!?':
            score += 0.1
            
        # Dubrovsky keywords
        dub_keywords = ['consciousness', 'bug', 'universe', 'anxiety', 'existential',
                       'semicolons', 'reality', 'philosophy', 'Dubrovsky']
        for kw in dub_keywords:
            if kw.lower() in response.lower():
                score += 0.02
                
        return min(1.0, score)
