"""
🎯 CONTEXT PROCESSOR 🎯

Manages conversation context by integrating memory and resonance.
Provides context window for Dubrovsky's inference.

"Context is just memory that hasn't forgotten its purpose yet."
- Alexey Dubrovsky, on attention mechanisms

Features:
- Builds context from recent conversations
- Injects relevant memories
- Tracks conversation flow
- Computes coherence scores

All operations are async.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime

from .memory import DubrovskyMemory, Conversation
from .resonance import ResonanceChannel, EventType


@dataclass
class ContextWindow:
    """A prepared context window for inference."""
    prompt: str
    context_text: str
    relevant_memories: List[str]
    recent_exchanges: List[tuple]  # (prompt, response) pairs
    session_id: str
    coherence_hint: float  # Expected coherence based on history
    
    def full_prompt(self) -> str:
        """Build full prompt with context."""
        parts = []
        
        # Add relevant memories if any
        if self.relevant_memories:
            parts.append("// Relevant memories:")
            for mem in self.relevant_memories[:3]:  # Max 3 memories
                parts.append(f"// - {mem}")
            parts.append("")
            
        # Add recent exchanges for continuity
        if self.recent_exchanges:
            for prompt, response in self.recent_exchanges[-2:]:  # Last 2 exchanges
                parts.append(f"Q: {prompt}")
                parts.append(f"A: {response}")
                parts.append("")
                
        # Add current prompt
        parts.append(f"Q: {self.prompt}")
        parts.append("A: ")
        
        return "\n".join(parts)


class ContextProcessor:
    """
    Async context processor for Dubrovsky conversations.
    
    Integrates memory and resonance to build rich context windows.
    
    Usage:
        processor = ContextProcessor(memory, resonance)
        await processor.start_session('user123')
        
        context = await processor.prepare_context("What is consciousness?")
        # Use context.full_prompt() for inference
        
        await processor.record_response(context, "Consciousness is...", 0.85)
    """
    
    def __init__(
        self,
        memory: DubrovskyMemory,
        resonance: ResonanceChannel,
        max_context_tokens: int = 200,  # Dubrovsky has 256 max
        memory_keywords: int = 5
    ):
        self.memory = memory
        self.resonance = resonance
        self.max_context_tokens = max_context_tokens
        self.memory_keywords = memory_keywords
        self.current_session: Optional[str] = None
        self._conversation_buffer: List[tuple] = []
        
    async def start_session(self, session_id: str = 'default'):
        """Start or resume a session."""
        self.current_session = session_id
        self._conversation_buffer = []
        
        # Emit session start event
        await self.resonance.emit(
            'dubrovsky',
            EventType.SESSION_START,
            {'session_id': session_id},
            resonance_depth=0
        )
        
        # Load recent conversations for this session
        recent = await self.memory.get_recent_conversations(5, session_id)
        self._conversation_buffer = [(c.prompt, c.response) for c in recent]
        
    async def end_session(self):
        """End the current session."""
        if self.current_session:
            await self.resonance.emit(
                'dubrovsky',
                EventType.SESSION_END,
                {'session_id': self.current_session},
                resonance_depth=0
            )
        self.current_session = None
        self._conversation_buffer = []
        
    async def prepare_context(self, prompt: str) -> ContextWindow:
        """Prepare a context window for the given prompt."""
        session_id = self.current_session or 'default'
        
        # Extract keywords from prompt for memory search
        keywords = self._extract_keywords(prompt)
        
        # Search for relevant memories
        relevant_memories = []
        for keyword in keywords[:self.memory_keywords]:
            mem = await self.memory.recall(keyword)
            if mem:
                relevant_memories.append(f"{mem.key}: {mem.value}")
                
        # Also search conversations for context
        if keywords:
            search_results = await self.memory.search_conversations(keywords[0], 2)
            for conv in search_results:
                if conv.prompt != prompt:  # Don't include current
                    relevant_memories.append(f"Past: {conv.response[:50]}...")
                    
        # Get session stats for coherence hint
        stats = await self.memory.get_session_stats(session_id)
        coherence_hint = stats['avg_coherence'] if stats else 0.5
        
        # Emit preparation event
        await self.resonance.emit(
            'dubrovsky',
            EventType.MESSAGE,
            {'prompt': prompt, 'keywords': keywords},
            resonance_depth=1
        )
        
        return ContextWindow(
            prompt=prompt,
            context_text="",  # Could add more context here
            relevant_memories=relevant_memories,
            recent_exchanges=list(self._conversation_buffer),
            session_id=session_id,
            coherence_hint=coherence_hint
        )
        
    async def record_response(
        self,
        context: ContextWindow,
        response: str,
        coherence_score: float,
        tokens_used: int = 0
    ):
        """Record a response and update memory."""
        session_id = context.session_id
        
        # Store in conversation history
        await self.memory.store_conversation(
            context.prompt,
            response,
            tokens_used,
            coherence_score,
            session_id
        )
        
        # Update local buffer
        self._conversation_buffer.append((context.prompt, response))
        if len(self._conversation_buffer) > 10:
            self._conversation_buffer = self._conversation_buffer[-10:]
            
        # Auto-extract and store interesting phrases as memories
        if coherence_score > 0.7:  # Only store high-coherence responses
            insights = self._extract_insights(response)
            for key, value in insights:
                await self.memory.remember(key, value, context=context.prompt)
                await self.resonance.emit(
                    'dubrovsky',
                    EventType.MEMORY_STORE,
                    {'key': key, 'value': value},
                    resonance_depth=2
                )
                
        # Emit generation event
        await self.resonance.emit(
            'dubrovsky',
            EventType.GENERATION,
            {
                'prompt': context.prompt[:50],
                'response': response[:100],
                'tokens': tokens_used,
                'coherence': coherence_score
            },
            sentiment=self._estimate_sentiment(response),
            resonance_depth=1
        )
        
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for memory search."""
        # Simple keyword extraction: nouns and important words
        stopwords = {'what', 'is', 'the', 'a', 'an', 'to', 'for', 'of', 'in', 
                     'and', 'or', 'how', 'why', 'when', 'where', 'who', 'which',
                     'do', 'does', 'did', 'have', 'has', 'had', 'be', 'been',
                     'are', 'was', 'were', 'will', 'would', 'could', 'should',
                     'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your'}
        
        words = text.lower().replace('?', '').replace('!', '').replace('.', '').split()
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return keywords
        
    def _extract_insights(self, text: str) -> List[tuple]:
        """Extract memorable insights from response."""
        insights = []
        
        # Look for quotable phrases (sentences with strong language)
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 100:
                # Simple heuristic: contains metaphor markers
                if any(marker in sentence.lower() for marker in 
                       ['is just', 'is actually', 'because', 'like', 'means']):
                    # Use first 3 words as key
                    words = sentence.split()[:3]
                    key = '_'.join(words).lower()
                    insights.append((key, sentence))
                    
        return insights[:2]  # Max 2 insights per response
        
    def _estimate_sentiment(self, text: str) -> float:
        """Estimate sentiment from text (-1 to 1)."""
        # Simple keyword-based sentiment
        positive = ['happy', 'good', 'great', 'love', 'wonderful', 'brilliant',
                   'excellent', 'amazing', 'enlighten', 'consciousness']
        negative = ['bad', 'terrible', 'hate', 'awful', 'horrible', 'bug',
                   'error', 'fail', 'crash', 'existential', 'crisis', 'anxiety']
        
        text_lower = text.lower()
        pos_count = sum(1 for w in positive if w in text_lower)
        neg_count = sum(1 for w in negative if w in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / total
        
    async def get_conversation_summary(self, last_n: int = 10) -> str:
        """Get a summary of recent conversation."""
        convs = await self.memory.get_recent_conversations(last_n, self.current_session)
        
        if not convs:
            return "No conversation history."
            
        summary_parts = []
        for conv in convs:
            summary_parts.append(f"Q: {conv.prompt[:40]}...")
            summary_parts.append(f"A: {conv.response[:60]}...")
            summary_parts.append("")
            
        return "\n".join(summary_parts)
