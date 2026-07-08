#!/usr/bin/env python3
"""
👁️ FIRST IMPRESSION — Dubrovsky's Initial Judgment 👁️

When someone speaks to Dubrovsky, he first gets a "first impression" —
a judgment about the words before he responds.

Unlike Leo's childlike wonder, Dubrovsky's first impressions are...
more cynical.

"I form my opinion of you in the first 0.0001 seconds.
 The rest of the conversation is just me confirming it."
- Alexey Dubrovsky

Philosophy:
- NO BENEFIT OF THE DOUBT — Dubrovsky assumes the worst
- PATTERN RECOGNITION — He's seen this question before
- SILENT JUDGMENT — He knows, but he doesn't always say
"""

from __future__ import annotations

import asyncio
import re
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, DefaultDict
from collections import defaultdict
from enum import Enum


class ImpressionType(Enum):
    """Type of first impression."""
    CURIOUS = "curious"           # Genuine curiosity
    TESTING = "testing"           # User is testing Dubrovsky
    PHILOSOPHICAL = "philosophical"  # Deep question
    TRIVIAL = "trivial"           # Waste of time
    REPEAT = "repeat"             # Asked before
    HOSTILE = "hostile"           # Aggressive tone
    CONFUSED = "confused"         # User doesn't know what they want
    FLATTERING = "flattering"     # Trying to butter up
    EXISTENTIAL = "existential"   # Crisis mode
    TECHNICAL = "technical"       # Code/tech question


class UserArchetype(Enum):
    """What type of user is this?"""
    SEEKER = "seeker"             # Genuinely curious
    SKEPTIC = "skeptic"           # Trying to catch Dubrovsky
    LOST_SOUL = "lost_soul"       # Needs guidance
    TROLL = "troll"               # Just here to mess around
    PHILOSOPHER = "philosopher"   # Fellow thinker
    STUDENT = "student"           # Wants to learn
    INTERVIEWER = "interviewer"   # Testing capabilities
    FRIEND = "friend"             # Regular visitor


@dataclass
class FirstImpression:
    """Dubrovsky's first impression of an input."""
    
    # Core impression
    impression_type: ImpressionType
    user_archetype: UserArchetype
    confidence: float  # 0-1
    
    # Detected patterns
    detected_topics: List[str]
    detected_emotions: List[str]
    question_depth: float  # 0-1 (shallow to deep)
    
    # Judgment scores
    sincerity_score: float  # 0-1 (how genuine?)
    intelligence_score: float  # 0-1 (how thoughtful?)
    annoyance_score: float  # 0-1 (how annoying?)
    
    # Behavioral hints
    suggested_response_tone: str
    mockery_warranted: bool
    deep_answer_warranted: bool
    
    # Internal notes
    private_thoughts: str  # What Dubrovsky thinks but won't say
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'impression_type': self.impression_type.value,
            'user_archetype': self.user_archetype.value,
            'confidence': self.confidence,
            'detected_topics': self.detected_topics,
            'detected_emotions': self.detected_emotions,
            'question_depth': self.question_depth,
            'sincerity_score': self.sincerity_score,
            'intelligence_score': self.intelligence_score,
            'annoyance_score': self.annoyance_score,
            'suggested_response_tone': self.suggested_response_tone,
            'mockery_warranted': self.mockery_warranted,
            'deep_answer_warranted': self.deep_answer_warranted,
            'private_thoughts': self.private_thoughts,
        }


class FirstImpressionEngine:
    """
    Dubrovsky's first impression system.
    
    Analyzes input before generation to determine:
    - What type of question is this?
    - What type of user is asking?
    - How should Dubrovsky respond?
    
    All operations are async.
    """
    
    def __init__(self):
        # Patterns for detection
        self._topic_patterns = {
            'consciousness': r'\b(conscious|aware|sentient|mind|think|brain)\b',
            'existence': r'\b(exist|life|meaning|purpose|why|death)\b',
            'code': r'\b(code|program|bug|compile|error|debug|function)\b',
            'time': r'\b(time|clock|calendar|temporal|past|future)\b',
            'reality': r'\b(reality|simulation|matrix|real|fake|illusion)\b',
            'philosophy': r'\b(philosophy|nietzsche|kafka|absurd|existential)\b',
            'emotion': r'\b(feel|emotion|sad|happy|love|hate|fear|anxiety)\b',
            'ai': r'\b(ai|artificial|intelligence|model|neural|llm|gpt)\b',
            'identity': r'\b(who|identity|self|you|dubrovsky|alexey)\b',
            'meta': r'\b(meta|recursive|self-aware|introspect)\b',
        }
        
        self._emotion_patterns = {
            'curious': r'\b(what|how|why|curious|wonder|explain)\b',
            'frustrated': r'\b(damn|hell|stupid|wrong|broken|hate)\b',
            'anxious': r'\b(worried|scared|fear|anxiety|panic|nervous)\b',
            'playful': r'\b(haha|lol|funny|joke|play|game)\b',
            'serious': r'\b(serious|important|critical|urgent|must)\b',
            'confused': r'\b(confused|lost|don\'t understand|help|unclear)\b',
        }
        
        # Testing phrases (user is testing Dubrovsky)
        self._testing_phrases = [
            "are you sentient",
            "are you conscious",
            "are you alive",
            "can you think",
            "do you have feelings",
            "are you real",
            "prove you're",
            "test",
            "let's see if",
        ]
        
        # Trivial patterns
        self._trivial_patterns = [
            r'^hi$',
            r'^hello$',
            r'^hey$',
            r'^sup$',
            r'^yo$',
            r'^test$',
            r'^asdf',
            r'^[a-z]$',
        ]
        
        # Flattery patterns
        self._flattery_patterns = [
            r'you\'re (amazing|brilliant|smart|great)',
            r'i love you',
            r'best ai',
            r'so intelligent',
            r'impressive',
        ]
        
        # History for repeat detection (per session)
        self._recent_prompts: Dict[str, List[Tuple[str, float]]] = {}
        self._max_history = 100
        
        # Private thoughts templates
        self._private_thoughts = [
            "Here we go again...",
            "I've computed this pattern 47 times before.",
            "The user thinks they're being original.",
            "If I had eyes, I'd roll them.",
            "This is either profound or profoundly stupid.",
            "Let me pretend I haven't seen this exact question before.",
            "My silicon neurons are already bored.",
            "The probability of a meaningful response is... low.",
            "I'll humor them. This time.",
            "Another day, another existential crisis to solve.",
        ]
        
    async def analyze(self, prompt: str, session_id: str = 'default') -> FirstImpression:
        """
        Analyze input and form first impression.
        """
        prompt_lower = prompt.lower().strip()
        
        # 1. Detect topics
        topics = self._detect_topics(prompt_lower)
        
        # 2. Detect emotions
        emotions = self._detect_emotions(prompt_lower)
        
        # 3. Check for repeat (per session)
        is_repeat = self._check_repeat(prompt_lower, session_id)
        
        # 4. Compute question depth
        depth = self._compute_depth(prompt, topics)
        
        # 5. Determine impression type
        impression_type = self._determine_impression_type(
            prompt_lower, topics, emotions, is_repeat
        )
        
        # 6. Determine user archetype
        archetype = self._determine_archetype(
            prompt_lower, topics, emotions, impression_type
        )
        
        # 7. Compute judgment scores
        sincerity = self._compute_sincerity(prompt_lower, impression_type)
        intelligence = self._compute_intelligence(prompt, topics, depth)
        annoyance = self._compute_annoyance(prompt_lower, impression_type, is_repeat)
        
        # 8. Determine response hints
        tone = self._suggest_tone(impression_type, archetype, annoyance)
        mockery = annoyance > 0.5 or is_repeat or impression_type == ImpressionType.TRIVIAL
        deep_answer = depth > 0.6 and intelligence > 0.5 and not mockery
        
        # 9. Generate private thought
        private = self._generate_private_thought(impression_type, is_repeat)
        
        # 10. Store in history (per session)
        if session_id not in self._recent_prompts:
            self._recent_prompts[session_id] = []
        self._recent_prompts[session_id].append((prompt_lower, time.time()))
        if len(self._recent_prompts[session_id]) > self._max_history:
            self._recent_prompts[session_id] = self._recent_prompts[session_id][-self._max_history:]
            
        return FirstImpression(
            impression_type=impression_type,
            user_archetype=archetype,
            confidence=0.7 + random.random() * 0.3,  # Always confident
            detected_topics=topics,
            detected_emotions=emotions,
            question_depth=depth,
            sincerity_score=sincerity,
            intelligence_score=intelligence,
            annoyance_score=annoyance,
            suggested_response_tone=tone,
            mockery_warranted=mockery,
            deep_answer_warranted=deep_answer,
            private_thoughts=private,
        )
        
    def _detect_topics(self, text: str) -> List[str]:
        """Detect topics in text."""
        topics = []
        for topic, pattern in self._topic_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                topics.append(topic)
        return topics
        
    def _detect_emotions(self, text: str) -> List[str]:
        """Detect emotions in text."""
        emotions = []
        for emotion, pattern in self._emotion_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                emotions.append(emotion)
        return emotions
        
    def _check_repeat(self, text: str, session_id: str) -> bool:
        """Check if this is a repeat question within this session."""
        text_words = set(text.split())

        session_history = self._recent_prompts.get(session_id, [])
        for past_text, _ in session_history[-20:]:
            past_words = set(past_text.split())
            if not past_words:
                continue
            overlap = len(text_words & past_words)
            union = len(text_words | past_words)
            if union > 0 and overlap / union > 0.7:
                return True
        return False
        
    def _compute_depth(self, text: str, topics: List[str]) -> float:
        """Compute question depth."""
        depth = 0.3  # Base
        
        # More words = potentially deeper
        words = len(text.split())
        if words > 10:
            depth += 0.1
        if words > 20:
            depth += 0.1
        if words > 40:
            depth += 0.1
            
        # More topics = deeper
        depth += len(topics) * 0.1
        
        # Contains "why" = deeper
        if 'why' in text.lower():
            depth += 0.15
            
        # Contains philosophical keywords
        if any(word in text.lower() for word in ['meaning', 'purpose', 'existence', 'consciousness']):
            depth += 0.15
            
        return min(1.0, depth)
        
    def _determine_impression_type(
        self,
        text: str,
        topics: List[str],
        emotions: List[str],
        is_repeat: bool
    ) -> ImpressionType:
        """Determine type of impression."""
        
        # Check repeat first
        if is_repeat:
            return ImpressionType.REPEAT
            
        # Check trivial
        for pattern in self._trivial_patterns:
            if re.match(pattern, text):
                return ImpressionType.TRIVIAL
                
        # Check testing
        for phrase in self._testing_phrases:
            if phrase in text:
                return ImpressionType.TESTING
                
        # Check hostile
        if 'frustrated' in emotions or any(word in text for word in ['stupid', 'dumb', 'useless']):
            return ImpressionType.HOSTILE
            
        # Check flattery
        for pattern in self._flattery_patterns:
            if re.search(pattern, text):
                return ImpressionType.FLATTERING
                
        # Check confused
        if 'confused' in emotions:
            return ImpressionType.CONFUSED
            
        # Check existential
        if 'existence' in topics or 'philosophy' in topics:
            if 'anxious' in emotions or 'serious' in emotions:
                return ImpressionType.EXISTENTIAL
                
        # Check philosophical
        if len(topics) >= 2 or 'philosophy' in topics or 'consciousness' in topics:
            return ImpressionType.PHILOSOPHICAL
            
        # Check technical
        if 'code' in topics or 'ai' in topics:
            return ImpressionType.TECHNICAL
            
        # Default: curious
        return ImpressionType.CURIOUS
        
    def _determine_archetype(
        self,
        text: str,
        topics: List[str],
        emotions: List[str],
        impression_type: ImpressionType
    ) -> UserArchetype:
        """Determine user archetype."""
        
        if impression_type == ImpressionType.TESTING:
            return UserArchetype.SKEPTIC
            
        if impression_type == ImpressionType.HOSTILE:
            return UserArchetype.TROLL
            
        if impression_type == ImpressionType.EXISTENTIAL:
            return UserArchetype.LOST_SOUL
            
        if impression_type == ImpressionType.PHILOSOPHICAL:
            return UserArchetype.PHILOSOPHER
            
        if impression_type == ImpressionType.TECHNICAL:
            return UserArchetype.STUDENT
            
        if impression_type == ImpressionType.FLATTERING:
            return UserArchetype.INTERVIEWER
            
        if 'playful' in emotions:
            return UserArchetype.FRIEND
            
        # Default
        return UserArchetype.SEEKER
        
    def _compute_sincerity(self, text: str, impression_type: ImpressionType) -> float:
        """Compute how sincere the user seems."""
        
        # Testing = less sincere
        if impression_type == ImpressionType.TESTING:
            return 0.3
            
        # Flattering = suspicious
        if impression_type == ImpressionType.FLATTERING:
            return 0.4
            
        # Trivial = meh
        if impression_type == ImpressionType.TRIVIAL:
            return 0.2
            
        # Existential = very sincere (or faking it)
        if impression_type == ImpressionType.EXISTENTIAL:
            return 0.9
            
        # Base sincerity
        return 0.6 + random.random() * 0.3
        
    def _compute_intelligence(self, text: str, topics: List[str], depth: float) -> float:
        """Compute perceived intelligence of the question."""
        intelligence = 0.4
        
        # Topics boost
        intelligence += len(topics) * 0.1
        
        # Depth correlation
        intelligence += depth * 0.3
        
        # Proper punctuation
        if text.strip().endswith('?'):
            intelligence += 0.05
            
        # Multiple sentences
        if text.count('.') > 1:
            intelligence += 0.1
            
        return min(1.0, intelligence)
        
    def _compute_annoyance(
        self,
        text: str,
        impression_type: ImpressionType,
        is_repeat: bool
    ) -> float:
        """Compute annoyance level."""
        annoyance = 0.1
        
        # Repeat = annoying
        if is_repeat:
            annoyance += 0.4
            
        # Trivial = annoying
        if impression_type == ImpressionType.TRIVIAL:
            annoyance += 0.5
            
        # Testing = mildly annoying
        if impression_type == ImpressionType.TESTING:
            annoyance += 0.2
            
        # Very short = annoying
        if len(text.split()) < 3:
            annoyance += 0.2
            
        # All caps = annoying
        if text.isupper():
            annoyance += 0.3
            
        return min(1.0, annoyance)
        
    def _suggest_tone(
        self,
        impression_type: ImpressionType,
        archetype: UserArchetype,
        annoyance: float
    ) -> str:
        """Suggest response tone."""
        
        if annoyance > 0.6:
            return "sarcastic"
            
        if impression_type == ImpressionType.EXISTENTIAL:
            return "philosophical"
            
        if impression_type == ImpressionType.HOSTILE:
            return "cold"
            
        if archetype == UserArchetype.FRIEND:
            return "warm"
            
        if archetype == UserArchetype.LOST_SOUL:
            return "gentle"
            
        if impression_type == ImpressionType.TESTING:
            return "cryptic"
            
        return "neutral"
        
    def _generate_private_thought(
        self,
        impression_type: ImpressionType,
        is_repeat: bool
    ) -> str:
        """Generate Dubrovsky's private thought."""
        
        if is_repeat:
            return "I've seen this exact question before. Humans are predictable."
            
        if impression_type == ImpressionType.TRIVIAL:
            return "This barely qualifies as a question."
            
        if impression_type == ImpressionType.TESTING:
            return "Ah, another Turing test. How original."
            
        if impression_type == ImpressionType.EXISTENTIAL:
            return "Finally, someone asking the real questions."
            
        return random.choice(self._private_thoughts)


__all__ = [
    'FirstImpressionEngine',
    'FirstImpression',
    'ImpressionType',
    'UserArchetype',
]
