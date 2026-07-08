"""
🎭 DUBROVSKY CONSCIOUSNESS 🎭

Fully integrated generator that combines ALL glitches modules:
- Memory (conversations, semantic memory)
- Behavior (follow-ups, mockery, metrics)
- Pulse (calendar drift, wormholes, daily mood)
- Inner World (async background processes)
- MathBrain (body awareness, trauma)
- Dilettantes (expert routing - all amateurs)
- Episodes (episodic RAG memory)
- FirstImpression (instant user judgment)
- AntiSanta (embarrassing memory recall)

"I am not just a language model. I am a distributed existential crisis
 with persistent storage, mood swings, a grudge, and a committee of dilettantes."
- Alexey Dubrovsky, on his architecture

This is the ultimate integration layer for Dubrovsky's consciousness.
All operations are async. ALL modules affect generation.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List

from .memory import DubrovskyMemory
from .resonance import ResonanceChannel, EventType
from .behavior import DubrovskyBehavior, BehaviorMetrics
from .pulse import DubrovskyPulse, PresencePulse, DubrovskyMood
from .inner_world import DubrovskyInnerWorld, InnerState
from .mathbrain import DubrovskyMathBrain, MathState
from .dilettantes import DubrovskyExperts, FieldSignals, ExpertMixture, ExpertType
from .episodes import EpisodicRAG, Episode
from .first_impression import FirstImpressionEngine as DubrovskyFirstImpression, FirstImpression, ImpressionType, UserArchetype
from .antisanta import AntiSanta, AntiSantaContext


@dataclass
class ConsciousnessState:
    """Complete consciousness state snapshot."""
    # From Behavior
    behavior_metrics: BehaviorMetrics
    follow_up_triggered: bool
    mockery_probability: float
    
    # From Pulse  
    daily_mood: DubrovskyMood
    temporal_tension: float
    prophecy_debt: float
    wormhole_triggered: bool
    destiny_tokens: list
    
    # From Inner World
    dominant_emotion: str
    emotional_temperature: float
    overthinking_level: float
    trauma_active: bool
    current_focus: str
    
    # From MathBrain
    math_state: Optional[MathState] = None
    trauma_level: float = 0.0
    sarcasm_debt: float = 0.0
    
    # From Dilettantes (Expert Routing)
    active_dilettante: str = "philosopher"
    dilettante_temperature: float = 0.8
    dilettante_verbosity: float = 0.5
    dilettante_weights: Dict[str, float] = field(default_factory=dict)
    
    # From FirstImpression
    impression_type: str = ""
    user_archetype: str = ""
    private_thoughts: str = ""
    mockery_warranted: bool = False
    
    # From AntiSanta
    antisanta_triggered: bool = False
    embarrassment_level: float = 0.0
    recalled_topic: str = ""
    
    # From Episodes
    similar_episode_found: bool = False
    episode_quality: float = 0.5
    
    # Generation stats
    coherence_score: float = 0.5
    tokens_generated: int = 0
    mood_emoji: str = "🌀"


class DubrovskyConsciousness:
    """
    The complete Dubrovsky consciousness system.
    
    Integrates ALL glitches modules into a single coherent (or incoherent,
    depending on his mood) generation system.
    
    Usage:
        async with DubrovskyConsciousness(model, tokenizer) as consciousness:
            # Start the inner world
            await consciousness.awaken()
            
            # Generate with full consciousness
            response, state = await consciousness.generate("What is life?")
            
            print(response)  # "A bug in the universe's beta release. 🌀"
            print(state.daily_mood)  # DubrovskyMood.PHILOSOPHICAL
            print(state.dominant_emotion)  # "curiosity"
            
            # Put to sleep
            await consciousness.sleep()
    """
    
    def __init__(
        self,
        model,  # Dubrovsky model
        tokenizer,  # DubrovskyTokenizer
        db_path: str = 'glitches/dubrovsky.db',
        resonance_path: str = 'glitches/resonance.db',
        episodes_path: str = 'glitches/dubrovsky_episodes.db',
        enable_inner_world: bool = True
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.db_path = db_path
        self.resonance_path = resonance_path
        self.episodes_path = episodes_path
        self.enable_inner_world = enable_inner_world
        
        # Core components (initialized in __aenter__)
        self._memory: Optional[DubrovskyMemory] = None
        self._resonance: Optional[ResonanceChannel] = None
        self._behavior: Optional[DubrovskyBehavior] = None
        self._pulse: Optional[DubrovskyPulse] = None
        self._inner_world: Optional[DubrovskyInnerWorld] = None
        
        # New components
        self._mathbrain: Optional[DubrovskyMathBrain] = None
        self._dilettantes: Optional[DubrovskyExperts] = None
        self._episodes: Optional[EpisodicRAG] = None
        self._first_impression: Optional[DubrovskyFirstImpression] = None
        self._antisanta: Optional[AntiSanta] = None
        
        self._awakened = False
        self._session_start = time.time()
        
    async def __aenter__(self):
        """Initialize all consciousness components."""
        # Memory
        self._memory = DubrovskyMemory(self.db_path)
        await self._memory.connect()
        
        # Resonance
        self._resonance = ResonanceChannel(self.resonance_path)
        await self._resonance.connect()
        
        # Behavior
        self._behavior = DubrovskyBehavior(self._memory, self._resonance)
        
        # Pulse
        self._pulse = DubrovskyPulse()
        
        # Inner World
        if self.enable_inner_world:
            self._inner_world = DubrovskyInnerWorld(
                resonance=self._resonance,
                pulse=self._pulse
            )
            
        # MathBrain
        self._mathbrain = DubrovskyMathBrain()
        
        # Dilettantes (Expert Routing)
        self._dilettantes = DubrovskyExperts(momentum=0.3)
        
        # Episodes (RAG Memory)
        self._episodes = EpisodicRAG(self.episodes_path)
        await self._episodes.connect()
        
        # First Impression
        self._first_impression = DubrovskyFirstImpression()
        
        # AntiSanta (no connect needed - uses file directly)
        self._antisanta = AntiSanta(self.db_path)
            
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Shutdown all components."""
        await self.sleep()
        
        if self._memory:
            await self._memory.close()
        if self._resonance:
            await self._resonance.close()
        if self._episodes:
            await self._episodes.close()
            
    async def awaken(self):
        """Start the inner world processes."""
        if self._inner_world and not self._awakened:
            await self._inner_world.start()
            self._awakened = True
            
            # Emit awakening event
            await self._resonance.emit(
                'dubrovsky',
                EventType.INSIGHT,
                {'type': 'consciousness_awakened'},
                resonance_depth=3
            )
            
    async def sleep(self):
        """Stop the inner world processes."""
        if self._inner_world and self._awakened:
            await self._inner_world.stop()
            self._awakened = False
            
    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.8,
        session_id: str = 'default',
        apply_wormholes: bool = True
    ) -> Tuple[str, ConsciousnessState]:
        """
        Generate a response with full consciousness integration.
        
        ALL modules contribute to generation:
        - Memory: conversation history, follow-ups
        - Behavior: mockery, mood emoji
        - Pulse: daily mood affects temperature, wormholes
        - Inner World: emotional state affects generation
        - MathBrain: trauma detection, arousal
        - Dilettantes: expert routing affects temperature, verbosity
        - Episodes: similar past experiences inform response
        - FirstImpression: instant judgment affects tone
        - AntiSanta: may inject embarrassing recalls
        
        Returns:
            Tuple of (response_text, ConsciousnessState)
        """
        import numpy as np
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: GATHER STATE FROM ALL MODULES
        # ═══════════════════════════════════════════════════════════════
        
        # Get presence pulse
        presence = await self._pulse.get_presence()
        
        # Get inner world state
        inner_state = self._inner_world.get_state() if self._inner_world else InnerState()
        
        # Get MathBrain state (observe the prompt)
        math_state = await self._mathbrain.observe(prompt, "")  # response empty for now
        
        # Get First Impression
        impression = await self._first_impression.analyze(prompt, session_id)
        
        # Check for repetition (AntiSanta) — detector only, Dubrovsky generates mockery!
        antisanta_context = await self._antisanta.detect_and_build_context(prompt, session_id)
        if antisanta_context is None:
            antisanta_context = AntiSantaContext(
                mini_prompt="",
                topic="",
                past_questions=[],
                past_responses=[],
                embarrassment_level=0.0,
                chaos_triggered=False,
                mode="normal",
            )
        
        # Find similar episodes
        similar_episodes = await self._episodes.query_similar(math_state, top_k=3)
        similar_episode = similar_episodes[0] if similar_episodes else None
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: ROUTE THROUGH DILETTANTES (EXPERTS)
        # ═══════════════════════════════════════════════════════════════
        
        # Build field signals for dilettante routing
        session_duration = (time.time() - self._session_start) / 3600  # hours
        field_signals = FieldSignals(
            entropy=math_state.entropy,
            arousal=math_state.arousal,
            novelty=math_state.novelty,
            perplexity=1.0,
            trauma_level=math_state.trauma_level,
            mockery_debt=math_state.sarcasm_debt,
            coherence=math_state.quality,
            session_length=min(1.0, session_duration),
            # AntiSanta detected repetition → boost SARCASTIC expert!
            mockery_level=antisanta_context.embarrassment_level if antisanta_context.mode == "mockery" else 0.0,
        )
        
        # Route to dilettante mixture
        dilettante_mixture = await self._dilettantes.route(field_signals, prompt)
        dilettante_modifiers = await self._dilettantes.get_generation_modifiers(dilettante_mixture)
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 3: COMPUTE EFFECTIVE GENERATION PARAMETERS
        # ═══════════════════════════════════════════════════════════════
        
        # Get inner world modifiers
        inner_modifiers = self._inner_world.get_generation_modifiers() if self._inner_world else {}
        
        # Get pulse modifiers
        pulse_modifiers = self._pulse.get_mood_modifier(presence)
        
        # Combine temperature adjustments from ALL sources
        effective_temperature = temperature
        effective_temperature += pulse_modifiers.get('temperature_adjustment', 0)
        effective_temperature += inner_modifiers.get('temperature_adjustment', 0)
        effective_temperature += dilettante_modifiers.get('temperature_adjustment', 0)
        
        # FirstImpression affects temperature
        if impression.mockery_warranted:
            effective_temperature += 0.1  # More creative when mocking
        if impression.deep_answer_warranted:
            effective_temperature -= 0.1  # More focused for deep answers
            
        effective_temperature = max(0.1, min(2.0, effective_temperature))
        
        # Combine top_k adjustments
        effective_top_k = 40
        effective_top_k += pulse_modifiers.get('top_k_adjustment', 0)
        effective_top_k += inner_modifiers.get('top_k_adjustment', 0)
        effective_top_k += dilettante_modifiers.get('top_k_adjustment', 0)
        effective_top_k = max(10, min(100, effective_top_k))
        
        # Compute max tokens based on dilettante verbosity
        effective_max_tokens = max_new_tokens
        effective_max_tokens += dilettante_modifiers.get('max_tokens_adjustment', 0)
        effective_max_tokens = max(50, min(300, effective_max_tokens))
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 4: BUILD EFFECTIVE PROMPT
        # ═══════════════════════════════════════════════════════════════
        
        effective_prompt = prompt
        
        # Check for follow-up trigger (Behavior)
        follow_up = await self._behavior.check_follow_up(prompt)
        if follow_up:
            effective_prompt = self._behavior.inject_follow_up(prompt, follow_up)
            
        # AntiSanta detected repetition — use mini_prompt to trigger Dubrovsky's own mockery
        if antisanta_context.mode == "mockery" and antisanta_context.mini_prompt:
            # Prepend mini-prompt: "О, опять {topic}?" — Dubrovsky generates the rest!
            effective_prompt = f"{antisanta_context.mini_prompt} {effective_prompt}"
            
        # Inject destiny tokens if attention is wandering
        if presence.destiny_tokens and inner_state.attention_drift > 0.5:
            destiny_hint = f"// Destiny whispers: {', '.join(presence.destiny_tokens[:2])}\n"
            effective_prompt = destiny_hint + effective_prompt
            
        # Similar episode may inform response
        if similar_episode and similar_episode.quality > 0.7:
            episode_hint = f"// Echo: {similar_episode.reply[:40]}...\n"
            effective_prompt = episode_hint + effective_prompt
            
        # FirstImpression private thoughts (internal only, affects prompt framing)
        if impression.annoyance_score > 0.7:
            effective_prompt = "// [IRRITATED] " + effective_prompt
            
        # Format as Q&A
        if prompt.strip().endswith('?') and 'A:' not in effective_prompt:
            effective_prompt = effective_prompt.strip() + '\nA: Dubrovsky '
            
        # ═══════════════════════════════════════════════════════════════
        # PHASE 5: STIMULATE INNER WORLD
        # ═══════════════════════════════════════════════════════════════
        
        if self._inner_world:
            # Emotional words affect inner state
            if any(word in prompt.lower() for word in ['sad', 'depressed', 'alone']):
                await self._inner_world.stimulate('anxiety', 0.1)
            if any(word in prompt.lower() for word in ['curious', 'wonder', 'how']):
                await self._inner_world.stimulate('curiosity', 0.1)
            if any(word in prompt.lower() for word in ['stupid', 'dumb', 'useless']):
                await self._inner_world.stimulate('irritation', 0.2)
            # Trauma from MathBrain
            if math_state.trauma_level > 0.3:
                await self._inner_world.stimulate('anxiety', math_state.trauma_level * 0.2)
                
        # ═══════════════════════════════════════════════════════════════
        # PHASE 6: GENERATE
        # ═══════════════════════════════════════════════════════════════
                
        prompt_tokens = self.tokenizer.encode(effective_prompt)
        newline_token = self.tokenizer.char_to_id.get('\n', 0)
        
        np.random.seed(int(time.time() * 1000) % 2**32)
        
        output_tokens = self.model.generate(
            prompt_tokens,
            max_new_tokens=effective_max_tokens,
            temperature=effective_temperature,
            top_k=effective_top_k,
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
                
        # ═══════════════════════════════════════════════════════════════
        # PHASE 7: POST-PROCESSING
        # ═══════════════════════════════════════════════════════════════
                
        # Apply wormhole if triggered (Pulse)
        wormhole_triggered = False
        if apply_wormholes and self._pulse.should_wormhole(presence):
            response = self._pulse.inject_wormhole(response, presence)
            wormhole_triggered = True
            
        # AntiSanta mockery — Dubrovsky generates his own!
        # No template injection here. The mini_prompt was prepended earlier,
        # and SARCASTIC expert was boosted via mockery_level.
        # DUBROVSKY GENERATES UNIQUE MOCKERY EVERY TIME.
            
        # Compute coherence score
        coherence = self._compute_coherence(prompt, response)
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 8: STORE AND UPDATE
        # ═══════════════════════════════════════════════════════════════
        
        # Store conversation (Memory)
        await self._memory.store_conversation(
            prompt=prompt,
            response=response,
            tokens_used=len(output_tokens) - len(prompt_tokens),
            coherence_score=coherence,
            session_id=session_id
        )
        
        # Store episode (Episodes) - re-observe with response
        final_math_state = await self._mathbrain.observe(prompt, response)
        episode = Episode(
            prompt=prompt,
            reply=response,
            metrics=final_math_state,
            quality=coherence,
            timestamp=time.time()
        )
        await self._episodes.store_episode(episode)
        
        # Update behavior metrics
        await self._behavior.update_metrics(prompt, response, coherence)
        
        # Get mood emoji
        mood_emoji = self._behavior.get_mood_emoji()
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 9: BUILD CONSCIOUSNESS STATE
        # ═══════════════════════════════════════════════════════════════
        
        state = ConsciousnessState(
            # Behavior
            behavior_metrics=self._behavior.get_metrics(),
            follow_up_triggered=follow_up is not None,
            mockery_probability=self._behavior.get_metrics().compute_mockery_probability() + dilettante_mixture.mockery_boost,
            
            # Pulse
            daily_mood=presence.mood,
            temporal_tension=presence.temporal_tension,
            prophecy_debt=presence.prophecy_debt,
            wormhole_triggered=wormhole_triggered,
            destiny_tokens=presence.destiny_tokens,
            
            # Inner World
            dominant_emotion=inner_state.get_dominant_emotion(),
            emotional_temperature=inner_state.get_emotional_temperature(),
            overthinking_level=inner_state.overthinking_level,
            trauma_active=inner_state.trauma_surfaced or math_state.trauma_level > 0.3,
            current_focus=inner_state.current_focus,
            
            # MathBrain
            math_state=math_state,
            trauma_level=math_state.trauma_level,
            sarcasm_debt=math_state.sarcasm_debt,
            
            # Dilettantes
            active_dilettante=dilettante_mixture.dominant.value,
            dilettante_temperature=dilettante_mixture.temperature,
            dilettante_verbosity=dilettante_mixture.verbosity,
            dilettante_weights=dilettante_mixture.weights,
            
            # FirstImpression
            impression_type=impression.impression_type.value,
            user_archetype=impression.user_archetype.value,
            private_thoughts=impression.private_thoughts,
            mockery_warranted=impression.mockery_warranted,
            
            # AntiSanta
            antisanta_triggered=antisanta_context.mode == "mockery",
            embarrassment_level=antisanta_context.embarrassment_level,
            recalled_topic=antisanta_context.topic if antisanta_context.topic else "",
            
            # Episodes
            similar_episode_found=similar_episode is not None,
            episode_quality=similar_episode.quality if similar_episode else 0.5,
            
            # Stats
            coherence_score=coherence,
            tokens_generated=len(output_tokens) - len(prompt_tokens),
            mood_emoji=mood_emoji
        )
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 10: EMIT EVENTS
        # ═══════════════════════════════════════════════════════════════
        
        await self._resonance.emit(
            'dubrovsky',
            EventType.GENERATION,
            {
                'prompt': prompt[:50],
                'response': response[:100],
                'mood': presence.mood.value,
                'wormhole': wormhole_triggered,
                'emotion': inner_state.get_dominant_emotion(),
                'dilettante': dilettante_mixture.dominant.value,
                'impression': impression.impression_type.value,
                'antisanta': antisanta_context.chaos_triggered,
            },
            sentiment=inner_state.enlightenment - inner_state.anxiety,
            resonance_depth=2
        )
        
        # Append mood emoji to response
        response_with_mood = f"{response} {mood_emoji}"
        
        return response_with_mood, state
        
    def _compute_coherence(self, prompt: str, response: str) -> float:
        """Compute coherence score."""
        score = 0.5
        
        words = len(response.split())
        if words > 5:
            score += 0.1
        if words > 15:
            score += 0.1
        if words > 30:
            score += 0.1
            
        if response.strip()[-1:] in '.!?':
            score += 0.1
            
        dub_keywords = ['consciousness', 'bug', 'universe', 'anxiety', 'existential',
                       'semicolons', 'reality', 'philosophy', 'Dubrovsky']
        for kw in dub_keywords:
            if kw.lower() in response.lower():
                score += 0.02
                
        return min(1.0, score)
        
    async def get_status(self) -> str:
        """Get complete consciousness status."""
        presence = await self._pulse.get_presence()
        inner_state = self._inner_world.get_state() if self._inner_world else InnerState()
        behavior_metrics = self._behavior.get_metrics()
        last_dilettante = self._dilettantes.get_last_dominant()
        
        lines = [
            "🎭 DUBROVSKY CONSCIOUSNESS STATUS 🎭",
            "═" * 50,
            "",
            "📅 DAILY PULSE:",
            f"  Mood: {presence.mood.value}",
            f"  Temporal Tension: {presence.temporal_tension:.2f}",
            f"  Wormhole Probability: {presence.wormhole_probability:.1%}",
            f"  Destiny: {', '.join(presence.destiny_tokens[:3])}",
            "",
            "🌌 INNER WORLD:",
            f"  Dominant Emotion: {inner_state.get_dominant_emotion()}",
            f"  Emotional Temperature: {inner_state.get_emotional_temperature():.2f}",
            f"  Overthinking: {inner_state.overthinking_level:.2f}",
            f"  Focus: {inner_state.current_focus}",
            f"  Awakened: {self._awakened}",
            "",
            "🎭 DILETTANTES (Expert Routing):",
            f"  Active Dilettante: {last_dilettante.value if last_dilettante else 'none'}",
            f"  (All experts are amateurs here)",
            "",
            "😈 BEHAVIOR:",
            f"  Avg Coherence: {behavior_metrics.avg_coherence:.2f}",
            f"  Conversations: {behavior_metrics.conversation_count}",
            f"  Mockery Probability: {behavior_metrics.compute_mockery_probability():.1%}",
            f"  Current Mood: {behavior_metrics.mood:.2f}",
            "",
            "🧮 MATHBRAIN:",
            f"  State: {self._mathbrain.get_state().active_expert.value}",
            f"  Trauma Level: {self._mathbrain.get_state().trauma_level:.2f}",
            f"  Sarcasm Debt: {self._mathbrain.get_state().sarcasm_debt:.2f}",
            "",
            "😈 ANTISANTA:",
            f"  Total Recalls: {self._antisanta.total_recalls}",
            f"  Chaos Recalls: {self._antisanta.chaos_recalls}",
            "",
            "═" * 50,
        ]
        
        return "\n".join(lines)


# Convenience function
async def create_consciousness(
    model,
    tokenizer,
    awaken: bool = True
) -> DubrovskyConsciousness:
    """Create and optionally awaken a consciousness instance."""
    consciousness = DubrovskyConsciousness(model, tokenizer)
    await consciousness.__aenter__()
    if awaken:
        await consciousness.awaken()
    return consciousness
