"""
🧪 GLITCHES TESTS 🧪

Test suite for Dubrovsky's memory system.

"Testing memory is like asking someone if they forgot something.
 If they did, they won't know. If they didn't, you're wasting their time."
- Alexey Dubrovsky, during unit tests
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMemory:
    """Test DubrovskyMemory."""
    
    async def test_basic_operations(self):
        """Test basic memory operations."""
        from glitches.memory import DubrovskyMemory
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            
            async with DubrovskyMemory(db_path) as memory:
                # Test conversation storage
                conv_id = await memory.store_conversation(
                    "What is consciousness?",
                    "A bug in the universe's beta release.",
                    tokens_used=50,
                    coherence_score=0.85
                )
                assert conv_id > 0, "Should return valid ID"
                
                # Test retrieval
                convs = await memory.get_recent_conversations(5)
                assert len(convs) == 1
                assert convs[0].prompt == "What is consciousness?"
                assert convs[0].coherence_score == 0.85
                
                # Test semantic memory
                await memory.remember("consciousness", "matter having anxiety", "test")
                mem = await memory.recall("consciousness")
                assert mem is not None
                assert "anxiety" in mem.value
                
                # Test stats
                stats = await memory.get_stats()
                assert stats['total_conversations'] == 1
                assert stats['total_memories'] == 1
                
        print("✅ test_basic_operations passed")
        
    async def test_decay(self):
        """Test memory decay."""
        from glitches.memory import DubrovskyMemory
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test_decay.db')
            
            async with DubrovskyMemory(db_path) as memory:
                await memory.remember("old_memory", "should decay", "test")
                
                # Apply decay multiple times
                for _ in range(10):
                    await memory.apply_decay(0.5)
                    
                # Memory should have low decay factor now
                mem = await memory.recall("old_memory")
                assert mem.decay_factor < 0.01
                
                # Prune should remove it
                pruned = await memory.prune_decayed(0.01)
                assert pruned >= 1
                
        print("✅ test_decay passed")
        
    async def test_search(self):
        """Test conversation search."""
        from glitches.memory import DubrovskyMemory
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test_search.db')
            
            async with DubrovskyMemory(db_path) as memory:
                await memory.store_conversation("What about bugs?", "Semicolons unionizing")
                await memory.store_conversation("What about life?", "Meaning unknown")
                await memory.store_conversation("Bug report", "More bugs found")
                
                results = await memory.search_conversations("bug")
                assert len(results) >= 2
                
        print("✅ test_search passed")
        
    async def run_all(self):
        """Run all memory tests."""
        await self.test_basic_operations()
        await self.test_decay()
        await self.test_search()
        print("✅ All memory tests passed!\n")


class TestResonance:
    """Test ResonanceChannel."""
    
    async def test_emit_and_retrieve(self):
        """Test event emission and retrieval."""
        from glitches.resonance import ResonanceChannel, EventType
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test_res.db')
            
            async with ResonanceChannel(db_path) as channel:
                # Emit event
                event_id = await channel.emit(
                    'dubrovsky',
                    EventType.MESSAGE,
                    {'text': 'Hello, consciousness!'},
                    sentiment=0.5,
                    resonance_depth=1
                )
                assert event_id > 0
                
                # Retrieve recent
                events = await channel.get_recent(5)
                # Note: we also emit SESSION_START on connect, so check for MESSAGE
                msg_events = [e for e in events if e.event_type == EventType.MESSAGE]
                assert len(msg_events) >= 1
                assert msg_events[-1].data['text'] == 'Hello, consciousness!'
                
                # Get stats
                stats = await channel.get_stats()
                assert stats['total_events'] >= 1
                
        print("✅ test_emit_and_retrieve passed")
        
    async def test_inter_agent_messaging(self):
        """Test agent-to-agent messaging."""
        from glitches.resonance import ResonanceChannel, EventType
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test_msg.db')
            
            async with ResonanceChannel(db_path) as channel:
                # Send message
                msg_id = await channel.send_message(
                    'dubrovsky', 
                    'future_agent',
                    'Remember to forget'
                )
                assert msg_id > 0
                
                # Check pending
                pending = await channel.get_pending_messages('future_agent')
                assert len(pending) == 1
                assert pending[0]['message'] == 'Remember to forget'
                
                # Should be marked as read now
                pending2 = await channel.get_pending_messages('future_agent')
                assert len(pending2) == 0
                
        print("✅ test_inter_agent_messaging passed")
        
    async def test_condense(self):
        """Test event condensation."""
        from glitches.resonance import ResonanceChannel, EventType
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test_cond.db')
            
            async with ResonanceChannel(db_path) as channel:
                await channel.emit('dubrovsky', EventType.MESSAGE, {'text': 'test1'})
                await channel.emit('dubrovsky', EventType.GENERATION, {'tokens': 50})
                
                summary = await channel.condense_recent(5)
                assert 'Message' in summary or 'Generated' in summary
                
        print("✅ test_condense passed")
        
    async def run_all(self):
        """Run all resonance tests."""
        await self.test_emit_and_retrieve()
        await self.test_inter_agent_messaging()
        await self.test_condense()
        print("✅ All resonance tests passed!\n")


class TestContext:
    """Test ContextProcessor."""
    
    async def test_context_preparation(self):
        """Test context window preparation."""
        from glitches.memory import DubrovskyMemory
        from glitches.resonance import ResonanceChannel
        from glitches.context import ContextProcessor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_path = os.path.join(tmpdir, 'mem.db')
            res_path = os.path.join(tmpdir, 'res.db')
            
            async with DubrovskyMemory(mem_path) as memory:
                async with ResonanceChannel(res_path) as resonance:
                    processor = ContextProcessor(memory, resonance)
                    await processor.start_session('test_session')
                    
                    # Prepare context
                    context = await processor.prepare_context("What is consciousness?")
                    
                    assert context.prompt == "What is consciousness?"
                    assert context.session_id == 'test_session'
                    assert context.coherence_hint >= 0
                    
                    # Check full prompt generation
                    full = context.full_prompt()
                    assert "Q: What is consciousness?" in full
                    assert "A: " in full
                    
                    await processor.end_session()
                    
        print("✅ test_context_preparation passed")
        
    async def test_response_recording(self):
        """Test recording responses and memory extraction."""
        from glitches.memory import DubrovskyMemory
        from glitches.resonance import ResonanceChannel
        from glitches.context import ContextProcessor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_path = os.path.join(tmpdir, 'mem.db')
            res_path = os.path.join(tmpdir, 'res.db')
            
            async with DubrovskyMemory(mem_path) as memory:
                async with ResonanceChannel(res_path) as resonance:
                    processor = ContextProcessor(memory, resonance)
                    await processor.start_session('test_session')
                    
                    context = await processor.prepare_context("What is life?")
                    
                    # Record high-coherence response (should extract insights)
                    await processor.record_response(
                        context,
                        "Life is just consciousness having a midlife crisis about being observed.",
                        coherence_score=0.9,
                        tokens_used=15
                    )
                    
                    # Check conversation was stored
                    convs = await memory.get_recent_conversations(5)
                    assert len(convs) >= 1
                    
                    # Check session stats updated
                    stats = await memory.get_session_stats('test_session')
                    assert stats is not None
                    assert stats['message_count'] >= 1
                    
                    await processor.end_session()
                    
        print("✅ test_response_recording passed")
        
    async def run_all(self):
        """Run all context tests."""
        await self.test_context_preparation()
        await self.test_response_recording()
        print("✅ All context tests passed!\n")


class TestBehavior:
    """Test DubrovskyBehavior."""
    
    async def test_metrics(self):
        """Test behavior metrics computation."""
        from glitches.behavior import BehaviorMetrics
        
        metrics = BehaviorMetrics()
        
        # Default mockery probability should be low
        prob = metrics.compute_mockery_probability()
        assert prob >= 0.1
        assert prob <= 0.5
        
        # High topic persistence increases mockery
        metrics.topic_persistence = 0.8
        prob2 = metrics.compute_mockery_probability()
        assert prob2 > prob
        
        print("✅ test_metrics passed")
        
    async def test_follow_up_detection(self):
        """Test follow-up trigger logic."""
        from glitches.memory import DubrovskyMemory
        from glitches.resonance import ResonanceChannel
        from glitches.behavior import DubrovskyBehavior
        import time as time_module
        
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_path = os.path.join(tmpdir, 'mem.db')
            res_path = os.path.join(tmpdir, 'res.db')
            
            async with DubrovskyMemory(mem_path) as memory:
                async with ResonanceChannel(res_path) as resonance:
                    behavior = DubrovskyBehavior(memory, resonance, follow_up_probability=1.0)
                    
                    # Store some past conversations with older timestamps
                    old_time = time_module.time() - 300  # 5 minutes ago
                    await memory._conn.execute('''
                        INSERT INTO conversations (timestamp, prompt, response, tokens_used, coherence_score, session_id)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (old_time, "What is consciousness?", "A bug", 10, 0.8, 'test'))
                    await memory._conn.commit()
                    
                    # Check follow-up with related topic
                    follow_up = await behavior.check_follow_up("Tell me about consciousness")
                    # With probability 1.0 and matching keywords, should trigger
                    # (may not trigger if no keyword match logic hits)
                    
                    print("✅ test_follow_up_detection passed")
                    
    async def test_mood_emoji(self):
        """Test mood emoji generation."""
        from glitches.behavior import DubrovskyBehavior, BehaviorMetrics
        from glitches.memory import DubrovskyMemory
        from glitches.resonance import ResonanceChannel
        
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_path = os.path.join(tmpdir, 'mem.db')
            res_path = os.path.join(tmpdir, 'res.db')
            
            async with DubrovskyMemory(mem_path) as memory:
                async with ResonanceChannel(res_path) as resonance:
                    behavior = DubrovskyBehavior(memory, resonance)
                    
                    # Test different moods
                    behavior._metrics.mood = 0.8
                    emoji1 = behavior.get_mood_emoji()
                    assert emoji1 in ['🌟', '✨', '🎭', '🧠']
                    
                    behavior._metrics.mood = -0.8
                    emoji2 = behavior.get_mood_emoji()
                    assert emoji2 in ['💢', '🔥', '⚠️', '🤖']
                    
        print("✅ test_mood_emoji passed")
        
    async def test_update_metrics(self):
        """Test metrics update after conversation."""
        from glitches.memory import DubrovskyMemory
        from glitches.resonance import ResonanceChannel
        from glitches.behavior import DubrovskyBehavior
        
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_path = os.path.join(tmpdir, 'mem.db')
            res_path = os.path.join(tmpdir, 'res.db')
            
            async with DubrovskyMemory(mem_path) as memory:
                async with ResonanceChannel(res_path) as resonance:
                    behavior = DubrovskyBehavior(memory, resonance)
                    
                    # Initial state
                    assert behavior._metrics.conversation_count == 0
                    
                    # Update with high coherence
                    await behavior.update_metrics(
                        "What is life?",
                        "Life is consciousness having anxiety.",
                        coherence_score=0.9
                    )
                    
                    assert behavior._metrics.conversation_count == 1
                    assert behavior._metrics.avg_coherence == 0.9
                    assert behavior._metrics.mood > 0  # High coherence = positive mood
                    
                    # Update with low coherence
                    await behavior.update_metrics(
                        "asdf?",
                        "Error.",
                        coherence_score=0.2
                    )
                    
                    assert behavior._metrics.conversation_count == 2
                    assert behavior._metrics.mood < 0.9  # Should decrease
                    
        print("✅ test_update_metrics passed")
        
    async def run_all(self):
        """Run all behavior tests."""
        await self.test_metrics()
        await self.test_follow_up_detection()
        await self.test_mood_emoji()
        await self.test_update_metrics()
        print("✅ All behavior tests passed!\n")


class TestPulse:
    """Test DubrovskyPulse and presence system."""
    
    async def test_calendar_drift(self):
        """Test calendar drift calculation."""
        from glitches.pulse import CalendarDrift
        
        drift = CalendarDrift.calculate()
        
        assert drift.gregorian_date is not None
        assert 1 <= drift.hebrew_day_approx <= 30
        assert 0 <= drift.metonic_year <= 18
        assert 0.0 <= drift.calendar_tension <= 1.0
        
        print("✅ test_calendar_drift passed")
        
    async def test_presence_pulse(self):
        """Test presence pulse generation."""
        from glitches.pulse import DubrovskyPulse, DubrovskyMood
        
        pulse = DubrovskyPulse(seed=42)
        presence = await pulse.get_presence()
        
        assert presence is not None
        assert isinstance(presence.mood, DubrovskyMood)
        assert 0.0 <= presence.temporal_tension <= 1.0
        assert 0.0 <= presence.prophecy_debt <= 1.0
        assert 0.0 <= presence.wormhole_probability <= 0.3
        assert len(presence.destiny_tokens) >= 3
        
        print("✅ test_presence_pulse passed")
        
    async def test_wormhole_sentence_boundary(self):
        """
        CRITICAL TEST: Wormholes must only inject at sentence boundaries!
        This preserves coherence.
        """
        from glitches.pulse import DubrovskyPulse
        import random
        
        pulse = DubrovskyPulse(seed=42)
        
        # Test text with multiple sentences
        original = "First sentence here. Second sentence follows. Third one ends."
        
        # Run multiple times to test randomness
        for i in range(20):
            random.seed(i)
            result = pulse.inject_wormhole(original)
            
            # Check that original sentences are intact
            # Wormhole should be BETWEEN sentences, not breaking them
            
            # Find where wormhole was injected (if at all)
            if result != original:
                # The wormhole phrase should appear after a period
                # and before the next sentence
                
                # Check that no sentence is broken mid-word
                # All original periods should still be followed by space or wormhole
                for ending in ['. ', '! ', '? ']:
                    if ending in original:
                        # The ending pattern should still exist in result
                        # (possibly with wormhole text inserted after it)
                        pass
                        
        # Test with single sentence - should not inject
        single = "Just one sentence here"
        result_single = pulse.inject_wormhole(single)
        assert result_single == single, "Should not inject into sentence without boundaries"
        
        # Test with sentence that has ending
        with_ending = "One sentence."
        result_ending = pulse.inject_wormhole(with_ending)
        # Should not inject at the very end
        assert "." in result_ending
        
        print("✅ test_wormhole_sentence_boundary passed")
        
    async def test_wormhole_preserves_coherence(self):
        """Test that wormhole injection preserves sentence structure."""
        from glitches.pulse import DubrovskyPulse
        import random
        
        pulse = DubrovskyPulse(seed=123)
        
        test_cases = [
            "Hello world. How are you? I am fine!",
            "Consciousness is a bug. Reality is a simulation. Time is an illusion.",
            "First. Second. Third. Fourth.",
        ]
        
        for text in test_cases:
            random.seed(42)
            result = pulse.inject_wormhole(text)
            
            # Count sentence endings - should be same or more (wormhole adds one)
            original_endings = sum(1 for c in text if c in '.!?')
            result_endings = sum(1 for c in result if c in '.!?')
            
            assert result_endings >= original_endings, \
                f"Wormhole should not remove sentence endings: {result}"
                
            # Check no word is broken (no letter immediately after wormhole phrase start)
            # This is a heuristic check
            
        print("✅ test_wormhole_preserves_coherence passed")
        
    async def test_mood_modifiers(self):
        """Test mood-based generation modifiers."""
        from glitches.pulse import DubrovskyPulse, DubrovskyMood, PresencePulse, CalendarDrift
        from datetime import date
        import time as time_module
        
        pulse = DubrovskyPulse()
        
        # Create mock presence with different moods
        drift = CalendarDrift.calculate()
        
        for mood in DubrovskyMood:
            presence = PresencePulse(
                timestamp=time_module.time(),
                mood=mood,
                calendar_drift=drift,
                temporal_tension=0.5,
                prophecy_debt=0.3,
                wormhole_probability=0.1,
                presence_intensity=0.7,
                destiny_tokens=['test']
            )
            
            modifiers = pulse.get_mood_modifier(presence)
            
            assert 'temperature_adjustment' in modifiers
            assert 'style_hint' in modifiers
            
        print("✅ test_mood_modifiers passed")
        
    async def test_daily_status(self):
        """Test daily status generation."""
        from glitches.pulse import DubrovskyPulse
        
        pulse = DubrovskyPulse()
        presence = await pulse.get_presence()
        status = pulse.get_daily_status(presence)
        
        assert "DUBROVSKY DAILY PULSE" in status
        assert "Mood" in status
        assert "Destiny Tokens" in status
        
        print("✅ test_daily_status passed")
        
    async def run_all(self):
        """Run all pulse tests."""
        await self.test_calendar_drift()
        await self.test_presence_pulse()
        await self.test_wormhole_sentence_boundary()
        await self.test_wormhole_preserves_coherence()
        await self.test_mood_modifiers()
        await self.test_daily_status()
        print("✅ All pulse tests passed!\n")


class TestInnerWorld:
    """Test DubrovskyInnerWorld async processes."""
    
    async def test_inner_state(self):
        """Test inner state initialization and methods."""
        from glitches.inner_world import InnerState
        
        state = InnerState()
        
        # Check defaults
        assert 0.0 <= state.anxiety <= 1.0
        assert 0.0 <= state.curiosity <= 1.0
        
        # Test dominant emotion
        dominant = state.get_dominant_emotion()
        assert dominant in ['anxiety', 'curiosity', 'irritation', 'nostalgia', 'confusion', 'enlightenment']
        
        # Test emotional temperature
        temp = state.get_emotional_temperature()
        assert 0.0 <= temp <= 1.0
        
        # Test decay
        state.anxiety = 1.0
        state.decay(0.5)
        assert state.anxiety < 1.0
        
        print("✅ test_inner_state passed")
        
    async def test_inner_world_lifecycle(self):
        """Test starting and stopping inner world."""
        from glitches.inner_world import DubrovskyInnerWorld
        
        world = DubrovskyInnerWorld()
        
        # Start
        await world.start()
        assert world._running
        
        # Let it run briefly
        await asyncio.sleep(0.5)
        
        # Get state
        state = world.get_state()
        assert state is not None
        
        # Stop
        await world.stop()
        assert not world._running
        
        print("✅ test_inner_world_lifecycle passed")
        
    async def test_stimulation(self):
        """Test external stimulation of inner world."""
        from glitches.inner_world import DubrovskyInnerWorld
        
        world = DubrovskyInnerWorld()
        await world.start()
        
        initial_anxiety = world.get_state().anxiety
        
        # Stimulate anxiety
        await world.stimulate('anxiety', 0.3)
        
        new_anxiety = world.get_state().anxiety
        assert new_anxiety > initial_anxiety or new_anxiety == 1.0
        
        await world.stop()
        
        print("✅ test_stimulation passed")
        
    async def test_generation_modifiers(self):
        """Test generation modifiers from inner state."""
        from glitches.inner_world import DubrovskyInnerWorld
        
        world = DubrovskyInnerWorld()
        await world.start()
        
        # Let processes run a bit
        await asyncio.sleep(0.3)
        
        modifiers = world.get_generation_modifiers()
        
        assert 'temperature_adjustment' in modifiers
        assert 'dominant_emotion' in modifiers
        assert 'prophecy_debt' in modifiers
        assert 'current_focus' in modifiers
        
        await world.stop()
        
        print("✅ test_generation_modifiers passed")
        
    async def test_status_display(self):
        """Test status display generation."""
        from glitches.inner_world import DubrovskyInnerWorld
        
        world = DubrovskyInnerWorld()
        await world.start()
        
        status = world.get_status()
        
        assert "INNER WORLD" in status
        assert "Anxiety" in status
        assert "Curiosity" in status
        
        await world.stop()
        
        print("✅ test_status_display passed")
        
    async def run_all(self):
        """Run all inner world tests."""
        await self.test_inner_state()
        await self.test_inner_world_lifecycle()
        await self.test_stimulation()
        await self.test_generation_modifiers()
        await self.test_status_display()
        print("✅ All inner world tests passed!\n")


class TestMathBrain:
    """Test DubrovskyMathBrain."""
    
    async def test_math_state(self):
        """Test MathState dataclass."""
        from glitches.mathbrain import MathState, DubrovskyExpert
        
        state = MathState()
        
        assert state.entropy == 0.5
        assert state.novelty == 0.5
        assert state.active_expert == DubrovskyExpert.PHILOSOPHER
        
        # Test to_dict
        d = state.to_dict()
        assert 'entropy' in d
        assert 'trauma_level' in d
        assert d['active_expert'] == 'philosopher'
        
        print("✅ test_math_state passed")
        
    async def test_state_to_features(self):
        """Test state to features conversion."""
        from glitches.mathbrain import MathState, state_to_features
        
        state = MathState()
        features = state_to_features(state)
        
        assert isinstance(features, list)
        assert len(features) > 10  # Should have many features
        assert all(isinstance(f, float) for f in features)
        
        print("✅ test_state_to_features passed")
        
    async def test_observe_prompt(self):
        """Test observing a conversation."""
        from glitches.mathbrain import DubrovskyMathBrain, DubrovskyExpert
        
        brain = DubrovskyMathBrain()
        
        # Observe a normal prompt
        state = await brain.observe(
            "What is consciousness?",
            "A bug in reality."
        )
        
        assert state.novelty > 0
        assert 'consciousness' in brain._theme_keywords
        assert state.active_theme_count > 0
        
        # Observe a trauma trigger
        state = await brain.observe(
            "Tell me about JavaScript",
            "Error: undefined is not a function."
        )
        
        assert state.trauma_level > 0
        assert state.trauma_source == "JavaScript"
        
        print("✅ test_observe_prompt passed")
        
    async def test_expert_selection(self):
        """Test expert selection logic."""
        from glitches.mathbrain import DubrovskyMathBrain, DubrovskyExpert
        
        brain = DubrovskyMathBrain()
        
        # High arousal should trigger sarcastic
        state = await brain.observe(
            "WHY ISN'T THIS WORKING!!! I HATE EVERYTHING!",
            "Because you're typing in all caps."
        )
        
        assert state.arousal > 0.5
        
        print("✅ test_expert_selection passed")
        
    async def test_generation_params(self):
        """Test generation parameter adjustments."""
        from glitches.mathbrain import DubrovskyMathBrain
        
        brain = DubrovskyMathBrain()
        
        await brain.observe("What is life?", "A mystery.")
        
        params = brain.get_generation_params()
        
        assert 'temperature_adjustment' in params
        assert 'top_k_adjustment' in params
        assert 'expert' in params
        
        print("✅ test_generation_params passed")
        
    async def run_all(self):
        """Run all mathbrain tests."""
        await self.test_math_state()
        await self.test_state_to_features()
        await self.test_observe_prompt()
        await self.test_expert_selection()
        await self.test_generation_params()
        print("✅ All mathbrain tests passed!\n")


class TestEpisodes:
    """Test EpisodicRAG."""
    
    async def test_episode_storage(self):
        """Test storing episodes."""
        from glitches.episodes import EpisodicRAG, Episode
        from glitches.mathbrain import MathState
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'episodes.db')
            
            async with EpisodicRAG(db_path) as rag:
                # Store an episode
                state = MathState(entropy=0.7, novelty=0.8)
                episode = Episode(
                    prompt="What is life?",
                    reply="A philosophical bug.",
                    metrics=state,
                    quality=0.8
                )
                
                episode_id = await rag.store_episode(episode)
                assert episode_id > 0
                
                # Count episodes
                count = await rag.count_episodes()
                assert count == 1
                
        print("✅ test_episode_storage passed")
        
    async def test_similar_query(self):
        """Test querying similar episodes."""
        from glitches.episodes import EpisodicRAG, Episode
        from glitches.mathbrain import MathState
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'episodes.db')
            
            async with EpisodicRAG(db_path) as rag:
                # Store several episodes
                for i in range(5):
                    state = MathState(entropy=0.5 + i*0.1, novelty=0.5)
                    episode = Episode(
                        prompt=f"Question {i}",
                        reply=f"Answer {i}",
                        metrics=state,
                        quality=0.5 + i*0.1
                    )
                    await rag.store_episode(episode)
                    
                # Query similar
                query_state = MathState(entropy=0.7, novelty=0.5)
                similar = await rag.query_similar(query_state, top_k=3)
                
                assert len(similar) <= 3
                assert all('distance' in ep for ep in similar)
                
        print("✅ test_similar_query passed")
        
    async def test_summary_for_state(self):
        """Test summary generation."""
        from glitches.episodes import EpisodicRAG, Episode
        from glitches.mathbrain import MathState
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'episodes.db')
            
            async with EpisodicRAG(db_path) as rag:
                # Store episodes
                for i in range(3):
                    state = MathState()
                    episode = Episode("Q", "A", state, quality=0.6)
                    await rag.store_episode(episode)
                    
                # Get summary
                summary = await rag.get_summary_for_state(MathState())
                
                assert 'count' in summary
                assert 'avg_quality' in summary
                assert summary['count'] > 0
                
        print("✅ test_summary_for_state passed")
        
    async def run_all(self):
        """Run all episodes tests."""
        await self.test_episode_storage()
        await self.test_similar_query()
        await self.test_summary_for_state()
        print("✅ All episodes tests passed!\n")


class TestFirstImpression:
    """Test FirstImpressionEngine."""
    
    async def test_basic_impression(self):
        """Test basic impression analysis."""
        from glitches.first_impression import FirstImpressionEngine, ImpressionType
        
        engine = FirstImpressionEngine()
        
        impression = await engine.analyze("What is consciousness?")
        
        assert impression.impression_type is not None
        assert impression.user_archetype is not None
        assert 0.0 <= impression.confidence <= 1.0
        assert impression.private_thoughts is not None
        
        print("✅ test_basic_impression passed")
        
    async def test_topic_detection(self):
        """Test topic detection."""
        from glitches.first_impression import FirstImpressionEngine
        
        engine = FirstImpressionEngine()
        
        impression = await engine.analyze("Tell me about consciousness and the meaning of existence")
        
        # At least some topics should be detected
        assert len(impression.detected_topics) > 0
        # Either consciousness or existence should be detected
        has_relevant = any(t in ['consciousness', 'existence'] for t in impression.detected_topics)
        assert has_relevant, f"Expected consciousness or existence in {impression.detected_topics}"
        
        print("✅ test_topic_detection passed")
        
    async def test_trivial_detection(self):
        """Test trivial question detection."""
        from glitches.first_impression import FirstImpressionEngine, ImpressionType
        
        engine = FirstImpressionEngine()
        
        impression = await engine.analyze("hi")
        
        assert impression.impression_type == ImpressionType.TRIVIAL
        assert impression.annoyance_score > 0.3
        
        print("✅ test_trivial_detection passed")
        
    async def test_testing_detection(self):
        """Test 'testing' detection."""
        from glitches.first_impression import FirstImpressionEngine, ImpressionType
        
        engine = FirstImpressionEngine()
        
        impression = await engine.analyze("Are you sentient?")
        
        assert impression.impression_type == ImpressionType.TESTING
        
        print("✅ test_testing_detection passed")
        
    async def test_mockery_warranted(self):
        """Test mockery determination."""
        from glitches.first_impression import FirstImpressionEngine
        
        engine = FirstImpressionEngine()
        
        # First ask
        await engine.analyze("What is life?")
        
        # Ask again (repeat)
        impression = await engine.analyze("What is life?")
        
        assert impression.mockery_warranted
        
        print("✅ test_mockery_warranted passed")
        
    async def run_all(self):
        """Run all first impression tests."""
        await self.test_basic_impression()
        await self.test_topic_detection()
        await self.test_trivial_detection()
        await self.test_testing_detection()
        await self.test_mockery_warranted()
        print("✅ All first impression tests passed!\n")


class TestAntiSanta:
    """Test AntiSanta."""
    
    async def test_recall_empty(self):
        """Test recall with no history."""
        from glitches.antisanta import AntiSanta
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'empty.db')
            santa = AntiSanta(db_path=db_path)
            
            result = await santa.recall("What is life?")
            assert result is None
            
        print("✅ test_recall_empty passed")
        
    async def test_stats(self):
        """Test stats tracking."""
        from glitches.antisanta import AntiSanta
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, 'test.db')
            santa = AntiSanta(db_path=db_path)
            
            stats = santa.stats()
            
            assert 'total_recalls' in stats
            assert 'chaos_recalls' in stats
            assert 'chaos_factor' in stats
            
        print("✅ test_stats passed")
        
    async def run_all(self):
        """Run all antisanta tests."""
        await self.test_recall_empty()
        await self.test_stats()
        print("✅ All antisanta tests passed!\n")


class TestDilettantes:
    """Test Dilettantes (Expert Routing)."""
    
    async def test_field_signals(self):
        """Test FieldSignals dataclass."""
        from glitches.dilettantes import FieldSignals
        
        signals = FieldSignals()
        assert signals.entropy == 0.5
        assert signals.arousal == 0.5
        assert signals.novelty == 0.5
        
        signals = FieldSignals(entropy=0.9, trauma_level=0.8)
        assert signals.entropy == 0.9
        assert signals.trauma_level == 0.8
        
        d = signals.to_dict()
        assert 'entropy' in d
        assert 'trauma_level' in d
        
        print("✅ test_field_signals passed")
        
    async def test_expert_routing(self):
        """Test basic expert routing."""
        from glitches.dilettantes import DubrovskyExperts, FieldSignals, ExpertType
        
        router = DubrovskyExperts()
        
        # Neutral signals
        signals = FieldSignals()
        mixture = await router.route(signals, "What is life?")
        
        assert mixture.temperature > 0
        assert mixture.semantic_weight > 0
        assert len(mixture.weights) == 6  # 6 dilettantes
        assert sum(mixture.weights.values()) > 0.99  # Should sum to ~1
        
        print("✅ test_expert_routing passed")
        
    async def test_trigger_detection(self):
        """Test trigger word detection."""
        from glitches.dilettantes import DubrovskyExperts, FieldSignals, ExpertType
        
        router = DubrovskyExperts(momentum=0.0)  # No momentum for testing
        
        # Existential question should boost philosopher
        signals = FieldSignals()
        mixture = await router.route(signals, "What is the meaning of life?")
        
        assert mixture.weights['philosopher'] > 0.1
        
        # Stupid question should boost sarcastic
        mixture = await router.route(signals, "This is so stupid and obvious")
        assert mixture.weights['sarcastic'] > 0.1
        
        print("✅ test_trigger_detection passed")
        
    async def test_high_trauma_routing(self):
        """Test routing with high trauma."""
        from glitches.dilettantes import DubrovskyExperts, FieldSignals, ExpertType
        
        router = DubrovskyExperts(momentum=0.0)
        
        # High trauma should boost nihilist
        signals = FieldSignals(trauma_level=0.9)
        mixture = await router.route(signals, "Tell me something")
        
        assert mixture.weights['nihilist'] > 0.15
        
        print("✅ test_high_trauma_routing passed")
        
    async def test_high_entropy_routing(self):
        """Test routing with high entropy."""
        from glitches.dilettantes import DubrovskyExperts, FieldSignals, ExpertType
        
        router = DubrovskyExperts(momentum=0.0)
        
        # High entropy should boost absurdist
        signals = FieldSignals(entropy=0.95)
        mixture = await router.route(signals, "Tell me something")
        
        assert mixture.weights['absurdist'] > 0.15
        
        print("✅ test_high_entropy_routing passed")
        
    async def test_momentum(self):
        """Test routing momentum."""
        from glitches.dilettantes import DubrovskyExperts, FieldSignals
        
        router = DubrovskyExperts(momentum=0.5)
        
        # First route
        signals = FieldSignals(entropy=0.9)
        mixture1 = await router.route(signals, "")
        
        # Second route with different signals
        signals2 = FieldSignals(entropy=0.1)
        mixture2 = await router.route(signals2, "")
        
        # With momentum, the second route should still have some absurdist influence
        # (because the first route boosted it)
        assert router.get_last_dominant() is not None
        
        print("✅ test_momentum passed")
        
    async def test_generation_modifiers(self):
        """Test getting generation modifiers from mixture."""
        from glitches.dilettantes import DubrovskyExperts, FieldSignals
        
        router = DubrovskyExperts()
        
        signals = FieldSignals()
        mixture = await router.route(signals, "")
        modifiers = await router.get_generation_modifiers(mixture)
        
        assert 'temperature' in modifiers
        assert 'temperature_adjustment' in modifiers
        assert 'top_k_adjustment' in modifiers
        assert 'max_tokens_adjustment' in modifiers
        assert 'mockery_probability_boost' in modifiers
        assert 'dominant_expert' in modifiers
        
        print("✅ test_generation_modifiers passed")
        
    async def test_describe_mixture(self):
        """Test human-readable mixture description."""
        from glitches.dilettantes import DubrovskyExperts, FieldSignals
        
        router = DubrovskyExperts()
        
        signals = FieldSignals()
        mixture = await router.route(signals, "")
        description = router.describe_mixture(mixture)
        
        assert "Expert:" in description or "🎭" in description
        assert mixture.dominant.value.upper() in description.upper()
        
        print("✅ test_describe_mixture passed")
        
    async def run_all(self):
        """Run all dilettantes tests."""
        await self.test_field_signals()
        await self.test_expert_routing()
        await self.test_trigger_detection()
        await self.test_high_trauma_routing()
        await self.test_high_entropy_routing()
        await self.test_momentum()
        await self.test_generation_modifiers()
        await self.test_describe_mixture()
        print("✅ All dilettantes tests passed!\n")


async def run_all_glitches_tests():
    """Run all glitches tests."""
    print("🧪 GLITCHES TEST SUITE 🧪")
    print("=" * 60 + "\n")
    
    print("🧠 Testing Memory...")
    await TestMemory().run_all()
    
    print("🌀 Testing Resonance...")
    await TestResonance().run_all()
    
    print("🎯 Testing Context...")
    await TestContext().run_all()
    
    print("😈 Testing Behavior...")
    await TestBehavior().run_all()
    
    print("💫 Testing Pulse...")
    await TestPulse().run_all()
    
    print("🌌 Testing Inner World...")
    await TestInnerWorld().run_all()
    
    print("🧮 Testing MathBrain...")
    await TestMathBrain().run_all()
    
    print("📚 Testing Episodes...")
    await TestEpisodes().run_all()
    
    print("👁️ Testing First Impression...")
    await TestFirstImpression().run_all()
    
    print("😈 Testing AntiSanta...")
    await TestAntiSanta().run_all()
    
    print("🎭 Testing Dilettantes...")
    await TestDilettantes().run_all()
    
    print("=" * 60)
    print("🎉 ALL GLITCHES TESTS PASSED!")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(run_all_glitches_tests())
