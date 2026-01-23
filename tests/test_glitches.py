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
    
    print("=" * 60)
    print("🎉 ALL GLITCHES TESTS PASSED!")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(run_all_glitches_tests())
