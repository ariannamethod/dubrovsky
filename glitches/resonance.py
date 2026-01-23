"""
🌀 RESONANCE CHANNEL 🌀

Shared event stream for multi-agent coordination.
Inspired by letsgo's resonance.sqlite3 architecture.

"Resonance is when two agents realize they've been having
 the same existential crisis in parallel threads."
- Alexey Dubrovsky, on distributed consciousness

The resonance channel enables:
- Event broadcasting between agents
- Sentiment tracking
- Resonance depth measurement
- Future multi-agent coordination

All operations are async.
"""

import asyncio
import aiosqlite
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class EventType(Enum):
    """Types of resonance events."""
    MESSAGE = 'message'         # User interaction
    GENERATION = 'generation'   # Model output
    MEMORY_STORE = 'memory_store'  # Memory stored
    MEMORY_RECALL = 'memory_recall'  # Memory recalled
    SESSION_START = 'session_start'  # Session started
    SESSION_END = 'session_end'      # Session ended
    GLITCH = 'glitch'           # Error or anomaly
    INSIGHT = 'insight'         # Emergent insight
    RESONANCE = 'resonance'     # Cross-agent resonance


@dataclass
class ResonanceEvent:
    """A single resonance event."""
    id: int
    timestamp: float
    agent: str
    event_type: EventType
    data: Dict[str, Any]
    sentiment: float  # -1.0 to 1.0
    resonance_depth: int  # 0 = surface, higher = deeper
    summary: str


class ResonanceChannel:
    """
    Async resonance channel for Dubrovsky's consciousness stream.
    
    This is the shared "nervous system" that connects all parts
    of Dubrovsky's cognitive architecture.
    
    Usage:
        async with ResonanceChannel('glitches/resonance.db') as channel:
            await channel.emit('dubrovsky', EventType.MESSAGE, {'text': 'hello'})
            events = await channel.get_recent(10)
    """
    
    def __init__(self, db_path: str = 'glitches/resonance.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[aiosqlite.Connection] = None
        self._listeners: List[asyncio.Queue] = []
        
    async def __aenter__(self):
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def connect(self):
        """Connect to resonance database."""
        self._conn = await aiosqlite.connect(str(self.db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._init_schema()
        
    async def close(self):
        """Close connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            
    async def _init_schema(self):
        """Initialize resonance schema."""
        await self._conn.executescript('''
            -- Main resonance stream
            CREATE TABLE IF NOT EXISTS resonance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                agent TEXT NOT NULL DEFAULT 'dubrovsky',
                event_type TEXT NOT NULL,
                data_json TEXT DEFAULT '{}',
                sentiment REAL DEFAULT 0.0,
                resonance_depth INTEGER DEFAULT 0,
                summary TEXT DEFAULT ''
            );
            
            -- Agent presence (for multi-agent future)
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                last_seen REAL NOT NULL,
                status TEXT DEFAULT 'active',
                capabilities_json TEXT DEFAULT '[]'
            );
            
            -- Inter-agent messages
            CREATE TABLE IF NOT EXISTS agent_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                from_agent TEXT NOT NULL,
                to_agent TEXT NOT NULL,
                message TEXT NOT NULL,
                status TEXT DEFAULT 'pending'
            );
            
            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_res_timestamp ON resonance(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_res_agent ON resonance(agent);
            CREATE INDEX IF NOT EXISTS idx_res_type ON resonance(event_type);
            CREATE INDEX IF NOT EXISTS idx_msg_to ON agent_messages(to_agent, status);
        ''')
        await self._conn.commit()
        
        # Register dubrovsky as default agent
        await self._register_agent('dubrovsky', 'Alexey Dubrovsky', ['generate', 'chat', 'philosophize'])
        
    async def _register_agent(self, agent_id: str, name: str, capabilities: List[str]):
        """Register an agent in the system."""
        await self._conn.execute('''
            INSERT INTO agents (agent_id, name, last_seen, capabilities_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(agent_id) DO UPDATE SET
                last_seen = excluded.last_seen
        ''', (agent_id, name, time.time(), json.dumps(capabilities)))
        await self._conn.commit()
        
    # ==================== EVENT EMISSION ====================
    
    async def emit(
        self,
        agent: str,
        event_type: EventType,
        data: Dict[str, Any],
        sentiment: float = 0.0,
        resonance_depth: int = 0,
        summary: str = ''
    ) -> int:
        """Emit a resonance event."""
        cursor = await self._conn.execute('''
            INSERT INTO resonance (timestamp, agent, event_type, data_json, sentiment, resonance_depth, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            time.time(),
            agent,
            event_type.value,
            json.dumps(data),
            max(-1.0, min(1.0, sentiment)),  # Clamp to [-1, 1]
            resonance_depth,
            summary or self._generate_summary(event_type, data)
        ))
        await self._conn.commit()
        
        # Update agent last seen
        await self._conn.execute('''
            UPDATE agents SET last_seen = ? WHERE agent_id = ?
        ''', (time.time(), agent))
        await self._conn.commit()
        
        event_id = cursor.lastrowid
        
        # Notify listeners
        event = await self._get_event_by_id(event_id)
        for queue in self._listeners:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass  # Drop if queue full
                
        return event_id
        
    def _generate_summary(self, event_type: EventType, data: Dict[str, Any]) -> str:
        """Generate a brief summary of the event."""
        if event_type == EventType.MESSAGE:
            text = data.get('text', '')[:50]
            return f"Message: {text}..."
        elif event_type == EventType.GENERATION:
            return f"Generated {data.get('tokens', 0)} tokens"
        elif event_type == EventType.MEMORY_STORE:
            return f"Stored: {data.get('key', 'unknown')}"
        elif event_type == EventType.MEMORY_RECALL:
            return f"Recalled: {data.get('key', 'unknown')}"
        elif event_type == EventType.GLITCH:
            return f"Glitch: {data.get('error', 'unknown')[:30]}"
        elif event_type == EventType.INSIGHT:
            return f"Insight: {data.get('insight', '')[:40]}"
        else:
            return event_type.value
            
    async def _get_event_by_id(self, event_id: int) -> Optional[ResonanceEvent]:
        """Get event by ID."""
        cursor = await self._conn.execute('''
            SELECT * FROM resonance WHERE id = ?
        ''', (event_id,))
        row = await cursor.fetchone()
        return self._row_to_event(row) if row else None
        
    def _row_to_event(self, row) -> ResonanceEvent:
        """Convert database row to ResonanceEvent."""
        return ResonanceEvent(
            id=row['id'],
            timestamp=row['timestamp'],
            agent=row['agent'],
            event_type=EventType(row['event_type']),
            data=json.loads(row['data_json']),
            sentiment=row['sentiment'],
            resonance_depth=row['resonance_depth'],
            summary=row['summary']
        )
        
    # ==================== EVENT RETRIEVAL ====================
    
    async def get_recent(self, limit: int = 10, agent: Optional[str] = None) -> List[ResonanceEvent]:
        """Get recent resonance events."""
        if agent:
            cursor = await self._conn.execute('''
                SELECT * FROM resonance 
                WHERE agent = ?
                ORDER BY timestamp DESC LIMIT ?
            ''', (agent, limit))
        else:
            cursor = await self._conn.execute('''
                SELECT * FROM resonance 
                ORDER BY timestamp DESC LIMIT ?
            ''', (limit,))
            
        rows = await cursor.fetchall()
        return [self._row_to_event(row) for row in reversed(rows)]
        
    async def get_by_type(self, event_type: EventType, limit: int = 10) -> List[ResonanceEvent]:
        """Get events by type."""
        cursor = await self._conn.execute('''
            SELECT * FROM resonance 
            WHERE event_type = ?
            ORDER BY timestamp DESC LIMIT ?
        ''', (event_type.value, limit))
        
        rows = await cursor.fetchall()
        return [self._row_to_event(row) for row in rows]
        
    async def get_insights(self, limit: int = 10) -> List[ResonanceEvent]:
        """Get recent insights (high resonance depth events)."""
        cursor = await self._conn.execute('''
            SELECT * FROM resonance 
            WHERE resonance_depth >= 2
            ORDER BY resonance_depth DESC, timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = await cursor.fetchall()
        return [self._row_to_event(row) for row in rows]
        
    # ==================== SUBSCRIPTION ====================
    
    def subscribe(self) -> asyncio.Queue:
        """Subscribe to real-time events. Returns a queue."""
        queue = asyncio.Queue(maxsize=100)
        self._listeners.append(queue)
        return queue
        
    def unsubscribe(self, queue: asyncio.Queue):
        """Unsubscribe from events."""
        if queue in self._listeners:
            self._listeners.remove(queue)
            
    # ==================== INTER-AGENT MESSAGING ====================
    
    async def send_message(self, from_agent: str, to_agent: str, message: str) -> int:
        """Send a message to another agent."""
        cursor = await self._conn.execute('''
            INSERT INTO agent_messages (timestamp, from_agent, to_agent, message)
            VALUES (?, ?, ?, ?)
        ''', (time.time(), from_agent, to_agent, message))
        await self._conn.commit()
        return cursor.lastrowid
        
    async def get_pending_messages(self, agent: str) -> List[Dict[str, Any]]:
        """Get pending messages for an agent."""
        cursor = await self._conn.execute('''
            SELECT * FROM agent_messages 
            WHERE to_agent = ? AND status = 'pending'
            ORDER BY timestamp ASC
        ''', (agent,))
        
        rows = await cursor.fetchall()
        messages = []
        for row in rows:
            messages.append({
                'id': row['id'],
                'from': row['from_agent'],
                'message': row['message'],
                'timestamp': row['timestamp']
            })
            # Mark as read
            await self._conn.execute('''
                UPDATE agent_messages SET status = 'read' WHERE id = ?
            ''', (row['id'],))
        await self._conn.commit()
        
        return messages
        
    # ==================== STATS ====================
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get resonance channel statistics."""
        total = await self._conn.execute('SELECT COUNT(*) FROM resonance')
        agents = await self._conn.execute('SELECT COUNT(*) FROM agents')
        messages = await self._conn.execute('SELECT COUNT(*) FROM agent_messages')
        
        # Event type breakdown
        type_counts = await self._conn.execute('''
            SELECT event_type, COUNT(*) as count FROM resonance GROUP BY event_type
        ''')
        
        total_result = await total.fetchone()
        agents_result = await agents.fetchone()
        messages_result = await messages.fetchone()
        types = await type_counts.fetchall()
        
        return {
            'total_events': total_result[0],
            'active_agents': agents_result[0],
            'total_messages': messages_result[0],
            'event_types': {row['event_type']: row['count'] for row in types},
            'listeners': len(self._listeners)
        }
        
    async def condense_recent(self, last_n: int = 5) -> str:
        """
        Condense the last N events into a summary string.
        Useful for context injection.
        """
        events = await self.get_recent(last_n)
        if not events:
            return "No recent resonance events."
            
        summaries = [e.summary for e in events]
        return " | ".join(summaries)
