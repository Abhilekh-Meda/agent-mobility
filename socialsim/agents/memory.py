"""
Hierarchical memory system for Phase 2.

Implements three-tier memory to reduce memory footprint for 10,000+ agent simulations.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from collections import deque
import json
from loguru import logger


class MemoryItem:
    """Individual memory item with metadata."""
    
    def __init__(
        self,
        content: Dict[str, Any],
        timestamp: Optional[datetime] = None,
        importance: float = 0.5
    ):
        """Initialize memory item.
        
        Args:
            content: Memory content
            timestamp: When memory was created
            importance: Importance score (0-1)
        """
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.importance = importance
        self.access_count = 0
        self.last_accessed = self.timestamp
    
    def access(self):
        """Record memory access."""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory item."""
        return {
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'importance': self.importance,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """Deserialize memory item."""
        item = cls(
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            importance=data.get('importance', 0.5)
        )
        item.access_count = data.get('access_count', 0)
        item.last_accessed = datetime.fromisoformat(
            data.get('last_accessed', data['timestamp'])
        )
        return item


class CompressedMemory:
    """Compressed memory block for short-term storage."""
    
    def __init__(self, memories: List[MemoryItem]):
        """Initialize compressed memory.
        
        Args:
            memories: List of memories to compress
        """
        self.created_at = datetime.now()
        self.memory_count = len(memories)
        
        # Extract key information
        self.summary = self._create_summary(memories)
        
        # Store compressed data
        self.compressed_data = self._compress(memories)
    
    def _create_summary(self, memories: List[MemoryItem]) -> Dict[str, Any]:
        """Create summary of memories.
        
        Args:
            memories: Memories to summarize
            
        Returns:
            Summary dictionary
        """
        if not memories:
            return {}
        
        # Count action types
        actions = {}
        for mem in memories:
            action = mem.content.get('actions', ['unknown'])[0]
            actions[action] = actions.get(action, 0) + 1
        
        # Time range
        timestamps = [m.timestamp for m in memories]
        time_range = (min(timestamps), max(timestamps))
        
        return {
            'action_counts': actions,
            'time_range': (time_range[0].isoformat(), time_range[1].isoformat()),
            'memory_count': len(memories),
            'avg_importance': sum(m.importance for m in memories) / len(memories)
        }
    
    def _compress(self, memories: List[MemoryItem]) -> str:
        """Compress memories to string.
        
        Args:
            memories: Memories to compress
            
        Returns:
            Compressed JSON string
        """
        # Keep only essential data
        compressed = [
            {
                'actions': m.content.get('actions', []),
                'step': m.content.get('step', 0),
                'importance': m.importance
            }
            for m in memories
        ]
        
        return json.dumps(compressed, separators=(',', ':'))
    
    def decompress(self) -> List[Dict[str, Any]]:
        """Decompress memories.
        
        Returns:
            List of memory dictionaries
        """
        return json.loads(self.compressed_data)
    
    def get_size(self) -> int:
        """Get compressed size in bytes."""
        return len(self.compressed_data.encode('utf-8'))
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize compressed memory."""
        return {
            'created_at': self.created_at.isoformat(),
            'memory_count': self.memory_count,
            'summary': self.summary,
            'compressed_data': self.compressed_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompressedMemory':
        """Deserialize compressed memory."""
        compressed = cls.__new__(cls)
        compressed.created_at = datetime.fromisoformat(data['created_at'])
        compressed.memory_count = data['memory_count']
        compressed.summary = data['summary']
        compressed.compressed_data = data['compressed_data']
        return compressed


class LongTermMemory:
    """Ultra-compressed long-term memory with LLM summarization."""
    
    def __init__(self, summary: str, source_memories: int):
        """Initialize long-term memory.
        
        Args:
            summary: Text summary of memories
            source_memories: Number of memories summarized
        """
        self.summary = summary
        self.source_memories = source_memories
        self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize long-term memory."""
        return {
            'summary': self.summary,
            'source_memories': self.source_memories,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LongTermMemory':
        """Deserialize long-term memory."""
        ltm = cls(data['summary'], data['source_memories'])
        ltm.created_at = datetime.fromisoformat(data['created_at'])
        return ltm


class HierarchicalMemory:
    """Three-tier hierarchical memory system.
    
    Tiers:
    1. Working Memory: Full detail, last 10 items, in-memory
    2. Short-Term Memory: Compressed, last 100 items, ~10x compression
    3. Long-Term Memory: Summarized, unlimited items, ~100x compression
    """
    
    def __init__(
        self,
        working_memory_size: int = 10,
        short_term_size: int = 10,  # Number of compressed blocks
        compression_ratio: int = 10  # Items per compressed block
    ):
        """Initialize hierarchical memory.
        
        Args:
            working_memory_size: Size of working memory
            short_term_size: Number of short-term compressed blocks
            compression_ratio: Items per compression block
        """
        self.working_memory_size = working_memory_size
        self.short_term_size = short_term_size
        self.compression_ratio = compression_ratio
        
        # Working memory: Full fidelity, fast access
        self.working_memory: deque[MemoryItem] = deque(maxlen=working_memory_size)
        
        # Short-term memory: Compressed blocks
        self.short_term_memory: deque[CompressedMemory] = deque(maxlen=short_term_size)
        
        # Long-term memory: Summarized text
        self.long_term_memory: List[LongTermMemory] = []
        
        # Statistics
        self.stats = {
            'total_added': 0,
            'compressions': 0,
            'summarizations': 0,
            'working_memory_bytes': 0,
            'short_term_bytes': 0,
            'long_term_bytes': 0
        }
        
        logger.debug("HierarchicalMemory initialized")
    
    def add(
        self,
        experience: Dict[str, Any],
        importance: float = 0.5
    ) -> None:
        """Add new experience to memory.
        
        Args:
            experience: Experience data
            importance: Importance score (0-1)
        """
        # Create memory item
        memory_item = MemoryItem(experience, importance=importance)
        
        # Check if working memory is full
        if len(self.working_memory) >= self.working_memory_size:
            # Move oldest to short-term
            self._compress_to_short_term()
        
        # Add to working memory
        self.working_memory.append(memory_item)
        self.stats['total_added'] += 1
        
        self._update_stats()
    
    def _compress_to_short_term(self) -> None:
        """Compress working memory to short-term."""
        if len(self.working_memory) == 0:
            return
        
        # Take oldest items for compression
        items_to_compress = []
        for _ in range(min(self.compression_ratio, len(self.working_memory))):
            if self.working_memory:
                items_to_compress.append(self.working_memory.popleft())
        
        if not items_to_compress:
            return
        
        # Create compressed block
        compressed = CompressedMemory(items_to_compress)
        
        # Check if short-term is full
        if len(self.short_term_memory) >= self.short_term_size:
            # Move oldest to long-term
            self._summarize_to_long_term()
        
        # Add to short-term
        self.short_term_memory.append(compressed)
        self.stats['compressions'] += 1
        
        logger.debug(
            f"Compressed {len(items_to_compress)} memories to short-term "
            f"({compressed.get_size()} bytes)"
        )
    
    def _summarize_to_long_term(self) -> None:
        """Summarize short-term memory to long-term.
        
        Note: In Phase 2, uses rule-based summarization.
        TODO Phase 3: Use LLM for better summarization.
        """
        if len(self.short_term_memory) == 0:
            return
        
        # Take oldest compressed block
        compressed_block = self.short_term_memory.popleft()
        
        # Create simple summary (rule-based for now)
        summary = self._create_summary_text(compressed_block)
        
        # Add to long-term
        long_term = LongTermMemory(summary, compressed_block.memory_count)
        self.long_term_memory.append(long_term)
        self.stats['summarizations'] += 1
        
        logger.debug(
            f"Summarized {compressed_block.memory_count} memories to long-term "
            f"({len(summary)} chars)"
        )
    
    def _create_summary_text(self, compressed: CompressedMemory) -> str:
        """Create text summary from compressed memory.
        
        Args:
            compressed: Compressed memory block
            
        Returns:
            Summary text
        """
        summary_data = compressed.summary
        
        # Create readable summary
        action_counts = summary_data.get('action_counts', {})
        action_text = ", ".join([
            f"{count}x {action}" 
            for action, count in action_counts.items()
        ])
        
        time_range = summary_data.get('time_range', ('', ''))
        
        summary = (
            f"Period: {time_range[0][:10]} to {time_range[1][:10]}. "
            f"Actions: {action_text}. "
            f"Total memories: {summary_data.get('memory_count', 0)}."
        )
        
        return summary
    
    def query(
        self,
        n_recent: int = 10,
        include_compressed: bool = False
    ) -> List[Dict[str, Any]]:
        """Query recent memories.
        
        Args:
            n_recent: Number of recent memories to retrieve
            include_compressed: Whether to include short-term memories
            
        Returns:
            List of memory dictionaries
        """
        results = []
        
        # Get from working memory first
        for item in reversed(self.working_memory):
            if len(results) >= n_recent:
                break
            item.access()
            results.append(item.content)
        
        # If requested, decompress short-term
        if include_compressed and len(results) < n_recent:
            for compressed in reversed(self.short_term_memory):
                decompressed = compressed.decompress()
                for mem in reversed(decompressed):
                    if len(results) >= n_recent:
                        break
                    results.append(mem)
        
        return results[:n_recent]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get memory system summary.
        
        Returns:
            Summary statistics and content
        """
        return {
            'working_memory_count': len(self.working_memory),
            'short_term_blocks': len(self.short_term_memory),
            'long_term_summaries': len(self.long_term_memory),
            'total_memories_stored': self.stats['total_added'],
            'compression_stats': {
                'compressions': self.stats['compressions'],
                'summarizations': self.stats['summarizations']
            },
            'memory_usage_bytes': {
                'working': self.stats['working_memory_bytes'],
                'short_term': self.stats['short_term_bytes'],
                'long_term': self.stats['long_term_bytes'],
                'total': sum([
                    self.stats['working_memory_bytes'],
                    self.stats['short_term_bytes'],
                    self.stats['long_term_bytes']
                ])
            }
        }
    
    def _update_stats(self) -> None:
        """Update memory usage statistics."""
        # Working memory size
        self.stats['working_memory_bytes'] = sum(
            len(json.dumps(item.to_dict()).encode('utf-8'))
            for item in self.working_memory
        )
        
        # Short-term memory size
        self.stats['short_term_bytes'] = sum(
            compressed.get_size()
            for compressed in self.short_term_memory
        )
        
        # Long-term memory size
        self.stats['long_term_bytes'] = sum(
            len(ltm.summary.encode('utf-8'))
            for ltm in self.long_term_memory
        )
    
    def clear(self) -> None:
        """Clear all memory."""
        self.working_memory.clear()
        self.short_term_memory.clear()
        self.long_term_memory.clear()
        
        # Reset stats
        for key in ['working_memory_bytes', 'short_term_bytes', 'long_term_bytes']:
            self.stats[key] = 0
        
        logger.debug("Memory cleared")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize hierarchical memory.
        
        Returns:
            Serialized memory system
        """
        return {
            'working_memory': [item.to_dict() for item in self.working_memory],
            'short_term_memory': [comp.to_dict() for comp in self.short_term_memory],
            'long_term_memory': [ltm.to_dict() for ltm in self.long_term_memory],
            'stats': self.stats.copy(),
            'config': {
                'working_memory_size': self.working_memory_size,
                'short_term_size': self.short_term_size,
                'compression_ratio': self.compression_ratio
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HierarchicalMemory':
        """Deserialize hierarchical memory.
        
        Args:
            data: Serialized memory data
            
        Returns:
            Restored HierarchicalMemory instance
        """
        config = data.get('config', {})
        memory = cls(
            working_memory_size=config.get('working_memory_size', 10),
            short_term_size=config.get('short_term_size', 10),
            compression_ratio=config.get('compression_ratio', 10)
        )
        
        # Restore working memory
        for item_data in data.get('working_memory', []):
            memory.working_memory.append(MemoryItem.from_dict(item_data))
        
        # Restore short-term memory
        for comp_data in data.get('short_term_memory', []):
            memory.short_term_memory.append(CompressedMemory.from_dict(comp_data))
        
        # Restore long-term memory
        for ltm_data in data.get('long_term_memory', []):
            memory.long_term_memory.append(LongTermMemory.from_dict(ltm_data))
        
        # Restore stats
        if 'stats' in data:
            memory.stats.update(data['stats'])
        
        memory._update_stats()
        
        return memory
    
    def get_compression_ratio(self) -> float:
        """Calculate actual compression ratio achieved.
        
        Returns:
            Compression ratio (original size / compressed size)
        """
        if self.stats['total_added'] == 0:
            return 1.0
        
        # Estimate original size (before compression)
        avg_memory_size = 200  # bytes (rough estimate)
        original_size = self.stats['total_added'] * avg_memory_size
        
        # Current size
        current_size = sum([
            self.stats['working_memory_bytes'],
            self.stats['short_term_bytes'],
            self.stats['long_term_bytes']
        ])
        
        if current_size == 0:
            return 1.0
        
        return original_size / current_size
    
    def __len__(self) -> int:
        """Get total number of memories (across all tiers)."""
        compressed_count = sum(
            comp.memory_count for comp in self.short_term_memory
        )
        long_term_count = sum(
            ltm.source_memories for ltm in self.long_term_memory
        )
        
        return len(self.working_memory) + compressed_count + long_term_count
    
    def __str__(self) -> str:
        return (
            f"HierarchicalMemory(working={len(self.working_memory)}, "
            f"short_term={len(self.short_term_memory)} blocks, "
            f"long_term={len(self.long_term_memory)} summaries, "
            f"total={len(self)} memories)"
        )


class MemoryCompressor:
    """Utility class for memory compression operations."""
    
    @staticmethod
    def compress_batch(
        memories: List[Dict[str, Any]],
        compression_level: int = 1
    ) -> str:
        """Compress a batch of memories.
        
        Args:
            memories: List of memory dictionaries
            compression_level: 1=light, 2=medium, 3=heavy
            
        Returns:
            Compressed JSON string
        """
        if compression_level == 1:
            # Light: Remove some fields
            compressed = [
                {k: v for k, v in m.items() if k in ['step', 'actions', 'reasoning']}
                for m in memories
            ]
        elif compression_level == 2:
            # Medium: Keep only essential fields
            compressed = [
                {'step': m.get('step'), 'actions': m.get('actions', [])}
                for m in memories
            ]
        else:
            # Heavy: Just action counts
            action_counts = {}
            for m in memories:
                for action in m.get('actions', []):
                    action_counts[action] = action_counts.get(action, 0) + 1
            compressed = [{'action_summary': action_counts}]
        
        return json.dumps(compressed, separators=(',', ':'))
    
    @staticmethod
    def estimate_compression_ratio(
        original: List[Dict[str, Any]],
        compressed: str
    ) -> float:
        """Estimate compression ratio.
        
        Args:
            original: Original memories
            compressed: Compressed JSON string
            
        Returns:
            Compression ratio
        """
        original_size = len(json.dumps(original).encode('utf-8'))
        compressed_size = len(compressed.encode('utf-8'))
        
        if compressed_size == 0:
            return 1.0
        
        return original_size / compressed_size