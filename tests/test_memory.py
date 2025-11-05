"""
Tests for Step 12: Memory Compression & Management
"""

# does not meet <100ms performance target

import pytest
import json
from datetime import datetime, timedelta

from socialsim.agents.memory import (
    MemoryItem,
    CompressedMemory,
    LongTermMemory,
    HierarchicalMemory,
    MemoryCompressor
)


# ============================================================================
# Test MemoryItem
# ============================================================================

class TestMemoryItem:
    """Tests for MemoryItem class."""
    
    def test_memory_item_creation(self):
        """Test creating a memory item."""
        content = {'step': 1, 'action': 'move', 'target': 'park'}
        item = MemoryItem(content, importance=0.7)
        
        assert item.content == content
        assert item.importance == 0.7
        assert item.access_count == 0
        assert isinstance(item.timestamp, datetime)
    
    def test_memory_item_access_tracking(self):
        """Test that access is tracked."""
        item = MemoryItem({'test': 'data'})
        
        initial_count = item.access_count
        initial_time = item.last_accessed
        
        item.access()
        
        assert item.access_count == initial_count + 1
        assert item.last_accessed >= initial_time
    
    def test_memory_item_serialization(self):
        """Test memory item serialization."""
        content = {'step': 5, 'actions': ['eat', 'rest']}
        item = MemoryItem(content, importance=0.8)
        item.access()
        
        data = item.to_dict()
        
        assert data['content'] == content
        assert data['importance'] == 0.8
        assert data['access_count'] == 1
        assert 'timestamp' in data
        assert 'last_accessed' in data
    
    def test_memory_item_deserialization(self):
        """Test memory item deserialization."""
        original = MemoryItem({'test': 'data'}, importance=0.6)
        original.access()
        
        data = original.to_dict()
        restored = MemoryItem.from_dict(data)
        
        assert restored.content == original.content
        assert restored.importance == original.importance
        assert restored.access_count == original.access_count


# ============================================================================
# Test CompressedMemory
# ============================================================================

class TestCompressedMemory:
    """Tests for CompressedMemory class."""
    
    def test_compressed_memory_creation(self):
        """Test creating compressed memory."""
        memories = [
            MemoryItem({'step': i, 'action': 'test'})
            for i in range(10)
        ]
        
        compressed = CompressedMemory(memories)
        
        assert compressed.memory_count == 10
        assert isinstance(compressed.summary, dict)
        assert isinstance(compressed.compressed_data, str)
    
    def test_compression_creates_summary(self):
        """Test that compression creates summary."""
        memories = [
            MemoryItem({'step': i, 'actions': ['move']})
            for i in range(5)
        ] + [
            MemoryItem({'step': i, 'actions': ['eat']})
            for i in range(5, 8)
        ]
        
        compressed = CompressedMemory(memories)
        summary = compressed.summary
        
        assert 'action_counts' in summary
        assert summary['action_counts']['move'] == 5
        assert summary['action_counts']['eat'] == 3
        assert summary['memory_count'] == 8
    
    def test_compression_reduces_size(self):
        """Test that compression reduces size."""
        memories = [
            MemoryItem({
                'step': i,
                'actions': ['move', 'eat'],
                'perception': {'data': 'x' * 100},
                'decision': {'reasoning': 'y' * 100}
            })
            for i in range(10)
        ]
        
        # Original size
        original_size = sum(
            len(json.dumps(m.to_dict()).encode('utf-8'))
            for m in memories
        )
        
        # Compressed size
        compressed = CompressedMemory(memories)
        compressed_size = compressed.get_size()
        
        # Should be significantly smaller
        assert compressed_size < original_size
        print(f"\nCompression: {original_size} -> {compressed_size} bytes "
              f"({original_size/compressed_size:.1f}x)")
    
    def test_decompression_restores_data(self):
        """Test that decompression works."""
        memories = [
            MemoryItem({'step': i, 'actions': [f'action_{i}']})
            for i in range(5)
        ]
        
        compressed = CompressedMemory(memories)
        decompressed = compressed.decompress()
        
        assert len(decompressed) == 5
        for i, mem in enumerate(decompressed):
            assert mem['step'] == i
    
    def test_compressed_memory_serialization(self):
        """Test serialization of compressed memory."""
        memories = [MemoryItem({'step': i}) for i in range(10)]
        compressed = CompressedMemory(memories)
        
        data = compressed.to_dict()
        
        assert 'created_at' in data
        assert 'memory_count' in data
        assert 'summary' in data
        assert 'compressed_data' in data
    
    def test_compressed_memory_deserialization(self):
        """Test deserialization of compressed memory."""
        memories = [MemoryItem({'step': i}) for i in range(10)]
        original = CompressedMemory(memories)
        
        data = original.to_dict()
        restored = CompressedMemory.from_dict(data)
        
        assert restored.memory_count == original.memory_count
        assert restored.compressed_data == original.compressed_data


# ============================================================================
# Test LongTermMemory
# ============================================================================

class TestLongTermMemory:
    """Tests for LongTermMemory class."""
    
    def test_long_term_memory_creation(self):
        """Test creating long-term memory."""
        summary = "Agent performed 50 actions over 2 days"
        ltm = LongTermMemory(summary, source_memories=50)
        
        assert ltm.summary == summary
        assert ltm.source_memories == 50
        assert isinstance(ltm.created_at, datetime)
    
    def test_long_term_memory_serialization(self):
        """Test long-term memory serialization."""
        ltm = LongTermMemory("Test summary", 100)
        
        data = ltm.to_dict()
        
        assert data['summary'] == "Test summary"
        assert data['source_memories'] == 100
        assert 'created_at' in data
    
    def test_long_term_memory_deserialization(self):
        """Test long-term memory deserialization."""
        original = LongTermMemory("Original summary", 75)
        
        data = original.to_dict()
        restored = LongTermMemory.from_dict(data)
        
        assert restored.summary == original.summary
        assert restored.source_memories == original.source_memories


# ============================================================================
# Test HierarchicalMemory
# ============================================================================

class TestHierarchicalMemory:
    """Tests for HierarchicalMemory system."""
    
    def test_hierarchical_memory_initialization(self):
        """Test creating hierarchical memory."""
        memory = HierarchicalMemory(
            working_memory_size=10,
            short_term_size=5,
            compression_ratio=10
        )
        
        assert len(memory.working_memory) == 0
        assert len(memory.short_term_memory) == 0
        assert len(memory.long_term_memory) == 0
    
    def test_add_to_working_memory(self):
        """Test adding experiences to working memory."""
        memory = HierarchicalMemory(working_memory_size=5)
        
        for i in range(3):
            memory.add({'step': i, 'action': 'test'})
        
        assert len(memory.working_memory) == 3
        assert memory.stats['total_added'] == 3
    
    def test_automatic_compression_to_short_term(self):
        """Test automatic compression when working memory full."""
        memory = HierarchicalMemory(
            working_memory_size=10,
            compression_ratio=5
        )
        
        # Fill working memory
        for i in range(10):
            memory.add({'step': i})
        
        assert len(memory.working_memory) == 10
        assert len(memory.short_term_memory) == 0
        
        # Add one more - should trigger compression
        memory.add({'step': 10})
        
        assert len(memory.working_memory) <= 10
        assert len(memory.short_term_memory) > 0
        assert memory.stats['compressions'] > 0
    
    def test_automatic_summarization_to_long_term(self):
        """Test automatic summarization when short-term full."""
        memory = HierarchicalMemory(
            working_memory_size=5,
            short_term_size=2,
            compression_ratio=5
        )
        
        # Add enough to fill working and short-term
        for i in range(25):  # Will trigger multiple compressions
            memory.add({'step': i, 'actions': ['test']})
        
        # Should have moved some to long-term
        assert len(memory.long_term_memory) > 0
        assert memory.stats['summarizations'] > 0
    
    def test_query_recent_memories(self):
        """Test querying recent memories."""
        memory = HierarchicalMemory(working_memory_size=20)
        
        # Add memories
        for i in range(15):
            memory.add({'step': i, 'value': i * 10})
        
        # Query recent
        recent = memory.query(n_recent=5)
        
        assert len(recent) == 5
        # Should be most recent (reverse order)
        assert recent[0]['step'] == 14
        assert recent[4]['step'] == 10
    
    def test_query_with_decompression(self):
        """Test querying with short-term decompression."""
        memory = HierarchicalMemory(
            working_memory_size=5,
            compression_ratio=5
        )
        
        # Add enough to trigger compression
        for i in range(15):
            memory.add({'step': i})
        
        # Query with decompression
        recent = memory.query(n_recent=12, include_compressed=True)
        
        # Should get from both working and short-term
        assert len(recent) > len(memory.working_memory)
    
    def test_memory_statistics(self):
        """Test memory statistics."""
        memory = HierarchicalMemory(working_memory_size=10)
        
        for i in range(5):
            memory.add({'step': i})
        
        stats = memory.get_summary()
        
        assert stats['working_memory_count'] == 5
        assert stats['total_memories_stored'] == 5
        assert 'memory_usage_bytes' in stats
        assert stats['memory_usage_bytes']['total'] > 0
    
    def test_compression_ratio_calculation(self):
        """Test compression ratio calculation."""
        memory = HierarchicalMemory(
            working_memory_size=10,
            compression_ratio=10
        )
        
        # Add memories to trigger compression
        for i in range(25):
            memory.add({
                'step': i,
                'action': 'test',
                'data': 'x' * 100  # Add bulk
            })
        
        ratio = memory.get_compression_ratio()
        
        # Should have some compression
        assert ratio > 1.0
        print(f"\nCompression ratio achieved: {ratio:.1f}x")
    
    def test_memory_clear(self):
        """Test clearing memory."""
        memory = HierarchicalMemory()
        
        for i in range(20):
            memory.add({'step': i})
        
        memory.clear()
        
        assert len(memory.working_memory) == 0
        assert len(memory.short_term_memory) == 0
        assert len(memory.long_term_memory) == 0
    
    def test_hierarchical_memory_serialization(self):
        """Test serializing hierarchical memory."""
        memory = HierarchicalMemory(working_memory_size=10)
        
        for i in range(15):
            memory.add({'step': i, 'action': f'action_{i}'})
        
        data = memory.to_dict()
        
        assert 'working_memory' in data
        assert 'short_term_memory' in data
        assert 'long_term_memory' in data
        assert 'stats' in data
        assert 'config' in data
    
    def test_hierarchical_memory_deserialization(self):
        """Test deserializing hierarchical memory."""
        original = HierarchicalMemory(
            working_memory_size=10,
            short_term_size=5
        )
        
        for i in range(20):
            original.add({'step': i})
        
        data = original.to_dict()
        restored = HierarchicalMemory.from_dict(data)
        
        assert len(restored.working_memory) == len(original.working_memory)
        assert len(restored.short_term_memory) == len(original.short_term_memory)
        assert restored.stats['total_added'] == original.stats['total_added']
    
    def test_memory_len(self):
        """Test __len__ returns total memories."""
        memory = HierarchicalMemory(
            working_memory_size=5,
            short_term_size=2,
            compression_ratio=5
        )
        
        # Add memories
        for i in range(25):
            memory.add({'step': i})
        
        total = len(memory)
        
        # Should account for all tiers
        assert total == 25


# ============================================================================
# Test MemoryCompressor
# ============================================================================

class TestMemoryCompressor:
    """Tests for MemoryCompressor utility."""
    
    def test_compress_batch_light(self):
        """Test light compression."""
        memories = [
            {
                'step': i,
                'actions': ['move'],
                'perception': {'test': 'data'},
                'decision': {'reasoning': 'test'},
                'extra': 'field'
            }
            for i in range(10)
        ]
        
        compressed = MemoryCompressor.compress_batch(memories, compression_level=1)
        
        # Should be valid JSON
        parsed = json.loads(compressed)
        assert len(parsed) == 10
        
        # Should have some fields
        assert 'step' in parsed[0]
        assert 'actions' in parsed[0]
    
    def test_compress_batch_medium(self):
        """Test medium compression."""
        memories = [{'step': i, 'actions': ['test'], 'extra': 'data'} for i in range(10)]
        
        compressed = MemoryCompressor.compress_batch(memories, compression_level=2)
        parsed = json.loads(compressed)
        
        # Should have minimal fields
        assert 'step' in parsed[0]
        assert 'actions' in parsed[0]
        assert 'extra' not in parsed[0]
    
    def test_compress_batch_heavy(self):
        """Test heavy compression."""
        memories = [
            {'step': i, 'actions': ['move', 'eat']}
            for i in range(10)
        ]
        
        compressed = MemoryCompressor.compress_batch(memories, compression_level=3)
        parsed = json.loads(compressed)
        
        # Should be heavily compressed (action summary)
        assert len(parsed) == 1
        assert 'action_summary' in parsed[0]
    
    def test_compression_ratio_estimation(self):
        """Test compression ratio estimation."""
        memories = [
            {'step': i, 'data': 'x' * 100}
            for i in range(20)
        ]
        
        compressed = MemoryCompressor.compress_batch(memories, compression_level=2)
        ratio = MemoryCompressor.estimate_compression_ratio(memories, compressed)
        
        assert ratio > 1.0  # Should have compression
        print(f"\nEstimated compression: {ratio:.1f}x")


# ============================================================================
# Performance Tests
# ============================================================================

class TestMemoryPerformance:
    """Performance tests for memory system."""
    
    def test_add_performance(self):
        """Test that adding memories is fast."""
        import time
        
        memory = HierarchicalMemory(working_memory_size=100)
        
        start = time.time()
        for i in range(1000):
            memory.add({'step': i, 'action': 'test'})
        duration = time.time() - start
        
        avg_time_us = (duration / 1000) * 1000000
        
        assert avg_time_us < 100  # <100 microseconds per add
        print(f"\nAdd performance: {avg_time_us:.1f} microseconds/memory")
    
    def test_query_performance(self):
        """Test that querying is fast."""
        import time
        
        memory = HierarchicalMemory(working_memory_size=100)
        
        for i in range(100):
            memory.add({'step': i})
        
        start = time.time()
        for _ in range(1000):
            memory.query(n_recent=10)
        duration = time.time() - start
        
        avg_time_us = (duration / 1000) * 1000000
        
        assert avg_time_us < 100  # <100 microseconds per query
        print(f"\nQuery performance: {avg_time_us:.1f} microseconds/query")
    
    def test_memory_footprint_reduction(self):
        """Test that memory footprint is reduced."""
        import sys
        
        # Simple list (baseline)
        simple_list = []
        for i in range(1000):
            simple_list.append({
                'step': i,
                'action': 'test',
                'data': 'x' * 100
            })
        
        simple_size = sys.getsizeof(simple_list) + sum(
            sys.getsizeof(item) for item in simple_list
        )
        
        # Hierarchical memory
        memory = HierarchicalMemory(
            working_memory_size=10,
            short_term_size=10,
            compression_ratio=10
        )
        
        for i in range(1000):
            memory.add({
                'step': i,
                'action': 'test',
                'data': 'x' * 100
            })
        
        summary = memory.get_summary()
        hierarchical_size = summary['memory_usage_bytes']['total']
        
        reduction_ratio = simple_size / hierarchical_size
        
        print(f"\nMemory reduction: {simple_size} -> {hierarchical_size} bytes "
              f"({reduction_ratio:.1f}x)")
        
        # Should have significant reduction
        assert hierarchical_size < simple_size


# ============================================================================
# Integration Tests
# ============================================================================

class TestMemoryIntegration:
    """Integration tests for memory system."""
    
    def test_realistic_agent_memory_usage(self):
        """Test memory with realistic agent usage pattern."""
        memory = HierarchicalMemory(
            working_memory_size=10,
            short_term_size=10,
            compression_ratio=10
        )
        
        # Simulate 1000 agent steps
        for step in range(1000):
            memory.add({
                'step': step,
                'actions': ['move', 'eat'] if step % 10 == 0 else ['rest'],
                'perception': {'nearby': []},
                'decision': {'reasoning': f'Step {step}'}
            })
        
        # Check memory usage
        summary = memory.get_summary()
        
        assert summary['total_memories_stored'] == 1000
        assert len(memory) == 1000
        
        # Should have all three tiers active
        assert summary['working_memory_count'] > 0
        assert summary['short_term_blocks'] > 0
        assert summary['long_term_summaries'] > 0
        
        # Memory should be compressed
        total_bytes = summary['memory_usage_bytes']['total']
        assert total_bytes < 50000  # <50KB for 1000 memories
        
        print(f"\n1000 memories stored in {total_bytes/1024:.1f} KB")
        print(f"Compression ratio: {memory.get_compression_ratio():.1f}x")
    
    def test_memory_across_simulation_restart(self):
        """Test memory serialization for simulation restart."""
        # Create and populate memory
        memory1 = HierarchicalMemory()
        
        for i in range(50):
            memory1.add({'step': i, 'action': f'action_{i}'})
        
        # Serialize
        data = memory1.to_dict()
        
        # "Restart" - deserialize
        memory2 = HierarchicalMemory.from_dict(data)
        
        # Should preserve state
        assert len(memory2) == len(memory1)
        assert memory2.stats['total_added'] == memory1.stats['total_added']
        
        # Should be able to continue adding
        memory2.add({'step': 50, 'action': 'new_action'})
        assert memory2.stats['total_added'] == 51


if __name__ == "__main__":
    pytest.main([__file__, "-v"])