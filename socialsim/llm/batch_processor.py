"""
Batch LLM processing for Phase 2.

Processes multiple agent decisions in batches to reduce API calls and costs.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time
from loguru import logger
from collections import defaultdict

from langchain_core.language_models import BaseChatModel


@dataclass
class LLMRequest:
    """Individual LLM request."""
    request_id: str
    agent_id: str
    prompt: str
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = 512
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LLMResponse:
    """LLM response."""
    request_id: str
    agent_id: str
    response: str
    success: bool = True
    error: Optional[str] = None
    tokens_used: int = 0
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class BatchLLMProcessor:
    """Process multiple LLM requests in batches.
    
    Features:
    - Automatic batching based on size and time
    - Parallel batch processing
    - Request queuing and routing
    - Error handling per request
    - Cost tracking
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        batch_size: int = 32,
        max_wait_time: float = 0.1,
        max_concurrent_batches: int = 4
    ):
        """Initialize batch processor.
        
        Args:
            llm: Language model instance
            batch_size: Maximum requests per batch
            max_wait_time: Maximum wait time before processing batch (seconds)
            max_concurrent_batches: Maximum batches to process concurrently
        """
        self.llm = llm
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_concurrent_batches = max_concurrent_batches
        
        # Request queue
        self.queue: asyncio.Queue[LLMRequest] = asyncio.Queue()
        self.pending_responses: Dict[str, asyncio.Future] = {}
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'total_batches': 0,
            'total_tokens': 0,
            'total_cost_usd': 0.0,
            'avg_batch_size': 0.0,
            'avg_latency_ms': 0.0,
            'error_count': 0
        }
        
        # Processing task
        self.processor_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        logger.info(
            f"BatchLLMProcessor initialized: "
            f"batch_size={batch_size}, max_wait={max_wait_time}s"
        )
    
    async def start(self):
        """Start the batch processor."""
        if self.is_running:
            return
        
        self.is_running = True
        self.processor_task = asyncio.create_task(self._process_loop())
        logger.info("BatchLLMProcessor started")
    
    async def stop(self):
        """Stop the batch processor."""
        self.is_running = False
        
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("BatchLLMProcessor stopped")
    
    async def process_request(
        self,
        agent_id: str,
        prompt: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: Optional[int] = 512,
        timeout: float = 30.0
    ) -> LLMResponse:
        """Process a single LLM request.
        
        Args:
            agent_id: Agent making the request
            prompt: Prompt text
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            timeout: Request timeout in seconds
            
        Returns:
            LLM response
        """
        request_id = f"{agent_id}_{int(time.time() * 1000)}"
        
        request = LLMRequest(
            request_id=request_id,
            agent_id=agent_id,
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Create future for response
        response_future = asyncio.Future()
        self.pending_responses[request_id] = response_future
        
        # Add to queue
        await self.queue.put(request)
        self.stats['total_requests'] += 1
        
        # Wait for response
        try:
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            logger.error(f"Request {request_id} timed out")
            return LLMResponse(
                request_id=request_id,
                agent_id=agent_id,
                response="",
                success=False,
                error="Request timed out"
            )
        finally:
            self.pending_responses.pop(request_id, None)
    
    async def process_batch_requests(
        self,
        requests: List[Tuple[str, str]]
    ) -> List[LLMResponse]:
        """Process multiple requests as a batch.
        
        Args:
            requests: List of (agent_id, prompt) tuples
            
        Returns:
            List of responses
        """
        tasks = [
            self.process_request(agent_id, prompt)
            for agent_id, prompt in requests
        ]
        
        return await asyncio.gather(*tasks)
    
    async def _process_loop(self):
        """Main processing loop."""
        while self.is_running:
            try:
                # Collect batch
                batch = await self._collect_batch()
                
                if not batch:
                    await asyncio.sleep(0.01)
                    continue
                
                # Process batch
                await self._process_batch(batch)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in process loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _collect_batch(self) -> List[LLMRequest]:
        """Collect a batch of requests.
        
        Returns:
            List of requests to process
        """
        batch = []
        start_time = time.time()
        
        while len(batch) < self.batch_size:
            # Check if we've waited too long
            if batch and (time.time() - start_time) > self.max_wait_time:
                break
            
            try:
                # Get request with timeout
                timeout = self.max_wait_time - (time.time() - start_time)
                if timeout <= 0:
                    break
                
                request = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=max(0.001, timeout)
                )
                batch.append(request)
                
            except asyncio.TimeoutError:
                break
        
        return batch
    
    async def _process_batch(self, batch: List[LLMRequest]):
        """Process a batch of requests.
        
        Args:
            batch: Requests to process
        """
        if not batch:
            return
        
        batch_start = time.time()
        self.stats['total_batches'] += 1
        
        logger.debug(f"Processing batch of {len(batch)} requests")
        
        # Group by model and temperature (can batch these together)
        groups = defaultdict(list)
        for req in batch:
            key = (req.model, req.temperature)
            groups[key].append(req)
        
        # Process each group
        responses = []
        for (model, temp), group_requests in groups.items():
            group_responses = await self._process_group(group_requests)
            responses.extend(group_responses)
        
        # Deliver responses
        for response in responses:
            if response.request_id in self.pending_responses:
                future = self.pending_responses[response.request_id]
                if not future.done():
                    future.set_result(response)
        
        # Update stats
        batch_duration = (time.time() - batch_start) * 1000  # ms
        self.stats['avg_batch_size'] = (
            (self.stats['avg_batch_size'] * (self.stats['total_batches'] - 1) + len(batch)) /
            self.stats['total_batches']
        )
        self.stats['avg_latency_ms'] = (
            (self.stats['avg_latency_ms'] * (self.stats['total_requests'] - len(batch)) + batch_duration) /
            self.stats['total_requests']
        )
        
        logger.debug(
            f"Batch processed: {len(batch)} requests in {batch_duration:.0f}ms "
            f"({len(batch)/(batch_duration/1000):.0f} req/sec)"
        )
    
    async def _process_group(
        self,
        requests: List[LLMRequest]
    ) -> List[LLMResponse]:
        """Process a group of similar requests.
        
        Args:
            requests: Requests with same model/temperature
            
        Returns:
            List of responses
        
        Note:
            Currently processes requests sequentially within the group.
            
            TODO Phase 3: Integrate true batch APIs when available:
            - OpenAI Batch API (when released)
            - Anthropic batch endpoints
            - Custom batch implementations
            
            For now, the main speedup comes from:
            1. Ray-based parallelization across agents
            2. Batch formation reducing overhead
            3. Async processing reducing blocking
        """
        responses = []
        
        # Process requests (sequential for safety and API compatibility)
        # Future: Replace with true batch API when available
        for request in requests:
            start_time = time.time()
            
            try:
                # Invoke LLM
                result = await asyncio.to_thread(
                    self.llm.invoke,
                    request.prompt
                )
                
                # Create response
                response = LLMResponse(
                    request_id=request.request_id,
                    agent_id=request.agent_id,
                    response=result.content if hasattr(result, 'content') else str(result),
                    success=True,
                    tokens_used=self._estimate_tokens(request.prompt, result),
                    latency_ms=(time.time() - start_time) * 1000
                )
                
                # Update stats
                self.stats['total_tokens'] += response.tokens_used
                self.stats['total_cost_usd'] += self._estimate_cost(
                    response.tokens_used,
                    request.model
                )
                
            except Exception as e:
                logger.error(f"Error processing request {request.request_id}: {e}")
                response = LLMResponse(
                    request_id=request.request_id,
                    agent_id=request.agent_id,
                    response="",
                    success=False,
                    error=str(e),
                    latency_ms=(time.time() - start_time) * 1000
                )
                self.stats['error_count'] += 1
            
            responses.append(response)
        
        return responses
    
    def _estimate_tokens(self, prompt: str, response: Any) -> int:
        """Estimate token count.
        
        Args:
            prompt: Input prompt
            response: LLM response
            
        Returns:
            Estimated token count
        """
        # Rough estimation: ~4 characters per token
        prompt_tokens = len(prompt) // 4
        response_text = response.content if hasattr(response, 'content') else str(response)
        response_tokens = len(response_text) // 4
        
        return prompt_tokens + response_tokens
    
    def _estimate_cost(self, tokens: int, model: str) -> float:
        """Estimate cost in USD.
        
        Args:
            tokens: Token count
            model: Model name
            
        Returns:
            Estimated cost in USD
        """
        # Rough pricing (per 1K tokens)
        pricing = {
            'gpt-4o-mini': 0.00015 + 0.0006,  # input + output
            'gpt-4': 0.03 + 0.06,
            'gpt-3.5-turbo': 0.0015 + 0.002,
            'claude-3-5-sonnet': 0.003 + 0.015
        }
        
        # Default pricing if model not found
        cost_per_1k = pricing.get(model, 0.001)
        
        return (tokens / 1000) * cost_per_1k
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            **self.stats,
            'queue_size': self.queue.qsize(),
            'pending_responses': len(self.pending_responses),
            'is_running': self.is_running,
            'success_rate': (
                (self.stats['total_requests'] - self.stats['error_count']) /
                max(1, self.stats['total_requests'])
            )
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'total_requests': 0,
            'total_batches': 0,
            'total_tokens': 0,
            'total_cost_usd': 0.0,
            'avg_batch_size': 0.0,
            'avg_latency_ms': 0.0,
            'error_count': 0
        }


class RequestQueue:
    """Priority queue for LLM requests."""
    
    def __init__(self, max_size: int = 10000):
        """Initialize request queue.
        
        Args:
            max_size: Maximum queue size
        """
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_size)
        self.stats = {
            'total_enqueued': 0,
            'total_dequeued': 0,
            'total_dropped': 0
        }
    
    async def enqueue(
        self,
        request: LLMRequest,
        priority: int = 5
    ) -> bool:
        """Add request to queue.
        
        Args:
            request: Request to add
            priority: Priority level (lower = higher priority)
            
        Returns:
            True if added, False if queue full
        """
        try:
            await self.queue.put((priority, request))
            self.stats['total_enqueued'] += 1
            return True
        except asyncio.QueueFull:
            self.stats['total_dropped'] += 1
            return False
    
    async def dequeue(self, timeout: Optional[float] = None) -> Optional[LLMRequest]:
        """Get request from queue.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Request or None if timeout
        """
        try:
            _, request = await asyncio.wait_for(
                self.queue.get(),
                timeout=timeout
            )
            self.stats['total_dequeued'] += 1
            return request
        except asyncio.TimeoutError:
            return None
    
    def size(self) -> int:
        """Get current queue size."""
        return self.queue.qsize()
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self.queue.empty()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            **self.stats,
            'current_size': self.size(),
            'drop_rate': self.stats['total_dropped'] / max(1, self.stats['total_enqueued'])
        }