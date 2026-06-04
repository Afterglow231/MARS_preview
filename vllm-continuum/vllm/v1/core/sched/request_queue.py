# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum
from typing import Tuple

from vllm.v1.request import Request
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
import time

class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""
    FCFS = "fcfs"
    PRIORITY = "priority"
    CONTINUUM = "continuum"
    CONTINUUM_DY = "continuum_dy"
    AUTELLIX = "autellix"
    INFERCEPT = "infercept"
    MARS = "mars"

class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass

    @abstractmethod
    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        self.extendleft(reversed(requests))

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [
            req for req in self if req not in requests_to_remove
        ]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
        return super().__reversed__()


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    Requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[int, float, Request]] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap,
                       (request.priority, request.arrival_time, request))

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        _, _, request = heapq.heappop(self._heap)
        return request

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        _, _, request = self._heap[0]
        return request

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.
        
        Note: In a priority queue, there is no concept of prepending to the 
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.
        
        Note: In a priority queue, there is no concept of prepending to the 
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap = [(p, t, r) for p, t, r in self._heap if r != request]
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        self._heap = [(p, t, r) for p, t, r in self._heap
                      if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            _, _, request = heapq.heappop(heap_copy)
            yield request

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse priority order."""
        return reversed(list(self))

# TODO (Hanchen) need to implement ContinuumRequestQueue that schedules requests based on the last func call, it can call another predictor class if needed
class ContinuumRequestQueue(deque[Request], RequestQueue):
    
    def __init__(self) -> None:
        super().__init__()
        # Track the first entry time for each job_id
        self.job_id_first_entry_time: dict[str, float] = {}
   
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        # Record the first entry time for this job_id if not already recorded
        if request.job_id not in self.job_id_first_entry_time:
            self.job_id_first_entry_time[request.job_id] = request.arrival_time
        self.append(request)

    def pop_request(self, pinned_requests: list[Tuple[Request, float]], kv_cache_manager: KVCacheManager, connector: KVConnectorBase_V1) -> Request:
        """Pop a request from the queue according to continuum policy."""
        request = self.peek_request(pinned_requests, kv_cache_manager, connector)
        self.remove_request(request)
        return request

    # NOTE (Hanchen): priority is pinned request -> job_id level FCFS
    def peek_request(self, pinned_requests: list[Tuple[Request, float]], kv_cache_manager: KVCacheManager, connector: KVConnectorBase_V1) -> Request:
        if not self:
            raise IndexError("peek from an empty queue")
        # Extract just the requests from pinned_requests tuples
        pinned_request_job_id_set = {req.job_id for req, _ in pinned_requests}

        # First, use the pinned request
        earliest_request = None
        earliest_entry_time = float('inf')
        for request in self:
            if request.job_id in pinned_request_job_id_set:
                job_entry_time = self.job_id_first_entry_time.get(request.job_id, request.arrival_time)
                if job_entry_time < earliest_entry_time:
                    earliest_entry_time = job_entry_time
                    earliest_request = request
        
        if earliest_request is not None:
            return earliest_request
        
        # Otherwise, use job_id level FCFS: find the request whose job_id has the earliest first entry time
        if self:
            earliest_request = None
            earliest_entry_time = float('inf')
            
            for request in self:
                job_entry_time = self.job_id_first_entry_time.get(request.job_id, request.arrival_time)
                if job_entry_time < earliest_entry_time:
                    earliest_entry_time = job_entry_time
                    earliest_request = request
            
            return earliest_request
        else:
            raise IndexError("peek from an empty queue")

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        # Record the first entry time for this job_id if not already recorded
        if request.job_id not in self.job_id_first_entry_time:
            self.job_id_first_entry_time[request.job_id] = request.arrival_time
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        # Record first entry times for new job_ids
        for request in requests:
            if request.job_id not in self.job_id_first_entry_time:
                self.job_id_first_entry_time[request.job_id] = request.arrival_time
        self.extendleft(reversed(requests))

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [
            req for req in self if req not in requests_to_remove
        ]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
        return super().__reversed__()


class AutellixRequestQueue(RequestQueue):
    def __init__(self, *, num_queues: int = 4) -> None:
        if num_queues <= 0:
            raise ValueError("num_queues must be positive")
        self._num_queues = num_queues
        self._queues: list[deque[Request]] = [
            deque() for _ in range(self._num_queues)
        ]
        self._req_to_q_idx: dict[str, int] = {}

    def _clamp_q_idx(self, q_idx: int) -> int:
        return max(0, min(int(q_idx), self._num_queues - 1))

    def _get_q_idx(self, request: Request) -> int:
        q_idx = getattr(request, "_autellix_q_idx", 0)
        return self._clamp_q_idx(int(q_idx))

    def add_request(self, request: Request) -> None:
        q_idx = self._get_q_idx(request)
        self._queues[q_idx].append(request)
        self._req_to_q_idx[request.request_id] = q_idx

    def pop_request(self) -> Request:
        for q_idx in range(self._num_queues):
            if self._queues[q_idx]:
                request = self._queues[q_idx].popleft()
                self._req_to_q_idx.pop(request.request_id, None)
                return request
        raise IndexError("pop from an empty queue")

    def peek_request(self) -> Request:
        for q_idx in range(self._num_queues):
            if self._queues[q_idx]:
                return self._queues[q_idx][0]
        raise IndexError("peek from an empty queue")

    def prepend_request(self, request: Request) -> None:
        q_idx = self._get_q_idx(request)
        self._queues[q_idx].appendleft(request)
        self._req_to_q_idx[request.request_id] = q_idx

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in reversed(list(requests)):
            self.prepend_request(request)

    def remove_request(self, request: Request) -> None:
        req_id = request.request_id
        q_idx = self._req_to_q_idx.get(req_id)
        if q_idx is not None:
            try:
                self._queues[q_idx].remove(request)
                self._req_to_q_idx.pop(req_id, None)
                return
            except ValueError:
                pass
        for q in self._queues:
            try:
                q.remove(request)
                self._req_to_q_idx.pop(req_id, None)
                return
            except ValueError:
                continue
        raise ValueError("request not found in queue")

    def remove_requests(self, requests: Iterable[Request]) -> None:
        for request in requests:
            try:
                self.remove_request(request)
            except ValueError:
                continue

    def __bool__(self) -> bool:
        return any(self._queues)

    def __len__(self) -> int:
        return sum(len(q) for q in self._queues)

    def __iter__(self) -> Iterator[Request]:
        for q in self._queues:
            yield from q

    def __reversed__(self) -> Iterator[Request]:
        for q in reversed(self._queues):
            yield from reversed(q)

def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy in (SchedulingPolicy.CONTINUUM, SchedulingPolicy.CONTINUUM_DY):
        return ContinuumRequestQueue()
    elif policy == SchedulingPolicy.AUTELLIX:
        return AutellixRequestQueue()
    elif policy == SchedulingPolicy.INFERCEPT:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.MARS:
        return MarsRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")


class MarsRequestQueue(RequestQueue):
    """A heap-backed tiered priority queue for WAITING requests.

    This queue is used only for the waiting set. The vllm V1 scheduler keeps RUNNING
    requests in a separate structure and may apply additional policy-specific
    ordering on top of it.

    Level 0 has the highest priority. By default, new requests are assigned an
    initial level based on prompt length buckets. `prepend_request` boosts a
    request to level 0 to prioritize resumption after preemption.
    """

    def __init__(
        self,
        *,
        prompt_len_thresholds: tuple[int, ...] = (2048, 8192, 32768),
    ) -> None:
        self._prompt_len_thresholds = prompt_len_thresholds
        # heap entry: (level, order, arrival_time, request_id, version)
        self._heap: list[tuple[int, int, float, str, int]] = []
        self._requests: dict[str, Request] = {}
        self._version: dict[str, int] = {}
        self._key_by_id: dict[str, tuple[int, int, float]] = {}
        self._seq: int = 0
        self._prepend_seq: int = 0

    def _get_level(self, request: Request) -> int:
        prompt_len = request.num_prompt_tokens
        for level, threshold in enumerate(self._prompt_len_thresholds):
            if prompt_len <= threshold:
                return level
        return len(self._prompt_len_thresholds)

    def _push(self, request: Request, *, level: int, order: int) -> None:
        req_id = request.request_id
        version = self._version.get(req_id, 0) + 1
        self._version[req_id] = version
        self._requests[req_id] = request
        key = (level, order, request.arrival_time)
        self._key_by_id[req_id] = key
        heapq.heappush(self._heap, (level, order, request.arrival_time, req_id,
                                    version))

    def _purge(self) -> None:
        while self._heap:
            level, order, arrival_time, req_id, version = self._heap[0]
            current = self._requests.get(req_id)
            if current is None:
                heapq.heappop(self._heap)
                continue
            if self._version.get(req_id) != version:
                heapq.heappop(self._heap)
                continue
            # Keep mypy/hygiene: assert key matches.
            _ = (level, order, arrival_time)
            return

    def add_request(self, request: Request) -> None:
        """Add a request with an initial priority level."""
        self._seq += 1
        self._push(request, level=self._get_level(request), order=self._seq)

    def pop_request(self) -> Request:
        """Pop the next request according to MARS priority."""
        self._purge()
        if not self._heap:
            raise IndexError("pop from empty heap")
        _, _, _, req_id, _ = heapq.heappop(self._heap)
        request = self._requests.pop(req_id)
        self._version.pop(req_id, None)
        self._key_by_id.pop(req_id, None)
        return request

    def peek_request(self) -> Request:
        """Peek at the next request without removing it."""
        self._purge()
        if not self._heap:
            raise IndexError("peek from empty heap")
        _, _, _, req_id, _ = self._heap[0]
        return self._requests[req_id]

    def prepend_request(self, request: Request) -> None:
        """Boost a request to the highest level and enqueue it first."""
        self._prepend_seq -= 1
        self._push(request, level=0, order=self._prepend_seq)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Boost and enqueue all requests first."""
        for request in requests:
            self.prepend_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        req_id = request.request_id
        self._requests.pop(req_id, None)
        self._version.pop(req_id, None)
        self._key_by_id.pop(req_id, None)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        for request in requests:
            self.remove_request(request)

    def __bool__(self) -> bool:
        return bool(self._requests)

    def __len__(self) -> int:
        return len(self._requests)

    def __iter__(self) -> Iterator[Request]:
        # Yield active requests in the same ordering used by peek/pop.
        for req_id, _ in sorted(self._key_by_id.items(),
                                key=lambda kv: kv[1]):
            yield self._requests[req_id]

    def __reversed__(self) -> Iterator[Request]:
        return reversed(list(self))
