import random
from collections import deque
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Experience:
    """Class to define experience used for training RL Agent"""
    state: np.ndarray
    action: np.ndarray      # Can be int for Agent as DQN
    next_state: np.ndarray
    reward: float
    done: bool
    error: float # Only used for Double QN

@dataclass
class DataMemoryUpdate:
    """Class to define input to update memory"""
    indexes: list
    errors: List[float]

class Memory(ABC):
    """Abstract Class to define memory used for training RL Agent"""
    
    @abstractmethod
    def add(self, experience: Experience):
        '''Add new experience to memory'''

    @abstractmethod
    def sample(self, batch_size: int):
        '''Choose randomly experiences on memory'''

    @abstractmethod
    def update(self, data: DataMemoryUpdate):
        '''Update the preselection of memory for next sample'''

    @abstractmethod
    def __len__(self):
        '''Measure length of memory'''

class SimpleMemory(Memory):
    """Most simple memory without prioritized experiences depending on some criteria"""

    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return batch, None

    def update(self, data: DataMemoryUpdate):
        pass # No preselection for SimpleMemory

    def __len__(self):
        return len(self.buffer)

class PER:
    """Prioritized Experience Replay depending on estimation error"""

    e: float = 0.01
    a: float = 0.6
    size: int = 0

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.tree = SumTree(self.max_size)
        
    def _getPriority(self, error: float):
        return (error + self.e) ** self.a

    def add(self, experience: Experience):
        p = self._getPriority(experience.error)
        self.tree.add(p, experience)
        self.size=min(self.size+1, self.max_size)

    def sample(self, batch_size: int):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, _, data) = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
        return batch, idxs

    def update(self, data: DataMemoryUpdate):
        # update priority
        for idx, error in zip(data.indexes, data.errors):
            p = self._getPriority(error)
            self.tree.update(idx, p)        

    def __len__(self):
        return self.size

class SumTree(object):
    """Datastructure used for PER to quickely find best prioritized experience"""
    write: int = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])