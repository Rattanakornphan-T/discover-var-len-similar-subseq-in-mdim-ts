from numba import float64, int32
from numba.experimental import jitclass
import numpy as np


spec = [
    ('data', float64[:]),   # the circular buffer
    ('capacity', int32),  # max capacity
    ('head', int32),      # index of first element
    ('tail', int32),      # index of one-past-last element
    ('size', int32),      # current number of elements
]

@jitclass(spec)
class Deque:
    def __init__(self, capacity):
        self.data = np.empty(capacity, dtype=np.float64)
        self.capacity = capacity
        self.head = 0
        self.tail = 0
        self.size = 0

    def append(self, x):
        if self.size == self.capacity:
            raise IndexError("Deque is full")
        self.data[self.tail] = x
        self.tail = (self.tail + 1) % self.capacity
        self.size += 1

    def appendleft(self, x):
        if self.size == self.capacity:
            raise IndexError("Deque is full")
        self.head = (self.head - 1 + self.capacity) % self.capacity
        self.data[self.head] = x
        self.size += 1

    def pop(self):
        if self.size == 0:
            raise IndexError("Deque is empty")
        self.tail = (self.tail - 1 + self.capacity) % self.capacity
        x = self.data[self.tail]
        self.size -= 1
        return x

    def popleft(self):
        if self.size == 0:
            raise IndexError("Deque is empty")
        x = self.data[self.head]
        self.head = (self.head + 1) % self.capacity
        self.size -= 1
        return x

    def __len__(self):
        return self.size
    
    def is_empty(self):
        return self.size == 0

    def left(self):
        if self.size == 0:
            raise IndexError("Deque is empty")
        return self.data[self.head]

    def right(self):
        if self.size == 0:
            raise IndexError("Deque is empty")
        return self.data[(self.tail - 1 + self.capacity) % self.capacity]