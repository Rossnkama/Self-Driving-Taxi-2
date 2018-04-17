# Experience replay
import heapq
import operator


class DQNPrioritisedReplay(object):

    def __init__(self, capacity):
        """ Initialises the sliding window capacity and memory data structure

            :param capacity: The number of transitions our replay memory can hold
            :type capacity: int
        """
        # Max no of transition batches we want in our replay memory.
        self.capacity = capacity
        # The priority queue binary heap holding the last {capacity} no of events.
        self.memory = []
        # Determines how much prioritisation is used where 0 is a uniform distribution.
        self.alpha = 0.2
        # Small positive number to make sure transitions can still be visited after TD = 0
        self.epsilon = 0.2

    def push(self, transition):
        """ Pushes a new markov process transition + TD-Error to the replay memory window

            :param transition: A single markov process transition + TD-Error
            :type transition: list
        """
        # Pushing transition and said transition's temporal difference to memory as a tuple...
        # ([transition], TDE, priority)
        heapq.heappush(self.memory, (transition[:len(transition)-1], abs(transition[-1]), (abs(transition[-1]) + self.epsilon ** self.alpha)))
        # Keep no of transitions in sliding window constant
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        pass
