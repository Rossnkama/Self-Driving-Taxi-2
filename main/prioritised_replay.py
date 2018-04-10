# Experience replay


class PrioritisedExperienceReplay(object):

    def __init__(self, capacity):
        """ Initialises the sliding window capacity and memory data struct

            :param capacity: The number of transitions our replay memory can hold
            :type capacity: int
        """
        # Max no of transition batches we want in our replay memory
        self.capacity = capacity
        # The memory holding the max of the last {capacity} no of events
        self.memory = []

    def push(self, transition):
        """ Pushes a new markov process transition + TD-Error to the replay memory window

            :param transition: A single markov process transition + TD-Error
            :type transition: list
        """
        self.memory.append(transition)
        # Keep no of transitions in sliding window constant
        if len(self.memory) > self.capacity:
            del self.memory[0]
