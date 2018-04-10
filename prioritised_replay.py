# Experience replay


class PrioritisedExperienceReplay(object):

    def __init__(self, capacity):

        # Max no of transition batches we want in our replay memory
        self.capacity = capacity
        # The memory holding the max of the last {capacity} no of events
        self.memory = []