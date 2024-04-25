from dataclasses import dataclass
import numpy as np

@dataclass
class Shape():
    def __init__(self, int_array: np.array):
        self.int_array = int_array
        self.volume = int_array[0]