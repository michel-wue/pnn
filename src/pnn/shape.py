from dataclasses import dataclass
import numpy as np

@dataclass
class Shape():
    def __init__(self, shape: tuple):
        self.shape = shape
        #self.volume = tuple[0]