from .shape import Shape
import numpy as np

class Tensor():
    def __init__(
            self, 
            # shape: Shape,
            elements: np.array,
            deltas: np.array) -> None:
        self.elements = elements
        self.deltas = deltas
        self.shape = elements.shape