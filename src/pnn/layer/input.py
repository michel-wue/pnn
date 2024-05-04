import numpy as np
from collections.abc import Sequence
from ..tensor import Tensor

class InputLayer:
    def forward(self, rawData: list[any]) -> list[Tensor]:
        tensorlist = []
        for datapoint in rawData:
            flattened_input = np.ndarray.flatten(datapoint, order='C')
            # normalize values
            # flattened_input = np.divide(flattened_input, np.max(flattened_input))
            # flattened_input = flattened_input.reshape(-1, 1)
            tensorlist.append(Tensor(elements=flattened_input, deltas=None))
            # tensorlist.append(Tensor(shape=flattened_input, elements=flattened_input, deltas=None))
        return tensorlist