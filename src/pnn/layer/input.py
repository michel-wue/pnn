import numpy as np
from ..tensor import Tensor

class InputLayer:
    def __init__(self, type) -> None:
        self.type = type
    def forward(self, rawData: list[any]) -> list[Tensor]:
        tensorlist = []
        if self.type == 'fully_connected':
            for datapoint in rawData:
                flattened_input = np.ndarray.flatten(datapoint, order='C')
                tensorlist.append(Tensor(elements=flattened_input, deltas=None))
            return tensorlist
        else:
            for datapoint in rawData:
                if len(datapoint.shape) < 3:
                    datapoint = datapoint.reshape(datapoint.shape[0], datapoint.shape[0], 1)
                tensorlist.append(Tensor(elements=datapoint, deltas=None))
            return tensorlist