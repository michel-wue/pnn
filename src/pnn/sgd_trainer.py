from .network import Network

class SGDTrainer:
    def __init__(
            self,
            learning_rate: float,
            amount_epochs: int,
            update_mechanism: SGDFlavor,
            batch_size: int = 1,
            shuffle: bool = True) -> None:
        self.learning_rate = learning_rate
        self.amount_epochs = amount_epochs
        self.update_mechanism = update_mechanism
        self.batch_size = batch_size
        self.shuffle = shuffle

    def optimize(network: Network):
        pass
