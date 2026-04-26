import numpy as np

from core.loss_functions.loss_function import LossFunction
from core.models.neural_network import NeuralNetwork
from core.optimizers.optimizer import Optimizer


class Learner:
    def __init__(
            self,
            model: NeuralNetwork,
            loss_fn: LossFunction,
            optimizer: Optimizer,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.optimizer.initialize(model)

    def learn_step(self, input_batch: np.ndarray, output_batch: np.ndarray) -> float:
        predictions = self.model.predict(input_batch)
        loss = self.loss_fn.execute(predictions, output_batch)
        d_loss = self.loss_fn.derivative(predictions, output_batch)

        self.model.backward(d_loss)
        self.optimizer.step()

        return loss

    def learn(
            self,
            input_samples: np.ndarray,
            output_samples: np.ndarray,
            epochs: int = 100,
            batch_size: int | None = None,
            shuffle: bool = True,
            verbose: bool = True,
    ) -> dict[str, list[float]]:
        input_samples = np.asarray(input_samples, dtype=np.float32)
        output_samples = np.asarray(output_samples, dtype=np.float32)

        # Convert (samples,) to (samples, 1 feature)
        if input_samples.ndim == 1:
            input_samples = input_samples.reshape(-1, 1)
        if output_samples.ndim == 1:
            output_samples = output_samples.reshape(-1, 1)

        if input_samples.shape[0] != output_samples.shape[0]:
            raise ValueError(
                f"Input batch and output batch must have the same number of samples."
                f" Got {input_samples.shape[0]} and {output_samples.shape[0]}"
            )

        num_samples = input_samples.shape[0]
        batch_size = batch_size or num_samples

        history = {"loss": []}

        for epoch in range(epochs):
            if shuffle:
                indices = np.random.permutation(num_samples)
                input_samples_epoch = input_samples[indices]
                output_samples_epoch = output_samples[indices]
            else:
                input_samples_epoch = input_samples
                output_samples_epoch = output_samples

            epoch_loss = 0.0

            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                input_batch = input_samples_epoch[start:end]
                output_batch = output_samples_epoch[start:end]

                loss = self.learn_step(input_batch, output_batch)

                epoch_loss += loss * len(input_batch)

            epoch_loss /= num_samples
            history["loss"].append(epoch_loss)

            if verbose:
                if epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0 or epoch == epochs - 1:
                    print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.6f}")

        return history

    def evaluate(self, input_samples: np.ndarray, output_samples: np.ndarray) -> float:
        input_samples = np.asarray(input_samples, dtype=np.float32)
        output_samples = np.asarray(output_samples, dtype=np.float32)

        if input_samples.ndim == 1:
            input_samples = input_samples.reshape(-1, 1)
        if output_samples.ndim == 1:
            output_samples = output_samples.reshape(-1, 1)

        predictions = self.model.predict(input_samples)
        return self.loss_fn.execute(predictions, output_samples)

    def predict(self, input_samples: np.ndarray) -> np.ndarray:
        input_samples = np.asarray(input_samples, dtype=np.float32)
        if input_samples.ndim == 1:
            input_samples = input_samples.reshape(-1, 1)
        return self.model.predict(input_samples)

