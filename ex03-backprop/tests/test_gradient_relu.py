import numpy as np
from typing import List

from lib.activations import ReLU
from lib.network_base import Module, Parameter


class Dummy(Module):
    """Dummy module.

    Note:
        This module is only used to check ReLU implementation in test_gradient_relu_bp
    """

    def __init__(self):
        super().__init__()
        self.relu1 = ReLU()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass calculation for the Dummy module.

        Args:
            x: Input data with shape (batch_size, in_features).

        Returns:
            Output data with shape (batch_size, out_features).

        """
        x = -x + 0.5 * np.ones_like(x)
        x = self.relu1.forward(x)
        x = -x + 0.5 * np.ones_like(x)
        self.input_cache = x
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Calculate the backward pass of the Dummy module.

        Args:
            grad: The gradient of the following layer with shape (batch_size, out_features).
                I.e. the given partial derivative of the loss w.r.t. this module's outputs.

        Returns:
            The gradient of this module with shape (batch_size, in_features).
                I.e. the partial derivative of the loss w.r.t. this module's inputs.

        """
        # grad will already be populated from check_gradients
        grad = grad * -1
        grad = self.relu1.backward(grad)
        grad = grad * -1
        return grad

    def parameters(self) -> List[Parameter]:
        """Return module parameters.

        Returns:
            An empty list, as the dummy module has no learnable parameters.
        """
        return []


def test_gradient_relu():
    """Test ReLU gradient."""
    input_vector = np.random.uniform(-1., 1., size=(2, 10))
    ReLU().check_gradients((input_vector,))

    input_vector = np.random.uniform(-1., 1., size=(4, 20))
    ReLU().check_gradients((input_vector,))

    input_vector = np.random.uniform(-1., 1., size=(6, 40))
    ReLU().check_gradients((input_vector,))


def test_gradient_relu_bp():
    """Test ReLU gradient. Test if the upstream gradient is backpropagated correctly."""
    input_vector = np.random.uniform(-1., 1., size=(2, 10))
    model = Dummy()
    model.check_gradients((input_vector,))

    input_vector = np.random.uniform(-1., 1., size=(4, 20))
    model = Dummy()
    model.check_gradients((input_vector,))

    input_vector = np.random.uniform(-1., 1., size=(6, 40))
    model = Dummy()
    model.check_gradients((input_vector,))


if __name__ == '__main__':
    test_gradient_relu()
    test_gradient_relu_bp()
    print("Test complete.")
