import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Any, Callable, Dict, Optional, Tuple, Union
from mlx.utils import tree_flatten

class Muon(optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-Schulz

    This optimizer is an MLX implementation of the Muon optimizer from:
    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which
    can be computed efficiently on the GPU.

    Some usage notes:
    - This optimizer works best for most matrix parameters, especially weight matrices in MLPs 
      and attention blocks.
    - It may not be suitable for the embedding layer, the final fully connected layer,
      or any {0,1}-D parameters; those should be optimized by a standard method (e.g., AdamW).
    - For 4D convolutional filters, it's best to reshape them to 2D before applying Muon.

    Args:
        learning_rate: The learning rate used by the internal SGD.
        momentum: The momentum coefficient used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum. Default is True.
        ns_steps: The number of Newton-Schulz iteration steps to use. Default is 5.
        alternate_optimizer: An optimizer to use for non-matrix parameters. Default is None.
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable] = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        alternate_optimizer: Optional[optim.Optimizer] = None,
        betas: Tuple[float, float] = None,
        eps: float = None,
        weight_decay: float = None,
    ):
        super().__init__()
        self._learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.alternate_optimizer = alternate_optimizer
        self.state = {}
        self.count = 0

    def zeropower_via_newtonschulz5(self, G, steps):
        """
        Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

        Args:
            G: Matrix to orthogonalize
            steps: Number of Newton-Schulz iterations to perform

        Returns:
            Approximately orthogonalized matrix
        """
        a, b, c = (3.4445, -4.7750, 2.0315)

        is_transposed = False
        if G.shape[-2] > G.shape[-1]:
            G = mx.transpose(G, axes=(-1, -2))
            is_transposed = True

        norm = mx.norm(G, axis=(-2, -1), keepdims=True)
        X = G / (norm + 1e-7)

        for _ in range(steps):
            A = X @ mx.transpose(X, axes=(-1, -2))
            B = b * A + c * A @ A
            X = a * X + B @ X

        if is_transposed:
            X = mx.transpose(X, axes=(-1, -2))

        return X

    def update(self, model: nn.Module, gradients: Dict[str, mx.array]) -> Dict[str, mx.array]:
        # Initialize full update dictionary
        updates = {}
        non_matrix_params = {}
        non_matrix_grads = {}

        for name, param in tree_flatten(model.parameters()):
            grad = gradients.get(name)
            if grad is None:
                continue

            if len(param.shape) == 2:
                if name not in self.state:
                    self.state[name] = {"momentum_buffer": mx.zeros_like(grad)}

                momentum_buffer = self.state[name]["momentum_buffer"]
                momentum_buffer = (1 - self.momentum) * grad + self.momentum * momentum_buffer
                self.state[name]["momentum_buffer"] = momentum_buffer

                if self.nesterov:
                    final_grad = grad + self.momentum * momentum_buffer
                else:
                    final_grad = momentum_buffer

                orthogonalized_grad = self.zeropower_via_newtonschulz5(final_grad, self.ns_steps)

                scaling = max(1, param.shape[0] / param.shape[1]) ** 0.5
                lr = self._learning_rate(self.count) if callable(self._learning_rate) else self._learning_rate

                updates[name] = -lr * scaling * orthogonalized_grad
            else:
                non_matrix_params[name] = param
                non_matrix_grads[name] = grad

        if self.alternate_optimizer is not None and non_matrix_params:
            alt_updates = self.alternate_optimizer.update(nn.Module(non_matrix_params), non_matrix_grads)
            updates.update(alt_updates)
        else:
            for name, param in non_matrix_params.items():
                grad = non_matrix_grads[name]
                if name not in self.state:
                    self.state[name] = {"momentum_buffer": mx.zeros_like(grad)}

                momentum_buffer = self.state[name]["momentum_buffer"]
                momentum_buffer = (1 - self.momentum) * grad + self.momentum * momentum_buffer
                self.state[name]["momentum_buffer"] = momentum_buffer

                if self.nesterov:
                    final_grad = grad + self.momentum * momentum_buffer
                else:
                    final_grad = momentum_buffer

                lr = self._learning_rate(self.count) if callable(self._learning_rate) else self._learning_rate
                updates[name] = -lr * final_grad

        self.count += 1
        return updates