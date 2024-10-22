"""Chain bijection which allows sequential application of arbitrary bijections."""

from collections.abc import Sequence

from paramax import AbstractUnwrappable, unwrap
from jax import Array
import jax.numpy as jnp

from flowjax.bijections.bijection import AbstractBijection
from flowjax.utils import check_shapes_match, merge_cond_shapes


class Chain(AbstractBijection):
    """Compose arbitrary bijections to form another bijection.

    Args:
        bijections: Sequence of bijections. The bijection shapes must match, and any
            none None condition shapes must match.
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    bijections: tuple[AbstractBijection | AbstractUnwrappable[AbstractBijection], ...]

    def __init__(
        self,
        bijections: Sequence[
            AbstractBijection | AbstractUnwrappable[AbstractBijection]
        ],
    ):
        unwrapped = unwrap(bijections)
        check_shapes_match([b.shape for b in unwrapped])
        self.shape = unwrapped[0].shape
        self.cond_shape = merge_cond_shapes([unwrap(b).cond_shape for b in unwrapped])
        self.bijections = tuple(bijections)

    def transform(self, x: Array, condition: Array | None = None) -> Array:
        for bijection in self.bijections:
            x = bijection.transform(x, condition)
        return x

    def transform_and_log_det(
        self, x: Array, condition: Array | None = None
    ) -> tuple[Array, Array]:
        log_abs_det_jac = jnp.zeros(())
        for bijection in self.bijections:
            x, log_abs_det_jac_i = bijection.transform_and_log_det(x, condition)
            log_abs_det_jac += log_abs_det_jac_i.sum()
        return x, log_abs_det_jac

    def inverse(self, y, condition=None):
        for bijection in reversed(self.bijections):
            y = bijection.inverse(y, condition)
        return y

    def inverse_and_log_det(
        self, y: Array, condition: Array | None = None
    ) -> tuple[Array, Array]:
        log_abs_det_jac = jnp.zeros(())
        for bijection in reversed(self.bijections):
            y, log_abs_det_jac_i = bijection.inverse_and_log_det(y, condition)
            log_abs_det_jac += log_abs_det_jac_i.sum()
        return y, log_abs_det_jac

    def inverse_gradient_and_val(
        self,
        y: Array,
        y_grad: Array,
        y_logp: Array,
        condition: Array | None = None,
    ) -> tuple[Array, Array, Array]:
        for bijection in reversed(self.bijections):
            y, y_grad, y_logp = bijection.inverse_gradient_and_val(
                y, y_grad, y_logp, condition
            )
        return y, y_grad, y_logp

    def __getitem__(self, i: int | slice) -> AbstractBijection:
        if isinstance(i, int):
            return self.bijections[i]
        if isinstance(i, slice):
            return Chain(self.bijections[i])
        raise TypeError(f"Indexing with type {type(i)} is not supported.")

    def __iter__(self):
        yield from self.bijections

    def __len__(self):
        return len(self.bijections)

    def merge_chains(self):
        """Returns an equivilent Chain object, in which nested chains are flattened."""
        bijections = self.bijections
        while any(isinstance(b, Chain) for b in bijections):
            bij = []
            for b in bijections:
                if isinstance(b, Chain):
                    bij.extend(b.bijections)
                else:
                    bij.append(b)
            bijections = bij
        return Chain(bijections)
