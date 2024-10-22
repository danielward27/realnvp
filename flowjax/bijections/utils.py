"""Utility bijections (embedding network, permutations, inversion etc.)."""

from collections.abc import Callable
from math import prod
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int

from flowjax.bijections.bijection import AbstractBijection
from flowjax.utils import arraylike_to_array


class Invert(AbstractBijection):
    """Invert a bijection.

    This wraps a bijection, such that the transform methods become the inverse
    methods and vice versa. Note that in general, we define bijections such that
    the forward methods are preffered, i.e. faster/actually implemented. For
    training flows, we generally want the inverse method (used in density
    evaluation), to be faster. Hence it is often useful to use this class to
    achieve this aim.

    Args:
        bijection: Bijection to invert.
    """

    bijection: AbstractBijection

    def transform(self, x, condition=None):
        return self.bijection.inverse(x, condition)

    def transform_and_log_det(self, x, condition=None):
        return self.bijection.inverse_and_log_det(x, condition)

    def inverse(self, y, condition=None):
        return self.bijection.transform(y, condition)

    def inverse_and_log_det(self, y, condition=None):
        return self.bijection.transform_and_log_det(y, condition)

    @property
    def shape(self):
        return self.bijection.shape

    @property
    def cond_shape(self):
        return self.bijection.cond_shape


class Permute(AbstractBijection):
    """Permutation transformation.

    Args:
        permutation: An array with shape matching the array to transform, with elements
            0-(array.size-1) representing the new order based on the flattened array
            (uses, C-like ordering).
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    permutation: tuple[Array, ...]
    inverse_permutation: tuple[Array, ...]

    def __init__(self, permutation: Int[Array | np.ndarray, "..."]):
        permutation = arraylike_to_array(permutation, dtype=int)
        permutation = eqx.error_if(
            permutation,
            permutation.ravel().sort() != jnp.arange(permutation.size, dtype=int),
            "Invalid permutation array provided.",
        )
        self.shape = permutation.shape

        indices = jnp.unravel_index(permutation.ravel(), permutation.shape)
        self.permutation = tuple(jnp.reshape(i, permutation.shape) for i in indices)

        inv_indices = jnp.unravel_index(
            jnp.argsort(permutation.ravel()),
            permutation.shape,
        )
        self.inverse_permutation = tuple(
            jnp.reshape(i, permutation.shape) for i in inv_indices
        )

    def transform(self, x, condition=None):
        return x[self.permutation]

    def transform_and_log_det(self, x, condition=None):
        return x[self.permutation], jnp.array(0.0)

    def inverse(self, y, condition=None):
        return y[self.inverse_permutation]

    def inverse_and_log_det(self, y, condition=None):
        return y[self.inverse_permutation], jnp.array(0.0)

    def inverse_gradient_and_val(
        self,
        y: Array,
        y_grad: Array,
        y_logp: Array,
        condition: Array | None = None,
    ) -> tuple[Array, Array, Array]:
        x = y[self.inverse_permutation]
        x_grad = y_grad[self.inverse_permutation]
        return x, x_grad, y_logp


class Flip(AbstractBijection):
    """Flip the input array. Condition argument is ignored.

    Args:
        shape: The shape of the bijection. Defaults to None.
    """

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def transform(self, x: Array, condition: Array | None = None) -> Array:
        return jnp.flip(x)

    def transform_and_log_det(
        self, x: Array, condition: Array | None = None
    ) -> tuple[Array, Array]:
        return jnp.flip(x), jnp.zeros(())

    def inverse(self, y: Array, condition: Array | None = None) -> Array:
        return jnp.flip(y)

    def inverse_and_log_det(
        self, y: Array, condition: Array | None = None
    ) -> tuple[Array, Array]:
        return jnp.flip(y), jnp.zeros(())


class Partial(AbstractBijection):
    """Applies bijection to specific indices of an input.

    Args:
        bijection: Bijection that is compatible with the subset
            of x indexed by idxs. idxs: Indices (Integer, a slice, or an ndarray
            with integer/bool dtype) of the transformed portion.
        idxs: The indexes to transform.
        shape: Shape of the bijection. Defaults to None.
    """

    bijection: AbstractBijection
    idxs: int | slice | Array | tuple
    shape: tuple[int, ...]

    def __check_init__(self):
        expected_shape = jnp.zeros(self.shape)[self.idxs].shape
        if expected_shape != self.bijection.shape:
            raise ValueError(
                f"The bijection shape is incompatible with the subset of the input "
                f"indexed by 'idxs'. Bijection has shape {self.bijection.shape}, "
                f"while the subset has a shape of {expected_shape}.",
            )

    def transform(self, x, condition=None):
        y = self.bijection.transform(x[self.idxs], condition)
        return x.at[self.idxs].set(y)

    def transform_and_log_det(self, x, condition=None):
        y, log_det = self.bijection.transform_and_log_det(x[self.idxs], condition)
        return x.at[self.idxs].set(y), log_det

    def inverse(self, y, condition=None):
        x = self.bijection.inverse(y[self.idxs], condition)
        return y.at[self.idxs].set(x)

    def inverse_and_log_det(self, y, condition=None):
        x, log_det = self.bijection.inverse_and_log_det(y[self.idxs], condition)
        return y.at[self.idxs].set(x), log_det

    @property
    def cond_shape(self):
        return self.bijection.cond_shape


class Identity(AbstractBijection):
    """The identity bijection.

    Args:
       shape: The shape of the bijection. Defaults to ().
    """

    shape: tuple[int, ...] = ()
    cond_shape: ClassVar[None] = None

    def transform(self, x, condition=None):
        return x

    def transform_and_log_det(self, x, condition=None):
        return x, jnp.zeros(())

    def inverse(self, y, condition=None):
        return y

    def inverse_and_log_det(self, y, condition=None):
        return y, jnp.zeros(())


class EmbedCondition(AbstractBijection):
    """Wrap a bijection to include an embedding network.

    Generally this is used to reduce the dimensionality of the conditioning
    variable. The returned bijection has cond_dim equal to the raw condition size.

    Args:
        bijection: Bijection with ``bijection.cond_dim`` equal to the embedded size.
        embedding_net: A callable (e.g. equinox module) that embeds a conditioning
            variable to size ``bijection.cond_dim``.
        raw_cond_shape: The dimension of the raw conditioning variable.
    """

    bijection: AbstractBijection
    cond_shape: tuple[int, ...]
    embedding_net: Callable

    def __init__(
        self,
        bijection: AbstractBijection,
        embedding_net: Callable,
        raw_cond_shape: tuple[int, ...],
    ) -> None:
        self.bijection = bijection
        self.embedding_net = embedding_net
        self.cond_shape = raw_cond_shape

    def transform(self, x, condition=None):
        condition = self.embedding_net(condition)
        return self.bijection.transform(x, condition)

    def transform_and_log_det(self, x, condition=None):
        condition = self.embedding_net(condition)
        return self.bijection.transform_and_log_det(x, condition)

    def inverse(self, y, condition=None):
        condition = self.embedding_net(condition)
        return self.bijection.inverse(y, condition)

    def inverse_and_log_det(self, y, condition=None):
        condition = self.embedding_net(condition)
        return self.bijection.inverse_and_log_det(y, condition)

    @property
    def shape(self):
        return self.bijection.shape


class Reshape(AbstractBijection):
    """Wraps bijection methods with reshaping operations.

    One use case for this is for bijections that do not directly support a scalar
    shape, but this allows construction with shape (1, ) and reshaping to ().

    Args:
        bijection: The bijection to wrap.
        shape: The new input and output shape of the bijection. Defaults to
            unchanged.
        cond_shape: The new cond_shape of the bijection. Defaults to unchanged.

    Example:
        .. doctest::

            >>> import jax.numpy as jnp
            >>> from flowjax.bijections import Affine, Reshape
            >>> affine = Affine(loc=jnp.arange(4))
            >>> affine.shape
            (4,)
            >>> affine = Reshape(affine, (2,2))
            >>> affine.shape
            (2, 2)
            >>> affine.transform(jnp.zeros((2,2)))
            Array([[0., 1.],
                   [2., 3.]], dtype=float32)
    """

    bijection: AbstractBijection
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None = None

    def __init__(
        self,
        bijection: AbstractBijection,
        shape: tuple[int, ...] | None = None,
        cond_shape: tuple[int, ...] | None = None,
    ):
        self.bijection = bijection
        self.shape = shape if shape is not None else bijection.shape
        self.cond_shape = cond_shape if cond_shape is not None else bijection.cond_shape

    def __check_init__(self):
        if self.bijection.cond_shape is None and self.cond_shape is not None:
            raise ValueError(
                "Cannot reshape cond_shape for unconditional bijection.",
            )
        shapes = {
            "shape": (self.shape, self.bijection.shape),
            "cond_shape": (self.cond_shape, self.bijection.cond_shape),
        }

        for k, v in shapes.items():
            if v != (None, None) and prod(v[0]) != prod(v[1]):
                raise ValueError(
                    f"Cannot reshape to a different number of elements. Got {k} "
                    f"{v[0]}, but bijection has shape {v[1]}.",
                )

    def transform(self, x, condition=None):
        x = x.reshape(self.bijection.shape)
        if self.cond_shape is not None:
            condition = condition.reshape(self.bijection.cond_shape)
        return self.bijection.transform(x, condition).reshape(self.shape)

    def inverse(self, y, condition=None):
        y = y.reshape(self.bijection.shape)
        if self.cond_shape is not None:
            condition = condition.reshape(self.bijection.cond_shape)
        return self.bijection.inverse(y, condition).reshape(self.shape)

    def transform_and_log_det(self, x, condition=None):
        x = x.reshape(self.bijection.shape)
        if self.cond_shape is not None:
            condition = condition.reshape(self.bijection.cond_shape)
        y, log_det = self.bijection.transform_and_log_det(x, condition)
        return y.reshape(self.shape), log_det

    def inverse_and_log_det(self, y, condition=None):
        y = y.reshape(self.bijection.shape)
        if self.cond_shape is not None:
            condition = condition.reshape(self.bijection.cond_shape)
        x, log_det = self.bijection.inverse_and_log_det(y, condition)
        return x.reshape(self.shape), log_det
