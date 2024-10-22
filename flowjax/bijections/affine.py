"""Affine bijections."""

from collections.abc import Callable
from functools import partial
from typing import ClassVar

import jax.numpy as jnp
from jax.nn import softplus
from jax.scipy.linalg import solve_triangular
from jaxtyping import Array, ArrayLike, Shaped
from paramax import AbstractUnwrappable, Parameterize, unwrap
from paramax.utils import inv_softplus

from flowjax.bijections.bijection import AbstractBijection
from flowjax.utils import arraylike_to_array


class Affine(AbstractBijection):
    r"""Elementwise affine transformation :math:`y = a \cdot x + b`.

    ``loc`` and ``scale`` should broadcast to the desired shape of the bijection.
    By default, we constrain the scale parameter to be postive using ``softplus``, but
    other parameterizations can be achieved by replacing the scale parameter after
    construction e.g. using ``eqx.tree_at``.

    Args:
        loc: Location parameter. Defaults to 0.
        scale: Scale parameter. Defaults to 1.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    loc: Array
    scale: Array | AbstractUnwrappable[Array]

    def __init__(
        self,
        loc: ArrayLike = 0,
        scale: ArrayLike = 1,
    ):
        self.loc, scale = jnp.broadcast_arrays(
            *(arraylike_to_array(a, dtype=float) for a in (loc, scale)),
        )
        self.shape = scale.shape
        self.scale = Parameterize(softplus, inv_softplus(scale))

    def transform(self, x, condition=None):
        return x * self.scale + self.loc

    def transform_and_log_det(self, x, condition=None):
        return x * self.scale + self.loc, jnp.log(jnp.abs(self.scale)).sum()

    def inverse(self, y, condition=None):
        return (y - self.loc) / self.scale

    def inverse_and_log_det(self, y, condition=None):
        return (y - self.loc) / self.scale, -jnp.log(jnp.abs(self.scale)).sum()

    def inverse_gradient_and_val(
        self,
        y: Array,
        y_grad: Array,
        y_logp: Array,
        condition: ArrayLike | None = None,
    ) -> tuple[Array, Array, Array]:
        x, logdet = self.inverse_and_log_det(y)
        x_grad = y_grad * self.scale
        return (x, x_grad, y_logp - logdet)


class Loc(AbstractBijection):
    r"""Location transformation :math:`y = a \cdot x + b`.

    Args:
        loc: Scale parameter. Defaults to 1.
    """

    loc: Array
    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None

    def __init__(self, loc: ArrayLike):
        self.loc = arraylike_to_array(loc, dtype=float)
        self.shape = self.loc.shape

    def transform(self, x, condition=None):
        return x + self.loc

    def transform_and_log_det(self, x, condition=None):
        return x + self.loc, jnp.zeros(())

    def inverse(self, y, condition=None):
        return y - self.loc

    def inverse_and_log_det(self, y, condition=None):
        return y - self.loc, jnp.zeros(())


class Scale(AbstractBijection):
    r"""Scale transformation :math:`y = a \cdot x`.

    Args:
        scale: Scale parameter. Defaults to 1.
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    scale: Array | AbstractUnwrappable[Array]

    def __init__(
        self,
        scale: ArrayLike,
    ):
        scale = arraylike_to_array(scale, "scale", dtype=float)
        self.scale = Parameterize(softplus, inv_softplus(scale))
        self.shape = jnp.shape(unwrap(scale))

    def transform(self, x, condition=None):
        return x * self.scale

    def transform_and_log_det(self, x, condition=None):
        return x * self.scale, jnp.log(jnp.abs(self.scale)).sum()

    def inverse(self, y, condition=None):
        return y / self.scale

    def inverse_and_log_det(self, y, condition=None):
        return y / self.scale, -jnp.log(jnp.abs(self.scale)).sum()


class TriangularAffine(AbstractBijection):
    r"""A triangular affine transformation.

    Transformation has the form :math:`Ax + b`, where :math:`A` is a lower or upper
    triangular matrix, and :math:`b` is the bias vector. We assume the diagonal
    entries are positive, and constrain the values using softplus. Other
    parameterizations can be achieved by e.g. replacing ``self.triangular``
    after construction.

    Args:
        loc: Location parameter. If this is scalar, it is broadcast to the dimension
            inferred from arr.
        arr: Triangular matrix.
        lower: Whether the mask should select the lower or upper
            triangular matrix (other elements ignored). Defaults to True (lower).
    """

    shape: tuple[int, ...]
    cond_shape: ClassVar[None] = None
    loc: Array
    triangular: Array | AbstractUnwrappable[Array]
    lower: bool

    def __init__(
        self,
        loc: Shaped[ArrayLike, " #dim"],
        arr: Shaped[Array, "dim dim"],
        *,
        lower: bool = True,
    ):
        loc, arr = (arraylike_to_array(a, dtype=float) for a in (loc, arr))
        if (arr.ndim != 2) or (arr.shape[0] != arr.shape[1]):
            raise ValueError("arr must be a square, 2-dimensional matrix.")
        dim = arr.shape[0]
        arr = jnp.fill_diagonal(arr, inv_softplus(jnp.diag(arr)), inplace=False)

        @partial(jnp.vectorize, signature="(d,d)->(d,d)")
        def _to_triangular(arr):
            tri = jnp.tril(arr) if lower else jnp.triu(arr)
            return jnp.fill_diagonal(tri, softplus(jnp.diag(tri)), inplace=False)

        self.triangular = Parameterize(_to_triangular, arr)
        self.lower = lower
        self.shape = (dim,)
        self.loc = jnp.broadcast_to(loc, (dim,))

    def transform(self, x, condition=None):
        return self.triangular @ x + self.loc

    def transform_and_log_det(self, x, condition=None):
        y = self.triangular @ x + self.loc
        return y, jnp.log(jnp.abs(jnp.diag(self.triangular))).sum()

    def inverse(self, y, condition=None):
        return solve_triangular(self.triangular, y - self.loc, lower=self.lower)

    def inverse_and_log_det(self, y, condition=None):
        x = solve_triangular(self.triangular, y - self.loc, lower=self.lower)
        return x, -jnp.log(jnp.abs(jnp.diag(self.triangular))).sum()


class AdditiveCondition(AbstractBijection):
    """Given a callable ``f``, carries out the transformation ``y = x + f(condition)``.

    If used to transform a distribution, this allows the "location" to be changed as a
    function of the conditioning variables. Note that the callable can be a callable
    module with trainable parameters.

    Args:
        module: A callable (e.g. a function or callable module) that maps array with
            shape cond_shape, to a shape that is broadcastable with the shape of the
            bijection.
        shape: The shape of the bijection.
        cond_shape: The condition shape of the bijection.

    Example:
        Conditioning using a linear transformation

        .. doctest::

            >>> from flowjax.bijections import AdditiveCondition
            >>> from equinox.nn import Linear
            >>> import jax.numpy as jnp
            >>> import jax.random as jr
            >>> bijection = AdditiveCondition(
            ...     Linear(2, 3, key=jr.key(0)), shape=(3,), cond_shape=(2,)
            ...     )
            >>> bijection.transform(jnp.ones(3), condition=jnp.ones(2))
            Array([1.9670618, 0.8156546, 1.7763454], dtype=float32)

    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...]
    module: Callable[[ArrayLike], ArrayLike]

    def __init__(
        self,
        module: Callable[[ArrayLike], ArrayLike],
        shape: tuple[int, ...],
        cond_shape: tuple[int, ...],
    ):
        self.module = module
        self.shape = shape
        self.cond_shape = cond_shape

    def transform(self, x, condition=None):
        return x + self.module(condition)

    def transform_and_log_det(self, x, condition=None):
        return self.transform(x, condition), jnp.zeros(())

    def inverse(self, y, condition=None):
        return y - self.module(condition)

    def inverse_and_log_det(self, y, condition=None):
        return self.inverse(y, condition), jnp.zeros(())
