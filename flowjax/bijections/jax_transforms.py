"""Bijections that wrap JAX function transforms (scan and vmap)."""

from collections.abc import Callable

import equinox as eqx
from jax import Array
import jax.numpy as jnp
from jax.lax import scan
from jax.tree_util import tree_leaves, tree_map
from jaxtyping import PyTree
from paramax import contains_unwrappables, unwrap

from flowjax.bijections.bijection import AbstractBijection


class Scan(AbstractBijection):
    """Repeatedly apply the same bijection with different parameter values.

    Internally, uses `jax.lax.scan` to reduce compilation time. Often it is convenient
    to construct these using ``equinox.filter_vmap``.

    Args:
        bijection: A bijection, in which the arrays leaves have an additional leading
            axis to scan over. It is often can convenient to create compatible
            bijections with ``equinox.filter_vmap``.

    Example:
        Below is equivilent to ``Chain([Affine(p) for p in params])``.

        .. doctest::

            >>> from flowjax.bijections import Scan, Affine
            >>> import jax.numpy as jnp
            >>> import equinox as eqx
            >>> params = jnp.ones((3, 2))
            >>> affine = eqx.filter_vmap(Affine)(params)
            >>> affine = Scan(affine)
    """

    bijection: AbstractBijection

    def transform_and_log_det(self, x, condition=None):
        def step(carry, bijection):
            x, log_det = carry
            y, log_det_i = bijection.transform_and_log_det(x, condition)
            return ((y, log_det + log_det_i.sum()), None)

        (y, log_det), _ = _filter_scan(step, (x, 0), self.bijection)
        return y, log_det

    def inverse_and_log_det(self, y, condition=None):
        def step(carry, bijection):
            y, log_det = carry
            x, log_det_i = bijection.inverse_and_log_det(y, condition)
            return ((x, log_det + log_det_i.sum()), None)

        (y, log_det), _ = _filter_scan(step, (y, 0), self.bijection, reverse=True)
        return y, log_det

    @property
    def shape(self):
        return self.bijection.shape

    @property
    def cond_shape(self):
        return self.bijection.cond_shape


def _filter_scan(f, init, xs, *, reverse=False):
    params, static = eqx.partition(xs, filter_spec=eqx.is_array)

    def _scan_fn(carry, x):
        module = eqx.combine(x, static)
        carry, y = f(carry, module)
        return carry, y

    return scan(_scan_fn, init, params, reverse=reverse)


def _check_no_unwrappables(pytree):
    if contains_unwrappables(pytree):
        raise ValueError(
            "In axes containing unwrappables is not supported. In axes must be "
            "specified to match the structure of the unwrapped pytree i.e after "
            "calling pararamax.unwrap."
        )


class Vmap(AbstractBijection):
    """Applies vmap to bijection methods to add a batch dimension to the bijection.

    Args:
        bijection: The bijection to vectorize.
        in_axes: Specify which axes of the bijection parameters to vectorise over. It
            should be a PyTree of ``None``, ``int`` with the tree structure being a
            prefix of the bijection, or a callable mapping ``Leaf -> Union[None, int]``.
            Note, if the bijection contains unwrappables, then in_axes should be
            specified for the unwrapped structure of the bijection. Defaults to None.
        axis_size: The size of the new axis. This should be left unspecified if in_axes
            is provided, as the size can be inferred from the bijection parameters.
            Defaults to None.
        in_axes_condition: Optionally define an axis of the conditioning variable to
            vectorize over. Defaults to None.

    Example:
        .. doctest::

            >>> # Add a bijection batch dimension, mapping over bijection parameters
            >>> import jax.numpy as jnp
            >>> import equinox as eqx
            >>> from flowjax.bijections import Vmap, RationalQuadraticSpline, Affine
            >>> bijection = eqx.filter_vmap(
            ...    lambda: RationalQuadraticSpline(knots=5, interval=2),
            ...    axis_size=10
            ... )()
            >>> bijection = Vmap(bijection, in_axes=eqx.if_array(0))
            >>> bijection.shape
            (10,)
            >>> # Add a bijection batch dimension, broadcasting bijection parameters:
            >>> bijection = RationalQuadraticSpline(knots=5, interval=2)
            >>> bijection = Vmap(bijection, axis_size=10)
            >>> bijection.shape
            (10,)

        A more advanced use case is to create bijections with more fine grained control
        over parameter broadcasting. For example, the ``Affine`` constructor broadcasts
        the location and scale parameters during initialization. What if we want an
        ``Affine`` bijection, with a global scale parameter, but an elementwise location
        parameter? We could achieve this as follows.

            >>> from jax.tree_util import tree_map
            >>> import paramax
            >>> bijection = Affine(jnp.zeros(()), jnp.ones(()))
            >>> bijection = eqx.tree_at(lambda bij: bij.loc, bijection, jnp.arange(3))
            >>> in_axes = tree_map(lambda _: None, paramax.unwrap(bijection))
            >>> in_axes = eqx.tree_at(
            ...     lambda bij: bij.loc, in_axes, 0, is_leaf=lambda x: x is None
            ...     )
            >>> bijection = Vmap(bijection, in_axes=in_axes)
            >>> bijection.shape
            (3,)
            >>> bijection.bijection.loc.shape
            (3,)
            >>> paramax.unwrap(bijection.bijection.scale).shape
            ()
            >>> x = jnp.ones(3)
            >>> bijection.transform(x)
            Array([1., 2., 3.], dtype=float32)

    """

    bijection: AbstractBijection
    in_axes: tuple
    axis_size: int
    cond_shape: tuple[int, ...] | None

    def __init__(
        self,
        bijection: AbstractBijection,
        *,
        in_axes: PyTree | None | int | Callable = None,
        axis_size: int | None = None,
        in_axes_condition: int | None = None,
    ):
        if in_axes is not None and axis_size is not None:
            raise ValueError("Cannot specify both in_axes and axis_size.")

        if axis_size is None:
            if in_axes is None:
                raise ValueError("Either axis_size or in_axes must be provided.")
            _check_no_unwrappables(in_axes)
            axis_size = _infer_axis_size_from_params(unwrap(bijection), in_axes)

        self.in_axes = (in_axes, 0, in_axes_condition)
        self.bijection = bijection
        self.axis_size = axis_size
        self.cond_shape = self.get_cond_shape(in_axes_condition)

    def vmap(self, f: Callable):
        return eqx.filter_vmap(f, in_axes=self.in_axes, axis_size=self.axis_size)

    def transform_and_log_det(self, x, condition=None):
        def _transform_and_log_det(bijection, x, condition):
            return bijection.transform_and_log_det(x, condition)

        y, log_det = self.vmap(_transform_and_log_det)(self.bijection, x, condition)
        return y, jnp.sum(log_det)

    def inverse_and_log_det(self, y, condition=None):
        def _inverse_and_log_det(bijection, x, condition):
            return bijection.inverse_and_log_det(x, condition)

        x, log_det = self.vmap(_inverse_and_log_det)(self.bijection, y, condition)
        return x, jnp.sum(log_det)

    @property
    def shape(self):
        return (self.axis_size, *self.bijection.shape)

    def get_cond_shape(self, cond_ax):
        if self.bijection.cond_shape is None or cond_ax is None:
            return self.bijection.cond_shape
        return (
            *self.bijection.cond_shape[:cond_ax],
            self.axis_size,
            *self.bijection.cond_shape[cond_ax:],
        )


def _infer_axis_size_from_params(tree: PyTree, in_axes) -> int:
    axes = _resolve_vmapped_axes(tree, in_axes)
    axis_sizes = tree_leaves(
        tree_map(
            lambda leaf, ax: leaf.shape[ax] if ax is not None else None,
            tree,
            axes,
        ),
    )
    if len(axis_sizes) == 0:
        raise ValueError("in_axes did not map to any leaves to vectorize.")
    return axis_sizes[0]


def _resolve_vmapped_axes(pytree, in_axes):
    """Returns pytree with ints denoting vmapped dimensions."""

    # Adapted from equinox filter_vmap
    def _resolve_axis(in_axes, elem):
        if in_axes is None or isinstance(in_axes, int):
            return tree_map(lambda _: in_axes, elem)
        if callable(in_axes):
            return tree_map(in_axes, elem)
        raise TypeError("`in_axes` must consist of None, ints, and callables.")

    return tree_map(_resolve_axis, in_axes, pytree, is_leaf=lambda x: x is None)
