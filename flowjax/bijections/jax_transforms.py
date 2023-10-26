"""Bijections that wrap jax function transforms (scan and vmap)."""

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jax.lax import scan
from jax.tree_util import tree_leaves, tree_map

from flowjax.bijections.bijection import AbstractBijection


class Scan(AbstractBijection):
    """Repeatedly apply the same bijection with different parameter values.

    Internally, uses `jax.lax.scan` to reduce compilation time.
    """

    bijection: AbstractBijection

    def __init__(self, bijection: AbstractBijection):
        """Initialize the scan bijection.

        The array leaves in `bijection` should have an additional leading axis to scan
        over. Often it is convenient to construct these using ``equinox.filter_vmap``.

        Args:
            bijection (AbstractBijection): A bijection, in which the arrays leaves have
                an additional leading axis to scan over. For complex bijections, it can
                be convenient to create compatible bijections with
                ``equinox.filter_vmap``.

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
        self.bijection = bijection

    def transform(self, x, condition=None):
        def step(x, bijection):
            return (bijection.transform(x, condition), None)

        y, _ = _filter_scan(step, x, self.bijection)
        return y

    def transform_and_log_det(self, x, condition=None):
        def step(carry, bijection):
            x, log_det = carry
            y, log_det_i = bijection.transform_and_log_det(x, condition)
            return ((y, log_det + log_det_i.sum()), None)

        (y, log_det), _ = _filter_scan(step, (x, 0), self.bijection)
        return y, log_det

    def inverse(self, y, condition=None):
        def step(y, bijection):
            return bijection.inverse(y, condition), None

        x, _ = _filter_scan(step, y, self.bijection, reverse=True)
        return x

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


def _filter_scan(f, init, xs, reverse=False):
    params, static = eqx.partition(xs, filter_spec=eqx.is_array)

    def _scan_fn(carry, x):
        module = eqx.combine(x, static)
        carry, y = f(carry, module)
        return carry, y

    return scan(_scan_fn, init, params, reverse=reverse)


class Vmap(AbstractBijection):
    """Applies vmap to bijection methods to add a batch dimension to the bijection.

    Example:
        The two most common use cases, are shown below:

        .. doctest::

            Add a batch dimension to a bijection, mapping over bijection parameters:

            >>> import jax.numpy as jnp
            >>> import equinox as eqx
            >>> from flowjax.bijections import Vmap, RationalQuadraticSpline
            >>> bijection = eqx.filter_vmap(RationalQuadraticSpline, axis_size=10)(5, 2)
            >>> bijection = Vmap(bijection, eqx.if_array(0))
            >>> bijection.shape
            (10,)

            Add a batch dimension to a bijection, broadcasting bijection parameters:
            >>> bijection = RationalQuadraticSpline(5, 2)
            >>> bijection = Vmap(bijection, axis_size=10)
            >>> bijection.shape
            (10,)

        A more advanced use case is to create bijections with more fine grained control
        over parameter broadcasting. For example, the ``Affine`` constructor broadcasts
        the location and scale parameters during initialization. What if we want an
        ``Affine`` bijection, with a global scale parameter, but an elementwise location
        parameter? We could achieve this as follows.

            >>> from jax.tree_util import tree_map
            >>> bijection = Affine(jnp.zeros(()), jnp.ones(()))
            >>> bijection = eqx.tree_at(lambda bij: bij.loc, bijection, jnp.arange(3))
            >>> in_axis = tree_map(lambda _: None, bijection)
            >>> in_axis = eqx.tree_at(
            ...     lambda bij: bij.loc, in_axis, 0, is_leaf=lambda x: x is None
            ...     )
            >>> bijection = Vmap(bijection, in_axis=in_axis)
            >>> bijection.shape
            (3,)
            >>> bijection.bijection.loc.shape
            (3,)
            >>> bijection.bijection.scale.shape
            ()

            >>> x = jnp.ones(3)
            >>> bijection.transform(x)
            Array([1., 2., 3.], dtype=float32)

    """

    bijection: AbstractBijection
    in_axes: tuple
    axis_size: int

    def __init__(
        self,
        bijection: AbstractBijection,
        in_axis: int | None | Callable = None,
        axis_size: int | None = None,
        in_axis_condition: int | None = None,
    ):
        """Initialize the bijection.

        Args:
            bijection (AbstractBijection): The bijection to vectorize.
            in_axis (int | None | Callable): Specify which axes of the bijection
                parameters to vectorise over. It should be a PyTree of ``None``, ``int``
                with the tree structure being a prefix of the bijection, or a callable
                mapping ``Leaf -> Union[None, int]``. Defaults to None.
            axis_size (int, optional): The size of the new axis. This should be left
                unspecified if in_axis is provided, as the size can be inferred from the
                bijection parameters. Defaults to None.
            in_axis_condition (int | None, optional): Optionally define an axis of
                the conditioning variable to vectorize over. Defaults to None.
        """
        if in_axis is not None and axis_size is not None:
            raise ValueError("Cannot specify both in_axis and axis_size.")

        if axis_size is None:
            if in_axis is None:
                raise ValueError("Either axis_size or in_axis must be provided.")
            axis_size = _infer_axis_size_from_params(bijection, in_axis)

        self.in_axes = (in_axis, 0, in_axis_condition)
        self.bijection = bijection
        self.axis_size = axis_size

    def transform(self, x, condition=None):
        def _transform(bijection, x, condition):
            return bijection.transform(x, condition)

        return eqx.filter_vmap(_transform, in_axes=self.in_axes)(
            self.bijection,
            x,
            condition,
        )

    def transform_and_log_det(self, x, condition=None):
        def _transform_and_log_det(bijection, x, condition):
            return bijection.transform_and_log_det(x, condition)

        y, log_det = eqx.filter_vmap(_transform_and_log_det, in_axes=self.in_axes)(
            self.bijection,
            x,
            condition,
        )
        return y, jnp.sum(log_det)

    def inverse(self, y, condition=None):
        def _inverse(bijection, x, condition):
            return bijection.inverse(x, condition)

        return eqx.filter_vmap(_inverse, in_axes=self.in_axes)(
            self.bijection,
            y,
            condition,
        )

    def inverse_and_log_det(self, y, condition=None):
        def _inverse_and_log_det(bijection, x, condition):
            return bijection.inverse_and_log_det(x, condition)

        x, log_det = eqx.filter_vmap(_inverse_and_log_det, in_axes=self.in_axes)(
            self.bijection,
            y,
            condition,
        )
        return x, jnp.sum(log_det)

    @property
    def shape(self):
        return (self.axis_size, *self.bijection.shape)

    @property
    def cond_shape(self):
        ax = self.in_axes[2]
        if self.bijection.cond_shape is None or ax is None:
            return self.bijection.cond_shape

        return (
            *self.bijection.cond_shape[:ax],
            self.axis_size,
            *self.bijection.cond_shape[ax:],
        )


def _infer_axis_size_from_params(tree, in_axis):
    axes = _resolve_vmapped_axes(tree, in_axis)
    axis_sizes = tree_leaves(
        tree_map(
            lambda leaf, ax: leaf.shape[ax] if ax is not None else None,
            tree,
            axes,
        ),
    )
    if len(axis_sizes) == 0:
        raise ValueError("in_axis did not map to any leaves to vectorize.")
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
