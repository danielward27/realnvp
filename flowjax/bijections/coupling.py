"""Implemenetation of Coupling flow layer with arbitrary transformer.

Ref: https://arxiv.org/abs/1605.08803.
"""

from collections.abc import Callable

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax
import paramax
from jaxtyping import PRNGKeyArray

from flowjax.bijections.bijection import AbstractBijection
from flowjax.bijections.jax_transforms import Vmap
from flowjax.utils import Array, get_ravelled_pytree_constructor


class Coupling(AbstractBijection):
    """Coupling layer implementation (https://arxiv.org/abs/1605.08803).

    Args:
        key: Jax key
        transformer: Unconditional bijection with shape () to be parameterised by the
            conditioner neural netork. Parameters wrapped with ``NonTrainable``
            are excluded from being parameterized.
        untransformed_dim: Number of untransformed conditioning variables (e.g. dim//2).
        dim: Total dimension.
        cond_dim: Dimension of additional conditioning variables. Defaults to None.
        nn_width: Neural network hidden layer width.
        nn_depth: Neural network hidden layer size.
        nn_activation: Neural network activation function. Defaults to jnn.relu.
    """

    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    untransformed_dim: int
    dim: int
    transformer_constructor: Callable
    conditioner: eqx.nn.MLP

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        transformer: AbstractBijection,
        untransformed_dim: int,
        dim: int,
        cond_dim: int | None = None,
        nn_width: int,
        nn_depth: int,
        nn_activation: Callable = jnn.relu,
    ):
        if transformer.shape != () or transformer.cond_shape is not None:
            raise ValueError(
                "Only unconditional transformers with shape () are supported.",
            )

        constructor, num_params = get_ravelled_pytree_constructor(
            transformer,
            filter_spec=eqx.is_inexact_array,
            is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
        )

        self.transformer_constructor = constructor
        self.untransformed_dim = untransformed_dim
        self.dim = dim
        self.shape = (dim,)
        self.cond_shape = (cond_dim,) if cond_dim is not None else None

        conditioner_output_size = num_params * (dim - untransformed_dim)

        self.conditioner = eqx.nn.MLP(
            in_size=(
                untransformed_dim if cond_dim is None else untransformed_dim + cond_dim
            ),
            out_size=conditioner_output_size,
            width_size=nn_width,
            depth=nn_depth,
            activation=nn_activation,
            key=key,
        )

    def transform(self, x, condition=None):
        x_cond, x_trans = x[: self.untransformed_dim], x[self.untransformed_dim :]
        nn_input = x_cond if condition is None else jnp.hstack((x_cond, condition))
        transformer_params = self.conditioner(nn_input)
        transformer = self._flat_params_to_transformer(transformer_params)
        y_trans = transformer.transform(x_trans)
        return jnp.hstack((x_cond, y_trans))

    def transform_and_log_det(self, x, condition=None):
        x_cond, x_trans = x[: self.untransformed_dim], x[self.untransformed_dim :]
        nn_input = x_cond if condition is None else jnp.hstack((x_cond, condition))
        transformer_params = self.conditioner(nn_input)
        transformer = self._flat_params_to_transformer(transformer_params)
        y_trans, log_det = transformer.transform_and_log_det(x_trans)
        y = jnp.hstack((x_cond, y_trans))
        return y, log_det

    def inverse(self, y, condition=None):
        x_cond, y_trans = y[: self.untransformed_dim], y[self.untransformed_dim :]
        nn_input = x_cond if condition is None else jnp.concatenate((x_cond, condition))
        transformer_params = self.conditioner(nn_input)
        transformer = self._flat_params_to_transformer(transformer_params)
        x_trans = transformer.inverse(y_trans)
        return jnp.hstack((x_cond, x_trans))

    def inverse_and_log_det(self, y, condition=None):
        x_cond, y_trans = y[: self.untransformed_dim], y[self.untransformed_dim :]
        nn_input = x_cond if condition is None else jnp.concatenate((x_cond, condition))
        transformer_params = self.conditioner(nn_input)
        transformer = self._flat_params_to_transformer(transformer_params)
        x_trans, log_det = transformer.inverse_and_log_det(y_trans)
        x = jnp.hstack((x_cond, x_trans))
        return x, log_det

    def _flat_params_to_transformer(self, params: Array):
        """Reshape to dim X params_per_dim, then vmap."""
        dim = self.dim - self.untransformed_dim
        transformer_params = jnp.reshape(params, (dim, -1))
        transformer = eqx.filter_vmap(self.transformer_constructor)(transformer_params)
        return Vmap(transformer, in_axes=eqx.if_array(0))

    def inverse_gradient_and_val(
        self,
        y: Array,
        y_grad: Array,
        y_logp: Array,
        condition: Array | None = None,
    ) -> tuple[Array, Array, Array]:
        y_cond, y_trans = y[: self.untransformed_dim], y[self.untransformed_dim :]
        x_cond = y_cond

        y_grad_cond, y_grad_trans = (
            y_grad[: self.untransformed_dim],
            y_grad[self.untransformed_dim :],
        )

        def conditioner(x_cond):
            nn_input = (
                x_cond if condition is None else jnp.concatenate((x_cond, condition))
            )
            return self.conditioner(nn_input)

        transformer_params, nn_pull = jax.vjp(conditioner, x_cond)

        def pull_transformer_grad(transformer_params):
            transformer = self._flat_params_to_transformer(transformer_params)

            x_trans, x_grad_trans, x_logp = transformer.inverse_gradient_and_val(
                y_trans, y_grad_trans, y_logp
            )

            return (x_logp, x_trans), x_grad_trans

        ((x_logp, x_trans), pull_pull_transformer_grad, x_grad_trans) = jax.vjp(
            pull_transformer_grad, transformer_params, has_aux=True
        )

        (co_transformer_params,) = pull_pull_transformer_grad((1.0, -x_grad_trans))
        (co_x_cond,) = nn_pull(co_transformer_params)

        x = jnp.hstack((x_cond, x_trans))
        x_grad = jnp.hstack((y_grad_cond + co_x_cond, x_grad_trans))
        return x, x_grad, x_logp
