"""Premade versions of common flow architetctures from ``flowjax.flows``."""
# Note that here although we could chain arbitrary bijections using `Chain`, here,
# we generally opt to use `Scan`, which avoids excessive compilation
# when the flow layers share the same structure.

from typing import Callable

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from equinox.nn import Linear
from jax.nn.initializers import glorot_uniform

from flowjax.bijections import (
    AdditiveCondition,
    Bijection,
    BlockAutoregressiveNetwork,
    Chain,
    Coupling,
    Flip,
    Invert,
    LeakyTanh,
    MaskedAutoregressive,
    Permute,
    Planar,
    RationalQuadraticSpline,
    Scan,
    TriangularAffine,
)
from flowjax.distributions import Distribution, Transformed


class CouplingFlow(Transformed):
    """Coupling flow (https://arxiv.org/abs/1605.08803)."""

    flow_layers: int
    nn_width: int
    nn_depth: int
    permute_strategy: str

    def __init__(
        self,
        key: jr.KeyArray,
        base_dist: Distribution,
        transformer: Bijection,
        cond_dim: int | None = None,
        flow_layers: int = 8,
        nn_width: int = 40,
        nn_depth: int = 2,
        nn_activation: Callable = jnn.relu,
        invert: bool = True,
    ):
        """
        Args:
            key (jr.KeyArray): Jax PRNGKey.
            base_dist (Distribution): Base distribution.
            transformer (Bijection): Bijection to be parameterised by conditioner.
            cond_dim (int): Dimension of conditioning variables. Defaults to None.
            flow_layers (int): Number of coupling layers. Defaults to 5.
            nn_width (int): Conditioner hidden layer size. Defaults to 40.
            nn_depth (int): Conditioner depth. Defaults to 2.
            nn_activation (int): Conditioner activation function. Defaults to jnn.relu.
            invert: (bool): Whether to invert the bijection. Broadly, True will
                prioritise a faster `inverse` methods, leading to faster `log_prob`,
                False will prioritise faster `transform` methods, leading to faster
                `sample`. Defaults to True
        """
        if base_dist.ndim != 1:
            raise ValueError(f"Expected base_dist.ndim==1, got {base_dist.ndim}")

        dim = base_dist.shape[0]

        def make_layer(key):  # coupling layer + permutation
            bij_key, perm_key = jr.split(key)
            bijection = Coupling(
                key=bij_key,
                transformer=transformer,
                untransformed_dim=dim // 2,
                dim=dim,
                cond_dim=cond_dim,
                nn_width=nn_width,
                nn_depth=nn_depth,
                nn_activation=nn_activation,
            )
            bijection = _add_default_permute(bijection, dim, perm_key)
            return bijection

        keys = jr.split(key, flow_layers)
        layers = eqx.filter_vmap(make_layer)(keys)
        bijection = Invert(Scan(layers)) if invert else Scan(layers)

        self.nn_width = nn_width
        self.nn_depth = nn_depth
        self.flow_layers = flow_layers
        self.permute_strategy = _get_default_permute_name(dim)
        self.shape = (dim,)
        self.cond_shape = None if cond_dim is None else (cond_dim,)
        super().__init__(base_dist, bijection)


class MaskedAutoregressiveFlow(Transformed):
    """Masked autoregressive flow (https://arxiv.org/abs/1606.04934,
    https://arxiv.org/abs/1705.07057v4). Parameterises a transformer with an
    autoregressive neural network.
    """

    flow_layers: int
    nn_width: int
    nn_depth: int
    permute_strategy: str

    def __init__(
        self,
        key: jr.KeyArray,
        base_dist: Distribution,
        transformer: Bijection,
        cond_dim: int | None = None,
        flow_layers: int = 8,
        nn_width: int = 40,
        nn_depth: int = 2,
        nn_activation: Callable = jnn.relu,
        invert: bool = True,
    ):
        """
        Args:
            key (jr.KeyArray): Random seed.
            base_dist (Distribution): Base distribution.
            transformer (Bijection): Bijection parameterised by autoregressive network.
            cond_dim (int): _description_. Defaults to 0.
            flow_layers (int): Number of flow layers. Defaults to 5.
            nn_width (int): Number of hidden layers in neural network. Defaults to 40.
            nn_depth (int): Depth of neural network. Defaults to 2.
            nn_activation (Callable): _description_. Defaults to jnn.relu.
            invert (bool): Whether to invert the bijection. Broadly, True will
                prioritise a faster inverse, leading to faster `log_prob`, False will
                prioritise faster forward, leading to faster `sample`. Defaults to True.
        """
        if len(base_dist.shape) != 1:
            raise ValueError(f"Expected base_dist.ndim==1, got {base_dist.ndim}")

        dim = base_dist.shape[0]

        def make_layer(key):  # masked autoregressive layer + permutation
            bij_key, perm_key = jr.split(key)
            bijection = MaskedAutoregressive(
                key=bij_key,
                transformer=transformer,
                dim=dim,
                cond_dim=cond_dim,
                nn_width=nn_width,
                nn_depth=nn_depth,
                nn_activation=nn_activation,
            )
            bijection = _add_default_permute(bijection, dim, perm_key)
            return bijection

        keys = jr.split(key, flow_layers)
        layers = eqx.filter_vmap(make_layer)(keys)
        bijection = Invert(Scan(layers)) if invert else Scan(layers)

        self.nn_width = nn_width
        self.nn_depth = nn_depth
        self.flow_layers = flow_layers
        self.permute_strategy = _get_default_permute_name(dim)
        self.shape = (dim,)
        self.cond_shape = None if cond_dim is None else (cond_dim,)
        super().__init__(base_dist, bijection)


class BlockNeuralAutoregressiveFlow(Transformed):
    """Block neural autoregressive flow (BNAF) (https://arxiv.org/abs/1904.04676).
    Each flow layer contains a
    :py:class:`~flowjax.bijections.block_autoregressive_network.BlockAutoregressiveNetwork`
    bijection. The bijection does not have an analytic inverse, so either ``log_prob``
    or ``sample`` and ``sample_and_log_prob`` will be unavailable, controlled using the
    invert argument.
    """

    flow_layers: int
    nn_block_dim: int
    nn_depth: int
    permute_strategy: str

    def __init__(
        self,
        key: jr.KeyArray,
        base_dist: Distribution,
        cond_dim: int | None = None,
        nn_depth: int = 1,
        nn_block_dim: int = 8,
        flow_layers: int = 1,
        invert: bool = True,
        activation: Bijection | Callable | None = None,
    ):
        """
        Args:
            key (jr.KeyArray): Jax PRNGKey.
            base_dist (Distribution): Base distribution.
            cond_dim (int | None): Dimension of conditional variables.
            nn_depth (int): Number of hidden layers within the networks.
                Defaults to 1.
            nn_block_dim (int): Block size. Hidden layer width is
                dim*nn_block_dim. Defaults to 8.
            flow_layers (int): Number of BNAF layers. Defaults to 1.
            invert: (bool): Use `True` for access of ``log_prob`` only (e.g.
                fitting by maximum likelihood), `False` for the forward direction
                (``sample`` and ``sample_and_log_prob``) only (e.g. for fitting
                variationally).
            activation: (Bijection | Callable | None). Activation function used within
                block neural autoregressive networks. Note this should be bijective and
                in some use cases should map real -> real. For more information, see
                :class:`~flowjax.bijections.block_autoregressive_network.BlockAutoregressiveNetwork`.
                Defaults to :py:class:`~flowjax.bijections.tanh.LeakyTanh`.
        """
        if len(base_dist.shape) != 1:
            raise ValueError(f"Expected base_dist.ndim==1, got {base_dist.ndim}")

        dim = base_dist.shape[-1]

        def make_layer(key):  # bnaf layer + permutation
            bij_key, perm_key = jr.split(key)
            bijection = BlockAutoregressiveNetwork(
                key=bij_key,
                dim=dim,
                cond_dim=cond_dim,
                depth=nn_depth,
                block_dim=nn_block_dim,
                activation=activation,
            )
            bijection = _add_default_permute(bijection, dim, perm_key)
            return bijection

        keys = jr.split(key, flow_layers)
        layers = eqx.filter_vmap(make_layer)(keys)
        bijection = Invert(Scan(layers)) if invert else Scan(layers)

        self.nn_block_dim = nn_block_dim
        self.nn_depth = nn_depth
        self.flow_layers = flow_layers
        self.permute_strategy = _get_default_permute_name(dim)
        self.shape = (dim,)
        self.cond_shape = None if cond_dim is None else (cond_dim,)
        super().__init__(base_dist, bijection)


class TriangularSplineFlow(Transformed):
    """A stack of layers, where each layer consists of a triangular affine
    transformation with weight normalisation, and an elementwise rational quadratic
    spline. Tanh is used to constrain to the input to [-1, 1] before spline
    transformations.
    """

    flow_layers: int
    permute_strategy: str
    knots: int
    tanh_max_val: float

    def __init__(
        self,
        key: jr.KeyArray,
        base_dist: Distribution,
        cond_dim: int | None = None,
        flow_layers: int = 8,
        knots: int = 8,
        tanh_max_val: float = 3.0,
        invert: bool = True,
        init: Callable | None = None,
    ):
        """
        Args:
            key (jr.KeyArray): Jax random seed.
            base_dist (Distribution): Base distribution of the flow.
            cond_dim (int | None): The number of conditioning features.
                Defaults to None.
            flow_layers (int): Number of flow layers. Defaults to 8.
            knots (int): Number of knots in the splines. Defaults to 8.
            tanh_max_val (float): Maximum absolute value beyond which we use linear
                "tails" in the tanh function. Defaults to 3.0.
            invert: (bool): Use `True` for access of `log_prob` only (e.g.
                fitting by maximum likelihood), `False` for the forward direction
                (sampling) only (e.g. for fitting variationally).
            init (Callable | None): Initialisation method for the lower triangular
                weights. Defaults to glorot_uniform().
        """
        init = init if init is not None else glorot_uniform()
        if len(base_dist.shape) != 1:
            raise ValueError(f"Expected base_dist.ndim==1, got {base_dist.ndim}")

        dim = base_dist.shape[-1]

        def make_layer(key):
            lt_key, perm_key, cond_key = jr.split(key, 3)
            weights = init(lt_key, (dim, dim))
            lt_weights = weights.at[jnp.diag_indices(dim)].set(1)
            lower_tri = TriangularAffine(
                jnp.zeros(dim), lt_weights, weight_normalisation=True
            )

            bijections = [
                LeakyTanh(tanh_max_val, (dim,)),
                RationalQuadraticSpline(knots, interval=1, shape=(dim,)),
                Invert(LeakyTanh(tanh_max_val, (dim,))),
                lower_tri,
            ]

            if cond_dim is not None:
                linear_condition = AdditiveCondition(
                    Linear(cond_dim, dim, use_bias=False, key=cond_key),
                    (dim,),
                    (cond_dim,),
                )
                bijections.append(linear_condition)

            bijection = Chain(bijections)
            bijection = _add_default_permute(bijection, dim, perm_key)
            return bijection

        keys = jr.split(key, flow_layers)
        layers = eqx.filter_vmap(make_layer)(keys)
        bijection = Invert(Scan(layers)) if invert else Scan(layers)

        self.flow_layers = flow_layers
        self.permute_strategy = _get_default_permute_name(dim)
        self.knots = knots
        self.tanh_max_val = tanh_max_val
        self.shape = (dim,)
        self.cond_shape = None if cond_dim is None else (cond_dim,)

        super().__init__(base_dist, bijection)


class PlanarFlow(Transformed):
    """Planar flow as introduced in https://arxiv.org/pdf/1505.05770.pdf, alternating
    between :py:class:`~flowjax.bijections.planar.Planar` and permutations. Note the
    definition here is inverted compared to the original paper.
    """

    flow_layers: int
    permute_strategy: str

    def __init__(
        self,
        key: jr.KeyArray,
        base_dist: Distribution,
        cond_dim: int | None = None,
        flow_layers: int = 8,
        invert: bool = True,
        **mlp_kwargs,
    ):
        """
        Args:
            key (jr.KeyArray): Jax PRNGKey.
            base_dist (Distribution): Base distribution.
            cond_dim (int): Dimension of conditioning variables. Defaults to None.
            flow_layers (int): Number of flow layers. Defaults to 5.
            invert: (bool): Whether to invert the bijection. Broadly, True will
                prioritise a faster `inverse` methods, leading to faster `log_prob`,
                False will prioritise faster `transform` methods, leading to faster
                `sample`. Defaults to True
            **mlp_kwargs: Key word arguments to construct the MLP conditioner. Ignored
                if cond_dim is None.
        """
        if base_dist.ndim != 1:
            raise ValueError(f"Expected base_dist.ndim==1, got {base_dist.ndim}")

        dim = base_dist.shape[0]

        def make_layer(key):  # Planar layer + permutation
            bij_key, perm_key = jr.split(key)
            bijection = Planar(bij_key, dim, cond_dim, **mlp_kwargs)
            bijection = _add_default_permute(bijection, dim, perm_key)
            return bijection

        keys = jr.split(key, flow_layers)
        layers = eqx.filter_vmap(make_layer)(keys)
        bijection = Invert(Scan(layers)) if invert else Scan(layers)

        self.flow_layers = flow_layers
        self.permute_strategy = _get_default_permute_name(dim)
        self.shape = (dim,)
        self.cond_shape = None if cond_dim is None else (cond_dim,)
        super().__init__(base_dist, bijection)


def _add_default_permute(bijection: Bijection, dim: int, key: jr.KeyArray):
    if dim == 1:
        return bijection
    if dim == 2:
        return Chain([bijection, Flip((dim,))]).merge_chains()
    else:
        perm = Permute(jr.permutation(key, jnp.arange(dim)))
        return Chain([bijection, perm]).merge_chains()


def _get_default_permute_name(dim):
    return {1: "none", 2: "flip"}.get(dim, "random")
