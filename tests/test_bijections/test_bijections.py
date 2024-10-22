"General tests for bijections (including transformers)."

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
import numpy as np
from scipy import stats

from flowjax.bijections import (
    AbstractBijection,
    AdditiveCondition,
    Affine,
    BlockAutoregressiveNetwork,
    Chain,
    Concatenate,
    Coupling,
    EmbedCondition,
    Exp,
    Flip,
    Identity,
    LeakyTanh,
    Loc,
    MaskedAutoregressive,
    Partial,
    Permute,
    Planar,
    Power,
    RationalQuadraticSpline,
    Reshape,
    Scale,
    Scan,
    Sigmoid,
    SoftPlus,
    Stack,
    Tanh,
    TriangularAffine,
    Vmap,
)
from flowjax.bijections.planar import _UnconditionalPlanar

DIM = 3
COND_DIM = 2
KEY = jr.key(0)


bijections = {
    "Identity": lambda: Identity((DIM,)),
    "Flip": lambda: Flip((DIM,)),
    "Permute": lambda: Permute(jnp.flip(jnp.arange(DIM))),
    "Permute (3D)": lambda: Permute(
        jnp.reshape(jr.permutation(KEY, jnp.arange(2 * 3 * 4)), (2, 3, 4)),
    ),
    "Partial (int)": lambda: Partial(Affine(jnp.array(2), jnp.array(2)), 0, (DIM,)),
    "Partial (bool array)": lambda: Partial(
        Flip((2,)),
        jnp.array([True, False, True]),
        (DIM,),
    ),
    "Partial (int array)": lambda: Partial(Flip((2,)), jnp.array([0, 2]), (DIM,)),
    "Partial (slice)": lambda: Partial(Affine(jnp.zeros(2)), slice(0, 2), (DIM,)),
    "Affine": lambda: Affine(jnp.ones(DIM), jnp.full(DIM, 2)),
    "Affine (pos and neg scales)": lambda: eqx.tree_at(
        lambda aff: aff.scale, Affine(scale=jnp.ones(3)), jnp.array([-1, 1, -2])
    ),
    "Tanh": lambda: Tanh((DIM,)),
    "LeakyTanh": lambda: LeakyTanh(1, (DIM,)),
    "LeakyTanh (broadcast max_val)": lambda: LeakyTanh(1, (2, 3)),
    "Loc": lambda: Loc(jnp.arange(DIM)),
    "Exp": lambda: Exp((DIM,)),
    "Sigmoid": lambda: Sigmoid((DIM,)),
    "SoftPlus": lambda: SoftPlus((DIM,)),
    "TriangularAffine (lower)": lambda: TriangularAffine(
        jnp.arange(DIM),
        jnp.full((DIM, DIM), 0.5),
    ),
    "TriangularAffine (upper)": lambda: TriangularAffine(
        jnp.arange(DIM),
        jnp.full((DIM, DIM), 0.5),
        lower=False,
    ),
    "TriangularAffine (pos and neg diag)": lambda: eqx.tree_at(
        lambda triaff: triaff.triangular,
        TriangularAffine(
            jnp.arange(3),
            jnp.diag(jnp.ones(3)),
        ),
        jnp.diag(jnp.array([-1, 2, -3])),
    ),
    "RationalQuadraticSpline": lambda: RationalQuadraticSpline(knots=4, interval=1),
    "Coupling (unconditional)": lambda: Coupling(
        KEY,
        transformer=Affine(),
        untransformed_dim=DIM // 2,
        dim=DIM,
        nn_width=5,
        nn_depth=2,
    ),
    "Coupling (conditional)": lambda: Coupling(
        KEY,
        transformer=Affine(),
        untransformed_dim=DIM // 2,
        dim=DIM,
        cond_dim=COND_DIM,
        nn_width=5,
        nn_depth=2,
    ),
    "MaskedAutoregressive_Affine (unconditional)": lambda: MaskedAutoregressive(
        KEY,
        transformer=Affine(),
        dim=DIM,
        nn_width=5,
        nn_depth=2,
    ),
    "MaskedAutoregressive_Affine (conditional)": lambda: MaskedAutoregressive(
        KEY,
        transformer=Affine(),
        cond_dim=COND_DIM,
        dim=DIM,
        nn_width=5,
        nn_depth=2,
    ),
    "MaskedAutoregressiveRationalQuadraticSpline (unconditional)": (
        lambda: MaskedAutoregressive(
            KEY,
            transformer=RationalQuadraticSpline(knots=5, interval=3),
            dim=DIM,
            nn_width=10,
            nn_depth=2,
        )
    ),
    "BlockAutoregressiveNetwork (unconditional)": lambda: BlockAutoregressiveNetwork(
        KEY,
        dim=DIM,
        block_dim=3,
        depth=1,
    ),
    "BlockAutoregressiveNetwork (conditional)": lambda: BlockAutoregressiveNetwork(
        KEY,
        dim=DIM,
        cond_dim=COND_DIM,
        block_dim=3,
        depth=1,
    ),
    "AdditiveCondtition": lambda: AdditiveCondition(
        lambda condition: jnp.arange(DIM) * jnp.sum(condition),
        (DIM,),
        (COND_DIM,),
    ),
    "EmbedCondition": lambda: EmbedCondition(
        BlockAutoregressiveNetwork(KEY, dim=DIM, cond_dim=1, block_dim=3, depth=1),
        eqx.nn.MLP(2, 1, 3, 1, key=KEY),
        (COND_DIM,),  # Raw
    ),
    "Chain": lambda: Chain([Flip((DIM,)), Affine(jnp.ones(DIM), jnp.full(DIM, 2))]),
    "Scan": lambda: Scan(eqx.filter_vmap(Affine)(jnp.ones((2, DIM)))),
    "Scale": lambda: Scale(jnp.full(DIM, 2)),
    "Scale (pos and neg scales)": lambda: eqx.tree_at(
        lambda scale: scale.scale,
        Scale(jnp.ones(3)),
        jnp.array([-1, 2, -3]),
    ),
    "Concatenate": lambda: Concatenate([Affine(jnp.ones(DIM)), Tanh(shape=(DIM,))]),
    "ConcatenateAxis1": lambda: Concatenate(
        [Affine(jnp.ones((3, 3))), Tanh(shape=((3, 3)))],
        axis=1,
    ),
    "ConcatenateAxis-1": lambda: Concatenate(
        [Affine(jnp.ones((3, 3))), Tanh(shape=((3, 3)))],
        axis=-1,
    ),
    "Stack": lambda: Stack([Tanh(()), Affine(), Tanh(())]),
    "StackAxis1": lambda: Stack([Tanh((2,)), Affine(jnp.ones(2)), Tanh((2,))], axis=1),
    "StackAxis-1": lambda: Stack(
        [Affine(jr.uniform(k, (1, 2, 3))) for k in jr.split(KEY, 3)],
        axis=-1,
    ),
    "_UnconditionalPlanar (leaky_relu +ve bias)": lambda: _UnconditionalPlanar(
        *jnp.split(jr.normal(KEY, (8,)), 2),
        bias=jnp.array(100.0),  # leads to evaluation in +ve relu portion
        negative_slope=0.1,
    ),
    "_UnconditionalPlanar (leaky_relu -ve bias)": lambda: _UnconditionalPlanar(
        *jnp.split(jr.normal(KEY, (8,)), 2),
        bias=-jnp.array(100.0),  # leads to evaluation in -ve relu portion
        negative_slope=0.1,
    ),
    "Planar": lambda: Planar(
        KEY,
        dim=DIM,
    ),
    "Power": lambda: Power(3, (2, 2)),
    "Vmap (broadcast params)": lambda: Vmap(Affine(1, 2), axis_size=10),
    "Vmap (vectorize params)": lambda: Vmap(
        eqx.filter_vmap(Affine)(jnp.ones(3)),
        in_axes=eqx.if_array(0),
    ),
    "Reshape (unconditional)": lambda: Reshape(Affine(scale=jnp.arange(1, 5)), (2, 2)),
    "Reshape (conditional)": lambda: Reshape(
        MaskedAutoregressive(
            KEY,
            transformer=Affine(),
            dim=4,
            cond_dim=1,
            nn_width=3,
            nn_depth=1,
        ),
        shape=(1, 4, 1),
        cond_shape=(),
    ),
}


POSITIVE_DOMAIN = [Power]


@pytest.mark.parametrize("constructor", bijections.values(), ids=bijections.keys())
def test_transform_inverse(constructor):
    """Tests transform and inverse methods."""
    bijection = constructor()
    shape = bijection.shape if bijection.shape is not None else (DIM,)
    x = jr.normal(jr.key(0), shape)

    if type(bijection) in POSITIVE_DOMAIN:
        x = jnp.exp(x)

    if bijection.cond_shape is not None:
        cond = jr.normal(jr.key(0), bijection.cond_shape)
    else:
        cond = None
    y = bijection.transform(x, cond)
    try:
        x_reconstructed = bijection.inverse(y, cond)
        assert x_reconstructed == pytest.approx(x, abs=1e-4)
    except NotImplementedError:
        pass


@pytest.mark.parametrize("constructor", bijections.values(), ids=bijections.keys())
def test_transform_inverse_and_log_dets(constructor):
    """Tests the transform_and_log_det and inverse_and_log_det methods,
    by 1) checking invertibility and 2) comparing log dets to those obtained with
    automatic differentiation.
    """
    bijection = constructor()
    shape = bijection.shape if bijection.shape is not None else (DIM,)
    x = jr.normal(jr.key(0), shape)

    if type(bijection) in POSITIVE_DOMAIN:
        x = jnp.exp(x)

    if bijection.cond_shape is not None:
        cond = jr.normal(jr.key(0), bijection.cond_shape)
    else:
        cond = None

    # We flatten the function so auto_jacobian is calculated correctly
    def flat_transform(x_flat, cond):
        x = x_flat.reshape(shape)
        y = bijection.transform(x, cond)
        return y.ravel()

    auto_jacobian = jax.jacobian(flat_transform)(x.ravel(), cond)
    auto_log_det = jnp.log(jnp.abs(jnp.linalg.det(auto_jacobian)))
    y, logdet = bijection.transform_and_log_det(x, cond)
    assert logdet == pytest.approx(auto_log_det, abs=1e-4)
    assert y == pytest.approx(bijection.transform(x, cond))

    try:
        x_reconstructed, logdetinv = bijection.inverse_and_log_det(y, cond)
        assert logdetinv == pytest.approx(-auto_log_det, abs=1e-4)
        assert x == pytest.approx(x_reconstructed, abs=1e-4)
        assert x == pytest.approx(bijection.inverse(y, cond), abs=1e-4)

    except NotImplementedError:
        pass


@pytest.mark.parametrize("bijection_name", bijections.keys())
def test_inverse_gradient_and_val(bijection_name):
    bijection = bijections[bijection_name]()
    shape = bijection.shape if bijection.shape is not None else (DIM,)
    y = jr.normal(jr.PRNGKey(0), shape)

    if type(bijection) in POSITIVE_DOMAIN:
        y = jnp.abs(y)

    if bijection.cond_shape is not None:
        cond = jr.normal(jr.PRNGKey(0), bijection.cond_shape)
    else:
        cond = None

    y_grad = -y
    y_logp = jnp.array(jax.scipy.stats.norm.logpdf(y).sum())

    try:
        x, x_grad, x_logp = AbstractBijection.inverse_gradient_and_val(
            bijection,
            y,
            y_grad,
            y_logp,
            cond,
        )

        x_actual, x_grad_actual, x_logp_actual = bijection.inverse_gradient_and_val(
            y, y_grad, y_logp, cond
        )

        np.testing.assert_allclose(x, x_actual)
        np.testing.assert_allclose(x_grad, x_grad_actual)
        np.testing.assert_allclose(x_logp, x_logp_actual)
    except NotImplementedError:
        pass


def test_affine_inverse_gradient_and_val():
    affine = Affine(scale=3.0, loc=1.0)

    rng = np.random.default_rng(0)
    x = jnp.array(rng.normal())

    x_grad = -x
    x_logp = jnp.array(stats.norm(scale=1, loc=0).logpdf(x))

    y = affine.transform(x)
    y_logp = jnp.array(stats.norm(scale=3, loc=1).logpdf(y))
    y_grad = -(y - 1) / 9

    expected = np.array([x, x_grad, x_logp])
    actual = np.array(
        affine.inverse_gradient_and_val(jnp.array(y), jnp.array(y_grad), y_logp)
    )
    np.testing.assert_allclose(expected, actual, rtol=1e-6)


def test_exp_inverse_gradient_and_val():
    exp = Exp()

    rng = np.random.default_rng(0)
    x = jnp.array(rng.normal())

    x_grad = -x
    x_logp = jnp.array(stats.norm(scale=1, loc=0).logpdf(x))

    y = exp.transform(x)
    y_logp = jnp.array(stats.lognorm(s=1, scale=1).logpdf(y))

    # Compute numerical derivative in float64
    y_f64 = np.array(y)
    y_grad = jnp.array(
        (
            stats.lognorm(s=1, scale=1).logpdf(y_f64 + 1e-8)
            - stats.lognorm(s=1, scale=1).logpdf(y_f64)
        )
        / 1e-8
    )

    expected = np.array([x, x_grad, x_logp])
    actual = np.array(exp.inverse_gradient_and_val(y, y_grad, y_logp))
    np.testing.assert_allclose(expected, actual, rtol=1e-6)


class _AstractTestBijection(AbstractBijection):
    shape: tuple[int, ...] = ()
    cond_shape: tuple[int, ...] | None = None

    def transform(self, x, condition=None):
        return x

    def transform_and_log_det(self, x, condition=None):
        return x, jnp.zeros(())

    @abstractmethod
    def inverse(self, y, condition=None):
        pass

    @abstractmethod
    def inverse_and_log_det(self, y, condition=None):
        return y, jnp.zeros(())

    def inverse_gradient_and_val(self, y, y_grad, y_logp, condition=None):
        return y, y_grad, jnp.zeros(())


class _TestBijection(_AstractTestBijection):
    # Test bijection (with inheritance + method overide)
    def transform(self, x, condition=None):  # Check overiding
        return x

    def inverse(self, y, condition=None):
        return y

    def inverse_and_log_det(self, y, condition=None):
        return y, jnp.zeros(())


test_cases = {
    "wrong x shape": (
        [(), None],
        [jnp.ones(2), None],
        "Expected input shape",
    ),
    "wrong condition shape": (
        [(), (2,)],
        [1, jnp.ones(3)],
        "Expected condition.shape",
    ),
    "missing condition": (
        [(), (2,)],
        [1, None],
        "Expected condition to be provided.",
    ),
}


@pytest.mark.parametrize(
    ("args", "inputs", "match"),
    test_cases.values(),
    ids=test_cases.keys(),
)
def test_argcheck(args, inputs, match):
    bijection = _TestBijection(*args)

    with pytest.raises(ValueError, match=match):
        bijection.transform(*inputs)

    with pytest.raises(ValueError, match=match):
        bijection.inverse(*inputs)

    with pytest.raises(ValueError, match=match):
        bijection.transform_and_log_det(*inputs)

    with pytest.raises(ValueError, match=match):
        bijection.inverse_and_log_det(*inputs)
