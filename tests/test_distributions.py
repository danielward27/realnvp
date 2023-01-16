import jax.numpy as jnp
import pytest
import jax.random as jr
from flowjax.distributions import (
    Cauchy,
    Gumbel,
    Normal,
    StandardNormal,
    StudentT,
    Uniform,
    _StandardCauchy,
    _StandardGumbel,
    _StandardStudentT,
    _StandardUniform,
    Distribution
)


# This sets up a number of constructors shape -> instance for testing
# the generic API of Distribution classes.
# Note we do not test the private "Standard" distributions, assuming they are
# sufficiently tested by their loc, scale public counterparts.

_test_distributions = {
    # flowjax.distributions
    "StandardNormal": lambda shape: StandardNormal(shape),
    "Normal": lambda shape: Normal(jnp.zeros(shape)),
    "_StandardUniform": lambda shape: _StandardUniform(shape),
    "Uniform": lambda shape: Uniform(jnp.zeros(shape), 1),
    "_StandardGumbel": _StandardGumbel,
    "Gumbel": lambda shape: Gumbel(jnp.zeros(shape)),
    "_StandardCauchy": _StandardCauchy,
    "Cauchy": lambda shape: Cauchy(jnp.zeros(shape)),
    "_StandardStudentT": lambda shape: _StandardStudentT(jnp.ones(shape)),
    "StudentT": lambda shape: StudentT(jnp.ones(shape)),
}


_test_distributions = [pytest.param(v, id=k) for k, v in _test_distributions.items()]
_test_shapes = [(), (2,), (2, 3)]


@pytest.mark.parametrize("distribution", _test_distributions)
@pytest.mark.parametrize("shape", _test_shapes)
def test_sample(distribution, shape):
    d = distribution(shape=shape)
    sample = d.sample(jr.PRNGKey(0))
    assert sample.shape == shape

    sample_shape = (2, 2)
    sample = d.sample(jr.PRNGKey(0), sample_shape=sample_shape)
    assert sample.shape == sample_shape + shape


@pytest.mark.parametrize("distribution", _test_distributions)
@pytest.mark.parametrize("shape", _test_shapes)
def test_log_prob(distribution, shape):
    d = distribution(shape=shape)
    x = d.sample(jr.PRNGKey(0))

    assert d.log_prob(x).shape == ()

    sample_shape = (2, 3)
    x = d.sample(jr.PRNGKey(0), sample_shape=sample_shape)
    assert d.log_prob(x).shape == sample_shape


@pytest.mark.parametrize("distribution", _test_distributions)
def test_log_prob_shape_mismatch(distribution):
    d = distribution(shape=(3,))

    with pytest.raises(ValueError):
        d.log_prob(jnp.ones((3, 2)))

    d = distribution(shape=(3, 2))
    with pytest.raises(ValueError):
        d.log_prob(jnp.ones((2,)))

def test_uniform_params():
    dist = Uniform(
        jnp.array([1.0, 2.0]),
        jnp.array([3.0, 4.0]),
    )

    assert dist.minval == pytest.approx(jnp.array([1.0, 2.0]))
    assert dist.maxval == pytest.approx(jnp.array([3.0, 4.0]))




# Since the broadcasting behaviour is shared by all, we test it for a single unconditional and conditional distribution only

dist_shape, sample_shape, condition_shape = [[(), (2,), (3,4)] for _ in range(3)]

class TestDist(Distribution):
    "Toy distribution object, for testing of distribution broadcasting."

    def __init__(self, shape, cond_shape = None) -> None:
        self.shape = shape
        self.cond_shape = cond_shape

    def _log_prob(self, x, condition = None):
        return jnp.zeros(())

    def _sample(self, key, condition = None):
        return jnp.zeros(self.shape)


@pytest.mark.parametrize("dist_shape", dist_shape)
@pytest.mark.parametrize("sample_shape", sample_shape)
def test_broadcasting_unconditional(dist_shape, sample_shape):
    d = TestDist(dist_shape)
    samples = d.sample(jr.PRNGKey(0), sample_shape=sample_shape)
    assert samples.shape == sample_shape + dist_shape

    log_probs = d.log_prob(samples)
    assert log_probs.shape == sample_shape

    with pytest.raises(ValueError):
        d.sample(jr.PRNGKey(0), condition=jnp.ones(3), sample_shape=sample_shape)

    with pytest.raises(ValueError):
        d.log_prob(samples, condition=jnp.ones(3))



@pytest.mark.parametrize("dist_shape", dist_shape)
@pytest.mark.parametrize("sample_shape", sample_shape)
@pytest.mark.parametrize("condition_shape", condition_shape)
def test_broadcasting_conditional_sample(dist_shape, sample_shape, condition_shape):

    key = jr.PRNGKey(0)
    d = TestDist(dist_shape, condition_shape)

    # No leading dimensions in condition
    condition = jnp.zeros(condition_shape) 
    samples = d.sample(key, condition=condition, sample_shape=sample_shape)
    assert samples.shape == sample_shape + dist_shape

    log_probs = d.log_prob(samples, condition)
    assert log_probs.shape == sample_shape

    # Leading dimensions in condition
    leading = (3,4)
    condition = jnp.zeros(leading + condition_shape) 
    samples = d.sample(key, condition=condition, sample_shape=sample_shape)
    assert samples.shape == sample_shape + leading + dist_shape

    log_probs = d.log_prob(samples, condition)
    assert log_probs.shape == sample_shape + leading
