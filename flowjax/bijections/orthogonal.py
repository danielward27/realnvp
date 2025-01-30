from flowjax.bijections.bijection import AbstractBijection
from jax import Array
import jax.numpy as jnp
import jax.nn as jnn
from jax.scipy import fft


class Neg(AbstractBijection):
    """A bijection that negates its input (multiplies by -1).

    This is a simple bijection that flips the sign of all elements in the input array.

    Attributes:
        shape: Shape of the input/output arrays
        cond_shape: Shape of conditional inputs (None as this bijection is unconditional)
    """
    shape: tuple[int, ...]
    cond_shape = None

    def __init__(self, shape):
        self.shape = shape

    def transform_and_log_det(self, x: jnp.ndarray, condition: Array | None = None):
        return -x, jnp.zeros(())

    def inverse_and_log_det(self, y: Array, condition: Array | None = None):
        return -y, jnp.zeros(())


class Householder(AbstractBijection):
    """A Householder reflection bijection.

    This bijection implements a Householder reflection, which is a linear
    transformation that reflects vectors across a hyperplane defined by a normal
    vector (params). The transformation is its own inverse and volume-preserving
    (determinant = ±1).

    Given a unit vector v, the transformation is:
    x → x - 2(x·v)v

    Attributes:
        shape: Shape of the input/output vectors
        cond_shape: Shape of conditional inputs (None as this bijection is unconditional)
        params: Normal vector defining the reflection hyperplane. The vector is
            normalized in the transformation, so scaling params will have no effect
            on the bijection.
    """
    shape: tuple[int, ...]
    params: Array
    cond_shape = None

    def __init__(self, params: Array):
        self.shape = (params.shape[-1],)
        self.params = params

    def _householder(self, x: Array, params: Array) -> Array:
        norm_sq = params @ params
        norm = jnp.sqrt(norm_sq)

        vec = params / norm
        return x - 2 * vec * (x @ vec)

    def transform_and_log_det(self, x: jnp.ndarray, condition: Array | None = None):
        return self._householder(x, self.params), jnp.zeros(())

    def inverse_and_log_det(self, y: Array, condition: Array | None = None):
        return self._householder(y, self.params), jnp.zeros(())

    def inverse_gradient_and_val(
        self,
        y: Array,
        y_grad: Array,
        y_logp: Array,
        condition: Array | None = None,
    ) -> tuple[Array, Array, Array]:
        x, logdet = self.inverse_and_log_det(y)
        x_grad = self._householder(y_grad, params=self.params)
        return (x, x_grad, y_logp - logdet)


class DCT(AbstractBijection):
    """Discrete Cosine Transform (DCT) bijection.

    This bijection applies the DCT or its inverse along a specified axis.

    Attributes:
        shape: Shape of the input/output arrays
        cond_shape: Shape of conditional inputs (None as this bijection is unconditional)
        axis: Axis along which to apply the DCT
        norm: Normalization method, fixed to 'ortho' to ensure bijectivity
    """

    shape: tuple[int, ...]
    cond_shape = None
    axis: int
    norm: str

    def __init__(self, shape, *, axis: int = -1):
        self.shape = shape
        self.axis = axis
        self.norm = "ortho"

    def _dct(self, x: Array, inverse: bool = False) -> Array:
        if inverse:
            z = fft.idct(x, norm=self.norm, axis=self.axis)
        else:
            z = fft.dct(x, norm=self.norm, axis=self.axis)

        return z

    def transform_and_log_det(self, x: jnp.ndarray, condition: Array | None = None):
        y = self._dct(x)
        return y, jnp.zeros(())

    def inverse_and_log_det(self, y: Array, condition: Array | None = None):
        x = self._dct(y, inverse=True)
        return x, jnp.zeros(())
