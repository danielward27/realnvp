from flowjax.bijections.bijection import AbstractBijection
from jax import Array
import jax.numpy as jnp
import jax.nn as jnn
from jax.scipy import fft


class Neg(AbstractBijection):
    shape: tuple[int, ...]
    cond_shape = None

    def __init__(self, shape):
        """Initialize the MvScale bijection with `params`."""
        self.shape = shape

    def transform_and_log_det(self, x: jnp.ndarray, condition: Array | None = None):
        return -x, jnp.zeros(())

    def inverse_and_log_det(self, y: Array, condition: Array | None = None):
        return -y, jnp.zeros(())


class Householder(AbstractBijection):
    shape: tuple[int, ...]
    params: Array
    cond_shape = None

    def __init__(self, params: Array):
        """Initialize the MvScale bijection with `params`."""
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


class DCT(AbstractBijection):
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
