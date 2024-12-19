Experimental numpyro interface
===============================

Supporting complex inference approaches such as MCMC or variational inference
with arbitrary probabilistic models is out of the scope of this package. However,
we do provide some basic suppport for interfacing with numpyro. We note this support is
in its infancy and there may be breaking changes without warning. 

.. note::
    When converting a FlowJAX distribution to a numpyro distribution, the shape of 
    the distribution Flowjax distribution will correspond to the event shape. This
    is because there is no concept of ``batch_shape`` in flowjax distributions.
    However, batch dimensions in conditioning variables will be converted to a
    corresponding batch shape for the converted distribution.
    
In general, we can use a combination of FlowJAX and numpyro distributions in a
numpyro model by using :func:`~flowjax.experimental.numpyro.sample`, in place of
numpyro's ``sample``. This will wrap FlowJAX distributions to numpyro
distributions, using :func:`~flowjax.experimental.numpyro.distribution_to_numpyro`.
This approach can be used for example to embed normalising flows into arbitrary
probabilistic models. Here is a simple example

.. doctest::


    >>> from numpyro.infer import MCMC, NUTS
    >>> from flowjax.experimental.numpyro import sample
    >>> from flowjax.distributions import Normal
    >>> import jax.random as jr
    >>> import numpy as np

    >>> def numpyro_model(X, y):
    ...     "Example regression model defined in terms of FlowJAX distributions"
    ...     beta = sample("beta", Normal(np.zeros(2)))
    ...     sample("y", Normal(X @ beta), obs=y)

    >>> X = np.random.randn(100, 2)
    >>> beta_true = np.array([-1, 1])
    >>> y = X @ beta_true + np.random.randn(100)
    >>> mcmc = MCMC(NUTS(numpyro_model), num_warmup=10, num_samples=100)
    >>> mcmc.run(jr.key(0), X, y)

.. automodule:: flowjax.experimental.numpyro
   :members:
   :undoc-members:
   :show-inheritance:
