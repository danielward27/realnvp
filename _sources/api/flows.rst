Flows
==========================
Normalizing flows define flexible distributions by transforming a simple base
distribution through a flexible bijection (more precisely, a diffeomorphism). For a
detailed introduction to normalizing flows, we recommend the following review paper:
`Papamakarios et al., 2021 <https://arxiv.org/abs/1912.02762>`_.

In FlowJAX, all the normalizing flows are convient constructors of
:py:class:`~flowjax.distributions.Transformed` distributions. Generally, the overall
transform is built using multiple layers, by composing many individual transforms.
In FlowJAX, there are two ways to compose bijections:

- :py:class:`~flowjax.bijections.Chain` Allows chaining arbitary and heterogeneous
  bijections, but compiles each layer seperately.
- :py:class:`~flowjax.bijections.Scan` Requires the layers to share the same
  structure, but avoids compiling each layer seperately.

All FlowJAX flows use :py:class:`~flowjax.bijections.Scan` to reduce compilation
times.

.. note::
   Bijections in normalizing flows typically have asymmetric computational efficiency. 
   Generally:
   
   - Bijections are implemented to favor efficiency in the forward transformation.
   - The forward transformation is used for sampling (or ``sample_and_log_prob``), 
     while the inverse is used for density evaluation.
   - By default, flows invert the bijection with :py:class:`~flowjax.bijections.Invert` 
     before transforming the base distribution with
     :py:class:`~flowjax.distributions.Transformed`. This prioritizes a faster 
     ``log_prob`` method at the cost of a slower ``sample`` and 
     ``sample_and_log_prob`` method. 
   - If faster ``sample`` and ``sample_and_log_prob`` methods are needed (e.g., for 
     certain variational objectives), setting ``invert=False`` is recommended.



.. automodule:: flowjax.flows
   :members:
   :undoc-members:
   :show-inheritance:
