FlowJAX
===========

FlowJAX: a package for continuous distributions, bijections and normalizing flows
using `equinox <https://github.com/patrick-kidger/equinox/>`_ and 
`jax <https://github.com/google/jax/>`_:

- Includes a wide range of distributions and bijections.
- Distributions and bijections are PyTrees, registered through 
  `equinox <https://github.com/patrick-kidger/equinox/>`_ modules, making them
  compatible with JAX transformations.
- Includes many state of the art normalizing flow models.
- First class support for conditional distributions, important for many
  applications such as amortized variational inference, and simulation-based inference.


Installation
------------------------
.. code-block:: bash

    pip install flowjax


.. toctree::
   :caption: Getting started
   :glob:
   :maxdepth: 1

   getting_started

.. toctree::
   :caption: Examples
   :glob:

   examples/examples

.. toctree::
   :caption: API
   :maxdepth: 1
   :glob:

   api/distributions
   api/bijections
   api/flows
   api/training
   api/losses
   api/experimental

.. toctree::
   :maxdepth: 1
   :caption: Miscellaneous
   
   faq
