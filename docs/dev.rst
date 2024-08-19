Developer Tools
================

Testing and Benchmarking
------------------------

The *run.py* script provides an easy interface for building, testing,
benchmarking and more.

    $ python run.py <command> <command> ...

Below are the possible commands and what they do.

Installing
~~~~~~~~~~

* *install*: Runs ``pip3 install .``
* *install.e*: Runs ``pip3 install --editable .``
* *install.ev*: Runs ``pip3 install --editable --verbose .``
* *install.ev*: Runs ``pip3 install .[dev,doc]``

Testing
~~~~~~~
Tests are ran with all algorithms available to ensure correctness of every
algorithm. When not performing unit tests, all applicable models in
:mod:`models` are used. To test a custom algorithm, add "custom" to the
algorithms to test in the file performing the tests.

* *test*:
  Runs all of the following tests

Unit
^^^^

* *test.unit*:
  Runs all unit tests
* *test.unit.<operation>*:
  Runs unit tests for the operation

Integration
^^^^^^^^^^^

#. Ensures outputs are equivalent before and after after swapping *conv2d*
   modules for the framework's implementations

   * *test.swap_conv2d*:
     Runs test for all :mod:`models`
   * *test.swap_conv2d.<model>*:
     Runs test for model from :mod:`models`

#. Ensures outputs are equivalent before and after swapping all modules for the
   framework's implementations

   * *test.swap_backend*:
     Runs test for all :mod:`models`

   * *test.swap_backend.<model>*:
     Runs test for model from :mod:`models`

Benchmarking
~~~~~~~~~~~~

By Layer
^^^^^^^^

* *bench.layer*:
  Shows latency for all operations, both original and framework implementations

* *bench.layer.<layer>*:
  Shows latency for the specified operation, both original and framework implementations


By *DNN*
^^^^^^^^

#. Shows latencies before and after after swapping *conv2d*
   modules for the framework's implementations

   * *bench.swap_conv2d*:
     Shows latency for all :mod:`models`
   * *bench.swap_conv2d.<model>*:
     Shows latency for the model from :mod:`models`


#. Shows latencies before and after swapping all modules for the
   framework's implementations

   * *bench.swap_backend*:
     Shows latency for all :mod:`models`
   * *bench.swap_backend.<model>*:
     Shows latency for the model from :mod:`models`

Documentation
~~~~~~~~~~~~~
* *docs*: Generate the documentation in *HTML* format.
* *readme*: Generate the *README.rst*

Develop
~~~~~~~
* *clangd*: Generate the *.clangd* file with the correct include paths

Models
------

.. automodule:: models
   :members:

.. automodule:: models.models
   :members:
   :undoc-members:
