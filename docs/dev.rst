Developer Tools
================

*run* Script
------------

An easy interface for building, testing, benchmarking and more.

    $ python run.py <command> <command> ...

Below are the possible commands and what they do.

Installing
~~~~~~~~~~

* *install*: Runs ``pip3 install .``
* *install_e*: Runs ``pip3 install --editable .``
* *install_ev*: Runs ``pip3 install --editable --verbose .``
* *install_ev*: Runs ``pip3 install .[dev,doc]``

Testing
~~~~~~~
Tests are ran with all algorithms available to ensure correctness of every
algorithm. When not performing unit tests, all applicable models in
:mod:`models` are used. To test a custom algorithm, add "custom" to the
algorithms to test in the file performing the tests.

* *test*:
  Runs all of the following tests for all models in :mod:`models`.

Unit
^^^^

* *utest:* and *test.unit*:
  Runs all unit tests with ``python -m test.unit``
* *utest.<>:* and *test.unit.<>*:
  Runs unit tests with for the operation where <> is the operation with
  ``python -m test.unit.<>``

Integration
^^^^^^^^^^^

#. Ensures results are equivalent after swapping original *conv2d* modules out of
   the original *DNN* for the framework's implementations.

   * *sctest:* and *test.swap_conv2d*:
     Runs ``python -m test.swap_conv2d``
   * *sctest.<>:* and *test.swap_conv2d.<>*:
     Runs ``python -m test.swap_conv2d <>``
     where <> is the model to use from :mod:`models`

#. Ensures results are equivalent after swapping all modules out of the original
   *DNN* for the framework's implementations.

   * *sbtest:* and *test.swap_backend*:
     Runs ``python -m test.swap_backend``
   * *sbtest.<>:* and *test.swap_backend.<>*:
     Runs ``python -m test.swap_backend <>``
     where <> is the model to use from :mod:`models`

Benchmarking
~~~~~~~~~~~~

By Layer
^^^^^^^^

* *lbench* and *bench.layer*:
  Shows latency in completing all operation originally
  and with the framework's implementation
* *lbench.<>* and *bench.layer.<>*:
  Shows latency in completing the specified operation originally
  and with the framework's implementation

By *DNN*
^^^^^^^^

* *scbench* and *bench.swap_conv2d*:
  Shows latency of all *DNNs* in :mod:`models`
  before and after swapping *conv2d* operations for
  the framework's implementation
* *scbench.<>* and *bench.swap_conv2d.<>*:
  Shows latency of the specific *DNN*, where <> is
  the *DNN* before and after swapping *conv2d*
  operations for the framework's implementation
* *sbbench* and *bench.swap_backend*:
  Shows latency of all *DNNs* in :mod:`models`
  before and after swapping all operations for
  the framework's implementation
* *sbbench.<>* and *bench.swap_backend.<>*:
  Shows latency of the specific *DNN*, where <> is
  the *DNN* before and after swapping all
  operations for the framework's implementation

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
