Inspect
~~~~~~~
- check torch.compile times on *HPC*
- examine impact of caching queue for *SYCL* then maybe use context

Algorithms
~~~~~~~~~~
- *adaptiveavgpool2d* where output dim isn't multiple of input dim
- *conv2d* groups *> 1* and more padding modes
- backpropagation is very slow need to implement a custom method check this on HPC
- add attention support similar to the current convolution

Publishing
~~~~~~~~~~
- host documentation *HTML* on github pages need repo public
- update the links to be the public links in *Doxyfile* and *intro.rst*
- add homepage information, docs, to pyproject.toml
- should have an example of an actual sped up example

Formats
~~~~~~~
- onnx
- tensorflow

Platforms
~~~~~~~~~
- metal
