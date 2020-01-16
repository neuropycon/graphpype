:orphan:

.. _how_to_wrap:



Wrapping nodes
**************


Neuropycon is based on Nipype, and is open to any contribution with the wrapping mechanism allowing anyone to contribute.

We provide a tutorial on how to wrap a python package of your choice, based on the explanation provided on the `nipype website <https://nipype.readthedocs.io/en/latest/devel/python_interface_devel.html>`_.

We provide here an example with the `BCT toolbox <https://sites.google.com/site/bctnet/>`_, a package largely used in the neuroscience community for computating graph-theoretical based metrics. BCT was originally a Matlab set of functions, but a python version called `bctpy <https://pypi.org/project/bctpy/>`_ also exists.

Here we provide an example of a wrap on one single fuction, incorporated in a `new pipeline <https://neuropycon.github.io/graphpype/auto_examples/plot_inv_ts_to_bct_graph.html#inv-ts-to-bct-graph>`_
in order check how long it took to wrap one function. By itself, the wrapping of a single function (Kcore computation of the BCT, one of the measures available in BCT and not in Radatools) took less than half an hour. A rough estimate of the wrap of the K-core function of bctpy package leads to a count of ~50 lines of code (see interfaces/bct/bct.py on github graphype project).

The source code of the corresponding wrapped node:

.. include:: ../graphpype/interfaces/bct/bct.py
   :literal:

The incorporation in a functional pipeline (after the matrix computation and thresholding) took another hour, and corresponds to another ~10 lines of code:

.. include:: ../graphpype/pipelines/conmat_to_graph.py
   :literal:
   :start-after: # create_pipeline_bct_graph
   :end-before: # create_pipeline_graph_module_properties
