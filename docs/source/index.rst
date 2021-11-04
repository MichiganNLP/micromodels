Micromodel Documentation
========================

**Micromodels** -- A framework for accurate, explainable, data efficient, and reusable NLP models.

A micromodel is a representation for a specific linguistic behavior.
Analagous to how recent applications are built using *microservices*, in which each microservice has a specific responsibility, a micromodel is responsible for representing a specific linguistic behavior.
By orchestrating a suite of micromodels, our framework allows researchers and developers to build explainable and reusable systems across multiple NLP tasks.

For more details, refer to our Findings of EMNLP 2021 paper, `Micromodels for Efficient, Explainable, and Reusable Systems: A Case Study on Mental Health <https://aclanthology.org/2021.findings-emnlp.360.pdf>`_.

Contributing
------------

Our repository can be found `here <https://github.com/MichiganNLP/micromodels>`_.
Pull requests are more than welcomed!
If there are any features you would like to add, or have any suggestions on design, email ajyl@umich.edu or create a pull request.


Contents
---------

.. toctree::
    :maxdepth: 2
    :caption: Getting Started

    install
    tutorial

.. toctree::
    :maxdepth: 2
    :caption: Module API

    orchestrator
    micromodels
    aggregators
    task_classifier



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
