# Micromodels

Micromodels -- A framework for accurate, explainable, data efficient, and reusable NLP models.

A micromodel is a representation for a specific linguistic behavior.
Analagous to how recent applications are built using *microservices*, in which each microservice has a specific responsibility, a micromodel is responsible for representing a specific linguistic behavior.
By orchestrating a suite of micromodels, our framework allows researchers and developers to build explainable and reusable systems across multiple NLP tasks.

For more details, refer to our Findings of EMNLP 2021 paper, [Micromodels for Efficient, Explainable, and Reusable Systems: A Case Study on Mental Health](https://arxiv.org/pdf/2109.13770.pdf).


### Setup

This repository requires Python version 3.7 or newer. 

Once cloned, install all required packages using pip. We highly recommend using a virtual environment.

`pip install -r requirements.txt`

Set an environment variable `MM_HOME` to point to the location of the cloned repository.
This will be the default location for where micromodels are stored. 

`export MM_HOME=/home/username/micromodels`


### Documentation, Tutorial

Documentation, as well as a short tutorial, can be found in our [readthedocs](https://nlpmicromodels.readthedocs.io/en/latest/index.html).


### Contributing

Pull requests are more than welcomed!
If there are any features you would like to add, or have any suggestions on design, email ajyl@umich.edu or create a pull request.
