# Micromodels

Micromodels -- A framework for accurate, explainable, data efficient, and reusable NLP models.

[![pytests](https://github.com/MichiganNLP/micromodels/actions/workflows/pytests/badge.svg)](https://github.com/MichiganNLP/micromodels/actions/workflows/pytests.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2109.13770-b31b1b.svg)](https://arxiv.org/abs/2109.13770)

A micromodel is a representation for a specific linguistic behavior.
Analagous to how recent applications are built using *microservices*, in which each microservice has a specific responsibility, a micromodel is responsible for representing a specific linguistic behavior.
By orchestrating a suite of micromodels, our framework allows researchers and developers to build explainable and reusable systems across multiple NLP tasks.

For more details, refer to our Findings of EMNLP 2021 paper, [Micromodels for Efficient, Explainable, and Reusable Systems: A Case Study on Mental Health](https://aclanthology.org/2021.findings-emnlp.360.pdf).


### Setup

This repository requires Python version 3.7 or newer. 

Once cloned, install all required packages using pip. We highly recommend using a virtual environment.

`pip install -r requirements.txt`

Set an environment variable `MM_HOME` to point to the location of the cloned repository.
This will be the default location for where micromodels are stored. 

`export MM_HOME=/home/username/micromodels`


### Documentation, Tutorial

Documentation can be found in our [readthedocs](https://nlpmicromodels.readthedocs.io/en/latest/index.html), which includes a [tutorial](https://nlpmicromodels.readthedocs.io/en/latest/tutorial.html).


### Contributing

Pull requests are more than welcomed!
If there are any features you would like to add, or have any suggestions on design, email ajyl@umich.edu or create a pull request.


### Citation

If you find our work helpful for your research, please cite our work:

```
@inproceedings{lee-etal-2021-micromodels-efficient,
    title = "Micromodels for Efficient, Explainable, and Reusable Systems: A Case Study on Mental Health",
    author = "Lee, Andrew  and
      Kummerfeld, Jonathan K.  and
      An, Larry  and
      Mihalcea, Rada",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.360",
    pages = "4257--4272",
}
```
