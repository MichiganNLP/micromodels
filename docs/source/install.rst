Installation
============

Our framework requires Python version 3.7 or newer.
We highly recommend using a virtual environment.

To get started with our framework, first clone our repository.

.. code-block:: console

 (venv) $ git clone git@github.com:MichiganNLP/micromodels.git

Install all required packages.

.. code-block:: console

 (venv) $ cd micromodels
 (venv) $ pip install -r requirements.txt

Set the environment variable `MM_HOME` to point to the cloned repsotiry.

.. code-block:: console

 (venv) $ export MM_HOME=/home/username/micromodels

To ensure that everything is installed correctly, run the included tests:

.. code-block:: console

 (venv) $ pytest

