Tutorial
========


Introduction
------------

Welcome! In this tutorial, we will go over how to use micromodels for the `IMDB Sentiment Analysis task <https://ai.stanford.edu/~amaas/data/sentiment/>`_.

Configuring Micromodels
-----------------------

To use the micromodel framework, we first need to configurate a list of micromodels.
Each micromodel is configurable using a dictionary object.

The following example configures 3 micromodels, one for a SVM classifier, one for a logical micromodel, and one for a bert-query micromodel. ::

    def load_configs():
        """ Configure our micromodels """

        mm_base_path = os.environ.get("MM_HOME")
        models_basepath = os.path.join(mm_base_path, "models")

        imdb_data_dir = os.path.join(mm_base_path, "example/data")
        imdb_data_path = os.path.join(imdb_data_dir, "imdb_dataset.json")
        svm_data_path = os.path.join(imdb_data_dir, "svm_train_data.json")

        positive_keywords = ["good", "great", "wonderful", "beautiful"]
        def _positive_keyword_lookup(utterance):
            """ inner logic function for micromodel """
            return any(keyword in utterance for keyword in positive_keywords)

        configs = [
            {
                "model_type": "logic",
                "name": "positive_logic",
                "model_path": os.path.join(models_basepath, "positive_keyword_logic"),
                "setup_args": {"logic_func": _positive_keyword_lookup},
            },
            {
                "model_type": "svm",
                "name": "film_sentiment_svm",
                "model_path": os.path.join(models_basepath, "film_sentiment_svm"),
                "setup_args": {"training_data_path": svm_data_path},
            },
            {
                "model_type": "bert_query",
                "name": "positive_sentiment_bert_query",
                "model_path": os.path.join(
                    models_basepath, "positive_sentiment_bert_query"
                ),
                "setup_args": {
                    "threshold": 0.8,
                    "seed": [
                        "This is a great movie",
                        "I really liked this movie",
                        "That was a fantastic movie.",
                        "Great plot",
                        "Very enjoyable movie",
                        "Great cast and great actors"
                    ],
                    "infer_config": {
                        "k": 2,
                        "segment_config": {"window_size": 5, "step_size": 3},
                    },
                },
            },
        ]
        return configs

Each configuration requires at least two fields: **model_type** and **name**.
**model_type** specifies what kind of micromodel to build.
Possible values currently include **svm**, **logic**, **fasttext**, and **bert_query** To add new types of micromodels, see :ref:`custom-micromodels`.

**model_path** is an optional argument to indicate where to save or load a micromodel from. If not specified, each micromodel will be saved to a default location (More on this later).

Depending on the type of micromodel, each configuration will require different arguments in **setup_args**.
These arguments will be passed in to the constructor of each micromodel using \*\*kwargs.
For details on micromodel-specific configurations, refer to the documentation for each `micromodel module <https://nlpmicromodels.readthedocs.io/en/latest/micromodels.html>`_, or refer to the constructor *__init__()* method for each micromodel.
Micromodels are defined in `src/micromodels/ <https://github.com/MichiganNLP/micromodels/tree/master/src/micromodels>`_.


Task-Specific Classification
----------------------------

Once our micromodels are configured, we are ready to train on a task.
In this example, we will be running 3 micromodels to featurize each movie review.

First, let's initialize our task-specific classifier. ::

    mm_base_path = os.environ.get("MM_HOME")
    models_basepath = os.path.join(mm_base_path, "models")
    configs = load_configs()
    clf = TaskClassifier(models_basepath, configs)

`models_basepath` specifies the default location for saving and loading the specified micromodels (unless *model_path* is specified in the configuration of a micromodel).

Let's also load our imdb data. The dataset for imdb is `included in our repository <https://github.com/MichiganNLP/micromodels/blob/master/example/data/imdb_dataset.json>`_. ::

    imdb_data_dir = os.path.join(mm_base_path, "example/data")
    imdb_data_path = os.path.join(imdb_data_dir, "imdb_dataset.json")
    train_data, test_data = load_data(imdb_data_path, train_ratio=0.7)

Please refer to `example/example_experiment.py <https://github.com/MichiganNLP/micromodels/blob/master/example/example_experiment.py>`_ for details of `load_data()`.

Once we have our text data, we are ready to featurize our data using our micromodels. ::

    featurized = clf.featurize_data(train_data)
    feature_vector = featurized["feature_vector"]
    labels = featurized["labels"]

For details on what's going on under the hood and how micromodels are used to featurize our data, see :ref:`under-the-hood`.
Once our data is featurized, we can train our classifier. ::

    clf.fit(feature_vector, labels)

With a trained classifier, we can now run inference or run tests! ::

    print(clf.infer(["I liked this movie.", "It was a good movie"]))
    print(clf.infer(["I hated this movie.", "It was a bad movie"])
    print(clf.test(test_data))


While the `TaskClassifier` module has most of the functionality needed to be trained on a task, you can build your own classifier to add any customization or additional features by inheriting from the `TaskClassifier` module.
This will allow you to featurize the data any way you'd like, or to swap out the inner classifier (`Explainable Boosting Machines <https://interpret.ml/docs/ebm.html>`_) to a different model.

For more details about the task-specific classifier, refer to its documentation page: `Task-Specific Classifier <https://nlpmicromodels.readthedocs.io/en/latest/task_classifier.html>`_.


.. _under-the-hood:

Orchestrator
------------

Under the hood of each task-specific classifier, there is an **Orchestrator** that manages training, saving, loading, and running our micromodels.
The task-specific classifier interfaces with the micromodels using the Orchestrator.

You can also interface with your micromodels directly without having to train a task-specific classifier. ::

    basepath = os.environ.get("MM_HOME")
    models_basepath = os.path.join(mm_base_path, "models")
    configs = load_configs()
    orchestrator = Orchestrator(basepath, configs)

Once your orchestrator is set, you can build (train) and run (infer) your micromodels. ::

    orchestrator.build_micromodels() # Build all the micromodels specified in the config.
    orchestrator.run_micromodels("This is a test") # Run the input query through all the micromodels


Each micromodel has a **build()** method and a **run()** method.
Calling `orchestrator.build_micromodels()` or `orchestrator.run_micromodels(...)` simply calls these **build()** and **run()** methods for each micromodel.
Within the task-specific classifier, during initialization as well as featurization, these functionalities from the orchestrator are used frequently.

For more details about the orchestrator, refer to its documentation page: `Orchestrator <https://nlpmicromodels.readthedocs.io/en/latest/orchestrator.html>`_.



.. _custom-micromodels:

Building your custom micromodel
-------------------------------

To build a new type of micromodel, create a new class that inherits from `AbstractMicromodel <https://github.com/MichiganNLP/micromodels/blob/master/src/micromodels/AbstractMicromodel.py>`_ and implement the following methods accordingly:

* _build()
* _run()
* _batch_run()
* save_model()
* load_model()
* is_loaded()

Once your micromodel is implemented, add it to the `micromodel factory <https://github.com/MichiganNLP/micromodels/blob/master/src/factory.py>`_.
