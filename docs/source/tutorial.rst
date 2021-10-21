Tutorial
========


Configuring Micromodels
-----------------------

To use the micromodel framework, first we need to configurate a list of micromodels.
Each micromodel is configurable using a dictionary.

The following example builds 3 micromodels, one for a SVM classifier, one for a logical micromodel, and one for a bert-query micromodel.::

    configs = [
        {
            "model_type": "svm",
            "name": "example_svm",
            "model_path": "micromodels/example_svm",
            "setup_args": {
                "training_data_path": "data/example_svm_data.json"
            },
        },
        {
            "model_type": "logic",
            "name": "example_logic",
            "model_path": "micromodels/example_logic",
            "setup_args": {"logic_func": _logic},
        },
        {
            "model_type": "bert_query",
            "name": "example_bert_query",
            "model_path": "micromodels/example_bert_query",
            "setup_args": {
                "threshold": 0.8,
                "seed": [
                    "This is a test",
                    "Arya is a hungry cat.",
                ],
                "infer_config": {
                    "k": 2,
                    "segment_config": {"window_size": 5, "step_size": 3},
                },
            },
        },
    ]

Each configuration requires at least two fields: **model_type** and **name**.
**model_type** specifies what kind of micromodel to build.
Possible values currently include **svm** **logic** and **bert_query** To add new types of micromodels, see *src.factory.MICROMODEL_FACTORY*.

**model_path** is an optional argument to indicate where to save or load a micromodel from. If not specified, each micromodel will be saved to a default location (More on this later).

Dependingn the type of micromodel, each configuration will require different arguments in **setup_args** These arguments will be passed in to the constructor of each micromodel using \*\*kwargs. For details on micromodel-specific configurations, see the *__init__()* function of each micromodel.
Micromodels are defined in *src/micromodels/*.


Orchestrator
------------

Once you've configured your micromodels, you can initialize your **Orchestrator**. The Orchestrator basically manages training, loading, and inferring from micromodels.::

    basepath = os.environ.get("MM_HOME")
    orchestrator = Orchestrator(basepath, configs)

The basepath is the default location in which the orchestrator will save and load micromodels from, only if *model_path* is not specified for any of the micromodels.

Once your orchestrator is set, you can now build (train) and run (infer) from your micromodels.::

    orchestrator.build_micromodels() # Build all the micromodels specified in the config.
    orchestrator.run_micromodels("This is a test") # Run the input query through all the micromodels

For details on how each micromodel is built, see the *build()* method for each micromodel (ex: src/micromodels/svm.py).


Building your custom micromodel
-------------------------------

To build a new type of micromodel, create a new class that inherits from *AbstractMicromodel* (src/micromodels/AbstractMicromodel.py), and implement the following methods accordingly:

* _build()
* _run()
* _batch_run()
* save_model()
* load_model()
* is_loaded()

Once your micromodel is implemented, add it to the micromodel factory (src.factory.MICROMODEL_FACTORY).
