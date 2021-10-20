Tutorial
========

To use the micromodel framework, first we need to configurate a list of micromodels.
Each micromodel is configurable using a dictionary.

The following example builds 3 micromodels, one for a SVM classifier, one for a logical micromodel, and one for a bert-query micromodel.

```
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
```

Each configuration requires at least two fields: `model_type` and `name`.
`model_type` specifies what kind of micromodel to build.
Possible values currently include `svm`, `logic`, and `bert_query`. To add new types of micromodels, see `src.factory.MICROMODEL_FACTORY`.

`model_path` is an optional argument to indicate where to save or load a micromodel from. If not specified, each micromodel will be saved to a default location (More on this later).

Dependingn the type of micromodel, each configuration will require different arguments in `setup_args`. These arguments will be passed in to the constructor of each micromodel using `**kwargs`. For details on micromodel-specific configurations, see the `__init__()` function of each micromodel.
Micromodels are defined in `src/micromodels/`.


Once you've configured your micromodels, you can initialize your `Orchestrator`. The Orchestrator basically manages training, loading, and inferring from micromodels.


```
basepath = os.environ.get("MM_HOME")
orchestrator = Orchestrator(basepath)
orchestrator.set_configs(configs)
```

The basepath is the default location that the orchestrator will save and load micromodels from, if `model_path` is not specified for any of the micromodels.

Once your orchestrator is set, you can now train and infer from your micromodels.

```
orchestrator.train_all()
orchestrator.infer("This is a test")
```
