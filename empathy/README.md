Running our empathy prediction experiments requires a couple of environment variables:

```
MM_HOME=/[path...]/micromodels (path to micromodel root director0
EMP_HOME=$MM_HOME/empathy
PYTHONPATH=$MM_HOME
```

To run empathy prediction, run `python empathy_prediction.py`
To run empathetic rationale extraction, run `python rationale_extraction.py`
