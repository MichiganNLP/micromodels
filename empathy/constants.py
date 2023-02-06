"""
Constants used in experiments.
"""

import os

mm_base_path = os.environ.get("MM_HOME")
_mm_model_dir = os.path.join(mm_base_path, "models")

EMP_TASKS = ["emotional_reactions", "interpretations", "explorations"]
EMP_CONFIGS = [
    {
        "name": "empathy_%s" % task,
        "model_type": "bert_query",
        "model_path": os.path.join(
            _mm_model_dir, "empathy_%s" % task
        ),
        "setup_args": {
            "infer_config": {
                "segment_config": {"window_size": 10, "step_size": 4},
            },
            "seed": [],
        },
    }
    for task in EMP_TASKS
]
