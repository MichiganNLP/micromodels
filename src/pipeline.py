"""
Feature Extraction Pipeline
"""

import json
from src.factory import FEATURE_CLASSIFIER_FACTORY
from src.utils import preprocess


class FeatureExtractionPipeline:
    """
    Extract Features
    """

    def __init__(self):
        self.pipeline = []

    def build_pipeline_from_file(self, config_file):
        """
        Build pipeline from config file.
        """
        with open(config_file, "r") as file_p:
            config = json.load(file_p)
        return self.build_pipeline(config)

    def build_pipeline(self, config):
        """
        Build pipeline from config.
        [
            {
                'model_type': 'svm',
                'model_path': 'file_path',
                'feature_name': 'feature_name'

                Optional:
                'data': 'file_path',
            },
            ...
        ]
        if optional 'data' is specified, it will train
        the model from file specified in 'data' and
        save to 'model_path' instead of loading
        from 'model_path'
        """

        try:
            output_features = [entry["feature_name"] for entry in config]
        except KeyError as ex:
            raise ValueError(
                "Invalid config, 'feature_name' missing for an entry."
            ) from ex

        if len(list(set(output_features))) != len(config):
            raise ValueError("Invalid config, redundant 'feature_name' found.")

        for entry in config:
            model_type = entry.get("model_type")
            train_data = entry.get("data")
            model_path = entry.get("model_path")
            feature_name = entry.get("feature_name")

            if not model_type:
                raise ValueError("Invalid config, must specify model_type.")

            model = FEATURE_CLASSIFIER_FACTORY.get(entry["model_type"])(
                feature_name
            )

            if not model:
                raise ValueError(
                    "Invalid config, specified model_type %s is not valid"
                    % model_type
                )
            if not feature_name:
                raise ValueError("Invalid config, must specify 'feature_name'.")

            if train_data:
                print("train_data", train_data)
                model.train(train_data)
                model.save_model(model_path)
            else:
                model.load_model(model_path)
            self.pipeline.append(
                {
                    "model": model,
                    "model_path": model_path,
                    "feature_name": feature_name,
                }
            )
        return self.pipeline

    def run_pipeline(self, query):
        """
        Run pipeline on query
        """
        query = preprocess(query)
        features = {}
        for entry in self.pipeline:
            model = entry["model"]
            feature_name = entry["feature_name"]

            feature = model.infer(query, do_preprocess=False)
            features[feature_name] = feature
        return features


if __name__ == "__main__":
    CONFIG = [
        {
            "model_type": "svm",
            "data": "data/testing.json",
            "model_path": "models/gender",
            "feature_name": "gender",
        },
        {
            "model_type": "svm",
            "data": "data/numeric.json",
            "model_path": "models/numeric",
            "feature_name": "numeric",
        },
    ]
    PIPELINE = FeatureExtractionPipeline()
    PIPELINE.build_pipeline(CONFIG)

    FEATURES = PIPELINE.run_pipeline("I am a boy, I am 3 years old")
    print(FEATURES)
