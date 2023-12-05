from typing import List

import numpy as np

from diagnostic_models.diagnostic_model import DiagnosticModel


class TestPerformances:
    def __init__(self, models: List[DiagnosticModel], test_data: np.ndarray, labels: np.ndarray):
        self.models = models
        self.test_data = test_data
        self.labels = labels
        self.accuracies = {}

    def _get_model_by_name(self, model_name):
        for model in self.models:
            if model.name == model_name:
                return model
        raise ValueError(f"Model {model_name} not found")

    def evaluate_single_model(self, model_name):
        model = self._get_model_by_name(model_name)
        predictions = model.predict(self.test_data)
        accuracy = np.equal(predictions, self.labels).sum() / len(predictions)
        self.accuracies[model_name] = accuracy

        return accuracy



