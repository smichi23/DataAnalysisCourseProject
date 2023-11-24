from typing import List

import numpy as np

from diagnostic_models.diagnostic_model import DiagnosticModel


class PValueComparisonOfModelPerformancesDistribution:
    def __init__(self, models: List[DiagnosticModel], data: np.ndarray, labels: np.ndarray):
        self.models = models
        self.accuracies = {}

    def evaluate_for_different_training_set_but_same_prediction_set(self,
                                                                    ratio_training_testing,
                                                                    hidden_training_ratio,
                                                                    nb_of_sets):
        pass

    def evaluate_for_different_training_and_prediction_set(self,
                                                           ratio_training_testing,
                                                           nb_of_sets):
        pass

    def _fit_model_and_predict_on_test_set(self, model, training_set, testing_set):
        model.reset()
        model.fit(training_set[0], training_set[1])
        predictions = model.predict(testing_set[0])
        accuracy = np.sum(predictions == testing_set[1]) / len(predictions)
        if model.name not in self.accuracies.keys():
            self.accuracies[model.name] = []
        self.accuracies[model.name].append(accuracy)

    def plot_statistics_between_two_models(self, model1_name, model2_name):
        pass

    def plot_statistics_between_all_models(self):
        pass