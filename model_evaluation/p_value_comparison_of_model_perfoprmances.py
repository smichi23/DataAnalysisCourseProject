from typing import List
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from diagnostic_models.diagnostic_model import DiagnosticModel


class PValueComparisonOfModelPerformancesDistribution:
    def __init__(self, models: List[DiagnosticModel], data: np.ndarray, labels: np.ndarray):
        self.models = models
        self.accuracies = {}
        self.data = data
        self.labels = labels

    def evaluate_for_different_training_and_prediction_set(self, nb_of_sets):
        kf = KFold(n_splits=nb_of_sets, shuffle=True)
        for train_index, test_index in kf.split(self.data):
            training_set = (self.data[train_index], self.labels[train_index])
            testing_set = (self.data[test_index], self.labels[test_index])
            for model in self.models:
                self._fit_model_and_predict_on_test_set(model, training_set, testing_set)

    def _fit_model_and_predict_on_test_set(self, model, training_set, testing_set):
        model.reset()
        model.fit(training_set[0], training_set[1])
        predictions = model.predict(testing_set[0])
        accuracy = np.equal(predictions, testing_set[1]).sum() / len(predictions)
        if model.name not in self.accuracies.keys():
            self.accuracies[model.name] = []
        self.accuracies[model.name].append(accuracy)

    def plot_statistics_for_single_model(self, model_name, nb_of_bins=10):
        plt.hist(self.accuracies[model_name], bins=nb_of_bins)
        plt.title(f"Accuracy distribution for model {model_name}")
        plt.xlabel("Accuracy")
        plt.ylabel("Frequency")
        plt.show()

    def plot_statistics_between_two_models(self, model1_name, model2_name):
        pass

    def plot_statistics_between_all_models(self):
        pass
