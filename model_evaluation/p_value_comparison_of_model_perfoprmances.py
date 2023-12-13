from typing import List

import scipy
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

    def compute_all_t_tests_and_produce_latex_table(self):

        latex_table = ""
        header = "Model"
        for model in self.models:
            header += f" & {model.name}"
            row = f"{model.name}"
            for other_model in self.models:
                if model.name == other_model.name:
                    row += " & - & - "
                else:
                    t_test, p_value = self.compute_t_test(model.name, other_model.name)
                    row += f"&{t_test:.2f} & {p_value:.4f}"
            latex_table += row + "\\\\ \n"
        print(header)
        print(latex_table)

    def compute_t_test(self, model1_name, model2_name):
        assert model1_name != model2_name
        model1_accuracies = np.array(self.accuracies[model1_name])
        model2_accuracies = np.array(self.accuracies[model2_name])
        t_test = scipy.stats.ttest_ind(model1_accuracies, model2_accuracies, equal_var=False)
        return t_test.statistic, t_test.pvalue

    def plot_statistics_between_all_models(self, bins):
        plt.figure(dpi=100, figsize=(10, 6))
        all_data = []
        all_labels = []
        for model in self.models:
            data = accuracies_in_np = np.array(self.accuracies[model.name])
            all_data.append(data)
            label = model.name + " " + r"$\bar{x}$" + f"$= {accuracies_in_np.mean():.2f}$" + f" and $s = {accuracies_in_np.std():.2f}$"
            all_labels.append(label)
        plt.hist(all_data, bins=np.linspace(0, 1, bins), density=True, range=(0, 1),
                 label=all_labels)
        plt.legend()
        plt.xlabel("Accuracy [-]")
        plt.ylabel("Frequency [-]")
        plt.show()
        plt.style.use('default')
