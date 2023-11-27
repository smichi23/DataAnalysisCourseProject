from typing import List
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from diagnostic_models.diagnostic_model import DiagnosticModel


class AUCComparisonOfModelPerformances:
    def __init__(self, models: List[DiagnosticModel], data: np.ndarray, labels: np.ndarray):
        self.models = models
        self.auc_points = {}
        self.data = data
        self.labels = labels

    def evaluate_for_different_training_and_prediction_set(self, nb_of_sets, nb_of_threshod_points=100):
        kf = KFold(n_splits=nb_of_sets, shuffle=True)
        for train_index, test_index in kf.split(self.data):
            training_set = (self.data[train_index], self.labels[train_index])
            testing_set = (self.data[test_index], self.labels[test_index])
            for model in self.models:
                self._fit_model_and_produce_roc(model, training_set, testing_set, nb_of_threshod_points)


    def _fit_model_and_produce_roc(self, model, training_set, testing_set, nb_of_threshod_points):
        model.reset()
        model.fit(training_set[0], training_set[1])
        threshold_domain = model.threshold_domain
        if model.name not in self.auc_points.keys():
            self.auc_points[model.name] = {"fpr": [], "tpr": []}
        self.auc_points[model.name]["fpr"].append([])
        self.auc_points[model.name]["tpr"].append([])
        if threshold_domain is None:
            predictions = model.predict(testing_set[0])
            fpr = np.equal(predictions - testing_set[1], 1).sum() / -(testing_set[1]-1).sum()
            tpr = np.equal(predictions + testing_set[1], 2).sum() / testing_set[1].sum()
            self.auc_points[model.name]["fpr"][-1].append(fpr)
            self.auc_points[model.name]["tpr"][-1].append(tpr)
        else:
            for threshold in np.linspace(threshold_domain[0], threshold_domain[1], nb_of_threshod_points):
                predictions = model.predict(testing_set[0], threshold)
                fpr = np.equal(predictions - testing_set[1], 1).sum() / -(testing_set[1] - 1).sum()
                tpr = np.equal(predictions + testing_set[1], 2).sum() / testing_set[1].sum()
                self.auc_points[model.name]["fpr"][-1].append(fpr)
                self.auc_points[model.name]["tpr"][-1].append(tpr)

    def plot_auc_for_single_model_on_multiple_splits(self, model_name):
        for i in range(len(self.auc_points[model_name]["fpr"])):
            roc = np.asarray([self.auc_points[model_name]["fpr"][i], self.auc_points[model_name]["tpr"][i]])
            roc.sort(axis=1)
            if len(roc[0]) == 1:
                plt.scatter(roc[0], roc[1])
            else:
                plt.plot(roc[0], roc[1])
        plt.title(f"ROC curve for model {model_name}")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.show()

    def get_intrapolated_auc_with_confidence_intervals(self, model_name, x_values, nb_sigmas=1):
        intrapoled_rocs = []
        for i in range(len(self.auc_points[model_name]["fpr"])):
            combined = np.asarray([self.auc_points[model_name]["fpr"][i], self.auc_points[model_name]["tpr"][i]])
            desceding_order = combined[:, np.flip(combined[1, :].argsort())]
            unique_x = desceding_order[:, np.unique(desceding_order[0, :], return_index=True)[1]]
            unique_acending = unique_x[:, unique_x[0, :].argsort()]
            if unique_acending[0,0] != 0:
                unique_acending = np.concatenate((np.asarray([0, 0]).reshape(2, 1), unique_acending), axis=1)
            if unique_acending[0,-1] != 1:
                unique_acending = np.concatenate((unique_acending, np.asarray([1, 1]).reshape(2, 1)), axis=1)
            intrapolated_roc = self.intrapolate_and_evaluate_to_new_x(unique_acending[0, :],
                                                                      unique_acending[1, :],
                                                                      x_values)
            intrapoled_rocs.append(intrapolated_roc)

        intrapoled_rocs = np.asarray(intrapoled_rocs)
        mean_roc = intrapoled_rocs.mean(axis=0)
        std_roc = intrapoled_rocs.std(axis=0) * nb_sigmas

        return mean_roc - std_roc, mean_roc, mean_roc + std_roc

    @staticmethod
    def intrapolate_and_evaluate_to_new_x(old_x, y, new_x):
        def interpolation(x):
            index = np.searchsorted(old_x, x, side="right", sorter=None)
            intrapo = y[index - 1] + (x - old_x[index - 1]) * (
                    (y[index] - y[index - 1]) / (old_x[index] - old_x[index - 1]))

            return intrapo

        vf = np.vectorize(interpolation)

        return vf(new_x)

    def plot_auc_on_all_models_on_multiple_splits(self, nb_sigmas=1):
        for model in self.models:
            if model.threshold_domain is None:
                points_x = np.asarray(self.auc_points[model.name]["fpr"]).mean(axis=0)
                err_x = np.asarray(self.auc_points[model.name]["fpr"]).std(axis=0)
                points_y = np.asarray(self.auc_points[model.name]["tpr"]).mean(axis=0)
                err_y = np.asarray(self.auc_points[model.name]["tpr"]).std(axis=0)
                plt.scatter(points_x, points_y, label=model.name)
                plt.errorbar(points_x, points_y, yerr=err_y, xerr=err_x, fmt="o")
            else:
                x_values = np.linspace(0, 0.999, 100)
                mean_minus_std_roc, mean_roc, mean_plus_std_roc = self.get_intrapolated_auc_with_confidence_intervals(model.name, x_values, nb_sigmas)
                plt.plot(x_values, mean_roc, label=model.name)
                plt.fill_between(x_values, mean_minus_std_roc, mean_plus_std_roc, alpha=0.2, label=f"{nb_sigmas} sigma for {model.name}")
        plt.title(f"ROC curves")
        ax = plt.gca()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.legend()
        plt.show()

