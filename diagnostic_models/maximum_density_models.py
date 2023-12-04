import numpy as np
import scipy

from diagnostic_models.diagnostic_model import DiagnosticModel
from scipy.optimize import curve_fit, basinhopping


class PreDefinedThresholdMaximumDensity(DiagnosticModel):
    def __init__(self, name, threshold, threshold_domain):
        super().__init__(name, threshold_domain)
        self.threshold = threshold

    def fit(self, data, labels):
        pass

    def predict(self, data, final_threshold=None):
        max_intensities = data.max(axis=(-2, -1))
        classes = np.greater(max_intensities, self.threshold)

        return classes

    def reset(self):
        pass


class OptimalThresholdMaximumDiffDensity(DiagnosticModel):
    def __init__(self, name, threshold_domain):
        super().__init__(name, threshold_domain)
        self.threshold = None

    def fit(self, data, labels):
        max_intensities = data.max(axis=(-2, -1))

        best_threshold = basinhopping(func=self.optimize_prediction, minimizer_kwargs={"args": (data, labels)},
                                      x0=(max_intensities.mean()))
        self.threshold = best_threshold.x[0]

    def predict(self, data, final_threshold=None):
        if self.threshold is None and final_threshold is None:
            raise ValueError("The model has not been fitted yet.")
        max_intensities = data.max(axis=(-2, -1))
        if final_threshold is None:
            classes = np.greater(max_intensities, self.threshold)
        else:
            classes = np.greater(max_intensities, final_threshold)

        return classes

    def reset(self):
        self.threshold = None


class SigmoidScipyCurveFitMaximumDensity(DiagnosticModel):
    def __init__(self, name, initial_values, threshold_domain):
        super().__init__(name, threshold_domain)
        self.initial_values = initial_values
        self.parameters = initial_values
        self.threshold = None

    def fit(self, data, labels):
        values = curve_fit(self.sigmoid, data.max(axis=(-2, -1)), labels, self.parameters)
        self.parameters = values[0]
        best_threshold = basinhopping(func=self.optimize_prediction, minimizer_kwargs={"args": (data, labels)},
                                      x0=(self.threshold_domain[1] - self.threshold_domain[0]) / 2)
        self.threshold = best_threshold.x[0]

    @staticmethod
    def sigmoid(x, L, x0, k, b):
        return L / (1 + np.exp(-k * (x - x0))) + b

    def predict(self, data, final_threshold=None):
        if self.threshold is None and final_threshold is None:
            raise ValueError("The model has not been fitted yet.")
        sigmoid_values = self.sigmoid(data.max(axis=(-2, -1)), *self.parameters)
        if final_threshold is None:
            classes = np.greater(sigmoid_values, self.threshold)
        else:
            classes = np.greater(sigmoid_values, final_threshold)

        return classes

    def reset(self):
        self.threshold = None
        self.parameters = self.initial_values
