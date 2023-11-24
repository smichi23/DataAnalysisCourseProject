import numpy as np
from scipy.optimize import curve_fit, basinhopping

from diagnostic_models.diagnostic_model import DiagnosticModel


class PreDefinedThresholdMaximumDiffDensity(DiagnosticModel):
    def __init__(self, name, threshold, threshold_domain):
        super().__init__(name, threshold_domain)
        self.threshold = threshold

    def fit(self, data, labels):
        pass

    def predict(self, data, final_threshold=None):
        max_intensities = data.max(axis=(-2, -1))
        min_intensities = data.min(axis=(-2, -1))
        diff_intensities = max_intensities - min_intensities
        classes = np.ma.masked_greater(diff_intensities, self.threshold)

        return classes.mask.astype(np.int32)


class SigmoidScipyCurveFitMaximumDiffDensity(DiagnosticModel):
    def __init__(self, name, initial_values, threshold_domain):
        super().__init__(name, threshold_domain)
        self.parameters = initial_values
        self.threshold = None

    def fit(self, data, labels):
        values = curve_fit(self.sigmoid, (data.max(axis=(-2, -1)) - data.min(axis=(-2, -1))), labels, self.parameters)
        self.parameters = values[0]

        def optimize_prediction(threshold):
            if type(threshold) is np.ndarray:
                accuracy = []
                for t in threshold:
                    predictions = self.predict(data, t)
                    accuracy.append(np.sum(predictions == labels) / len(predictions))
                accuracy = np.array(accuracy)
            else:
                predictions = self.predict(data, threshold)

                accuracy = np.sum(predictions == labels) / len(predictions)
            return -accuracy

        self.threshold = 0
        best_threshold = basinhopping(func=optimize_prediction,
                                      x0=(self.threshold_domain[1] - self.threshold_domain[0])/2)
        self.threshold = best_threshold.x[0]

    @staticmethod
    def sigmoid(x, L, x0, k, b):
        return L / (1 + np.exp(-k * (x - x0))) + b

    def predict(self, data, final_threshold=None):
        if self.threshold is None:
            raise ValueError("The model has not been fitted yet.")
        sigmoid_values = self.sigmoid((data.max(axis=(-2, -1)) - data.min(axis=(-2, -1))), *self.parameters)
        if final_threshold is None:
            classes = np.greater(sigmoid_values, self.threshold)
        else:
            classes = np.greater(sigmoid_values, final_threshold)

        return classes
