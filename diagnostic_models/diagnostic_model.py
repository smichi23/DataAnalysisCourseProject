import numpy as np


class DiagnosticModel:
    def __init__(self, name, threshold_domain):
        self.name = name
        self.threshold_domain = threshold_domain

    def fit(self, data, labels):
        raise NotImplementedError

    def predict(self, data, final_threshold=None):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def optimize_prediction(self, threshold, data, labels):
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

