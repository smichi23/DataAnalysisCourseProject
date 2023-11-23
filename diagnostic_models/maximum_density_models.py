import numpy as np
from diagnostic_models.diagnostic_model import DiagnosticModel

class PreDefinedThresholdMaximumDensity(DiagnosticModel):
    def __init__(self, name, model, threshold):
        super().__init__(name, model)
        self.threshold = threshold

    def fit(self, data, labels):
        pass

    def predict(self, data):
        max_intensities = data.max(axis=(-2, -1))
        classes = np.ma.masked_greater(max_intensities, self.threshold)

        return classes.mask.astype(np.int32)

