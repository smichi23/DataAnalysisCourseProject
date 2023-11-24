
class DiagnosticModel:
    def __init__(self, name, threshold_domain):
        self.name = name
        self.threshold_domain = threshold_domain

    def fit(self, data, labels):
        raise NotImplementedError

    def predict(self, data, final_threshold=None):
        raise NotImplementedError

