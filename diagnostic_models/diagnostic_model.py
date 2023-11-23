
class DiagnosticModel:
    def __init__(self, name, model):
        self.name = name
        self.model = model

    def fit(self, data, labels):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

