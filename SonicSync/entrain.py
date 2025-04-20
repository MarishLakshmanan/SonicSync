# Define EnsembleModel
class EnsembleModel:
    def __init__(self):
        self.name = "Example"

    def predict(self, x):
        return [0]  # Dummy prediction

# Save
import joblib
model = EnsembleModel()
joblib.dump(model, 'test_model.pkl')

# Load
loaded_model = joblib.load('test_model.pkl')
print("Loaded model name:", loaded_model.name)