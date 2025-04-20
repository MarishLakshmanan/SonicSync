from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.models import load_model, Sequential

#creating a custom Python class to combine the CNN feature extractor and Random Forest classifier into a cohesive ensemble pipeline.
import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class EnsembleModel(BaseEstimator, ClassifierMixin):
    def __init__(self, cnn_model, rf_model):
        self.cnn_model = cnn_model  # The trained CNN model
        self.rf_model = rf_model    # The trained Random Forest model

    def extract_features(self, X):
        """
        Extract features using the CNN model.
        """
        cnn_feature_extractor = Sequential(self.cnn_model.layers[:-1])  # Exclude the last Dense layer
        cnn_features = cnn_feature_extractor.predict(X)
        return cnn_features
    

    def fit(self, X, y):
        """
        Fit the Random Forest model using CNN-extracted features.
        """
        # Extract features
        cnn_features = self.extract_features(X)
        # Fit the Random Forest model
        self.rf_model.fit(cnn_features, y)
        return self

    def predict(self, X):
        """
        Predict using the ensemble model.
        """
        # Extract features
        cnn_features = self.extract_features(X)
        # Predict using the Random Forest model
        return self.rf_model.predict(cnn_features)

    def predict_proba(self, X):
        """
        Predict probabilities for each class.
        """
        # Extract features
        cnn_features = self.extract_features(X)
        # Predict probabilities using the Random Forest model
        return self.rf_model.predict_proba(cnn_features)

