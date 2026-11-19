from networksecurity.constants.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME

import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def transform(self, x):
        try:
            return self.preprocessor.transform(x)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def predict(self, x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def predict_proba(self, x):
        try:
            x_transform = self.preprocessor.transform(x)
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(x_transform)
            raise AttributeError("Underlying model does not support predict_proba")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def predict_with_proba(self, x):
        """
        Runs preprocessing transform exactly once and returns both predictions and probabilities.
        """
        try:
            x_transform = self.preprocessor.transform(x)
            y_pred = self.model.predict(x_transform)
            y_proba = (
                self.model.predict_proba(x_transform)
                if hasattr(self.model, "predict_proba")
                else None
            )
            return y_pred, y_proba
        except Exception as e:
            raise NetworkSecurityException(e, sys)