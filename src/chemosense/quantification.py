import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error
from typing import Tuple, Dict

def train_calibration_model(x: np.ndarray, y: np.ndarray, degree: int = 1) -> Dict:
    """
    Trains a calibration model (Linear or Polynomial).
    x: Analytical signal (e.g., Delta E)
    y: Known concentrations
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
        
    if degree == 1:
        model = LinearRegression()
    else:
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        
    model.fit(x, y)
    y_pred = model.predict(x)
    
    metrics = {
        'R2': r2_score(y, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y, y_pred))
    }
    
    return {'model': model, 'metrics': metrics, 'degree': degree}

def predict_concentration(model_dict: Dict, x_unknown: np.ndarray) -> np.ndarray:
    """
    Predicts concentration for unknown samples using the trained model.
    """
    if x_unknown.ndim == 1:
        x_unknown = x_unknown.reshape(-1, 1)
        
    return model_dict['model'].predict(x_unknown)
