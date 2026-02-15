import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from typing import Tuple, List

def run_pca(df_rgb: pd.DataFrame, n_components: int = 2) -> Tuple[PCA, pd.DataFrame]:
    """
    Performs PCA on RGB or CIELAB data.
    """
    # Extract numeric columns (R, G, B or L*, a*, b*)
    features = [col for col in df_rgb.columns if col in ['R', 'G', 'B', 'L*', 'a*', 'b*']]
    x = df_rgb[features].values
    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(x_scaled)
    
    df_pca = pd.DataFrame(
        data=principal_components,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    if 'label' in df_rgb.columns:
        df_pca['label'] = df_rgb['label'].values
        
    return pca, df_pca

def train_classifier(df: pd.DataFrame, label_col: str = 'label') -> Dict:
    """
    Boilerplate for training a supervised classifier (SVM) on color data.
    """
    features = [col for col in df.columns if col in ['R', 'G', 'B', 'L*', 'a*', 'b*']]
    X = df[features].values
    y = df[label_col].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return {'model': clf, 'report': report}
