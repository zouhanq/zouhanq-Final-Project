# regime_detection.py

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def detect_regimes(data, n_clusters=3):
    """Detect market regimes using K-means clustering."""
    features = ['svd_component_1', 'svd_component_2', 'svd_component_3']
    
    # Drop rows with any NaNs in the SVD components as a final check
    print("NaNs before KMeans:", data[features].isna().sum().sum())
    data = data.dropna(subset=features)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    data['regime'] = kmeans.fit_predict(data[features])
    
    return data, kmeans

def plot_regimes(data):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['svd_component_1'], data['svd_component_2'], c=data['regime'], cmap='viridis')
    plt.xlabel('SVD Component 1')
    plt.ylabel('SVD Component 2')
    plt.title('Market Regimes Detected by K-means')
    plt.colorbar(label='Regime')
    plt.show()

