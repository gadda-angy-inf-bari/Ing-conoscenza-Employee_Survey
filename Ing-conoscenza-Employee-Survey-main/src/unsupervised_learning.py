import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from preprocessing_data import load_data

def main():
    # Caricamento del dataset
    df = load_data()

    # Codifica variabili categoriali
    for column in df.select_dtypes(include='object').columns:
        df[column] = LabelEncoder().fit_transform(df[column])

    # Gestione dei valori nulli
    df.fillna(df.median(), inplace=True)

    # Standardizzazione dei dati
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # Calcolare la funzione al Gomito per determinare il numero ottimale di cluster
    inertia = []
    k_values = range(2, 11)  # Testiamo valori di K da 2 a 10

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(df_scaled)
        inertia.append(kmeans.inertia_)

    # Visualizzare la curva del gomito (Elbow plot)
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertia, 'bo-', markersize=8)
    plt.xlabel('Numero di Cluster K')
    plt.ylabel('Inertia')
    plt.title('Metodo del Gomito per determinare il K ottimale')
    plt.show()

    # valore ottimale di k
    optimal_k = 3

    # modelli di clustering
    # K-means con il numero ottimale di cluster
    kmeans_model = KMeans(n_clusters=optimal_k, random_state=0)
    kmeans_labels = kmeans_model.fit_predict(df_scaled)

    # EM con Gaussian Mixture Model
    gmm_model = GaussianMixture(n_components=optimal_k, random_state=0)
    gmm_labels = gmm_model.fit_predict(df_scaled)

    # Stampare il centroide di ogni cluster (K-means)
    print("Centroidi di K-means (valori delle feature per ogni centroide):")
    for i, centroid in enumerate(kmeans_model.cluster_centers_):
        nearest_point_idx = np.argmin(np.linalg.norm(df_scaled - centroid, axis=1))
        nearest_point_values = df.iloc[nearest_point_idx].values
        print(f"Cluster {i+1} - Elemento più vicino al centroide: Indice {nearest_point_idx}, Valori: {nearest_point_values.tolist()}")
    
    
    # Grafico a torta della distribuzione dei cluster
    # Distribuzione dei cluster per K-means
    kmeans_counts = np.bincount(kmeans_labels)
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.pie(kmeans_counts, labels=[f"Cluster {i+1} ({count})" for i, count in enumerate(kmeans_counts)], autopct='%1.1f%%', startangle=90)
    plt.title("Distribuzione dei Cluster per K-means")

    # Distribuzione dei cluster per EM con Gaussian Mixture Model
    gmm_counts = np.bincount(gmm_labels)
    plt.subplot(1, 2, 2)
    plt.pie(gmm_counts, labels=[f"Cluster {i+1} ({count})" for i, count in enumerate(gmm_counts)], autopct='%1.1f%%', startangle=90)
    plt.title("Distribuzione dei Cluster per Gaussian Mixture Model")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()