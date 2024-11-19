import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# Konstanter
DATAFILSVÄG = "C:/temp2/user-faas/avanza/Dataset till case - (2024).csv"
AVSKILJARE = ';'
FEATURES = [
    'Totalt kapital på Avanza', 
    'Totalt kapital i Auto', 
    'Kapital i aktier', 
    'Kapital i fonder (inklusive Auto)', 
    'Inloggade dagar senaste månaden'
]
STANDARD_ANTAL_KLUSTER = 9
KLUSTERSPAN = range(2, 11)
SLUMPTAL = 42

def ladda_data(filväg, avskiljare):
    """Laddar dataset från angiven filväg."""
    return pd.read_csv(filväg, delimiter=avskiljare)

def förbered_features(data, features):
    """Extraherar och standardiserar utvalda features."""
    utvalda_features = data[features].fillna(0)
    scaler = StandardScaler()
    return scaler.fit_transform(utvalda_features)

def beräkna_kluster_mått(skalade_features, klusterspan):
    """Beräknar WCSS och silhouette score för olika antal kluster."""
    wcss = []
    silhouette_scores = []
    for antal_kluster in klusterspan:
        kmeans = KMeans(n_clusters=antal_kluster, random_state=SLUMPTAL)
        etiketter = kmeans.fit_predict(skalade_features)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(skalade_features, etiketter))
    return wcss, silhouette_scores

def plotta_elbow_metod(klusterspan, wcss):
    """Plottar Elbow-metoden för att bestämma optimalt antal kluster."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(klusterspan), y=wcss, mode='lines+markers', name='WCSS'))
    fig.update_layout(
        title="Optimalt antal kluster (Elbow-metoden)",
        xaxis_title="Antal kluster",
        yaxis_title="WCSS",
        template="plotly_white"
    )
    fig.show()

def plotta_silhouette_scores(klusterspan, silhouette_scores):
    """Plottar silhouette scores för olika antal kluster."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(klusterspan), 
        y=silhouette_scores, 
        mode='lines+markers', 
        name='Silhouette Score', 
        line=dict(color='orange')
    ))
    fig.update_layout(
        title="Silhouette Scores för olika antal kluster",
        xaxis_title="Antal kluster",
        yaxis_title="Silhouette Score",
        template="plotly_white"
    )
    fig.show()

def kör_kmeans(skalade_features, antal_kluster):
    """Utför KMeans-klustring på de skalade features."""
    kmeans = KMeans(n_clusters=antal_kluster, random_state=SLUMPTAL)
    return kmeans.fit_predict(skalade_features)

def reducera_med_pca(skalade_features, komponenter=2):
    """Reducerar features till angivet antal PCA-komponenter."""
    pca = PCA(n_components=komponenter)
    return pca.fit_transform(skalade_features)

def plotta_kluster_2d(pca_features, kluster_etiketter):
    """Visualiserar kluster i 2D PCA-rymd."""
    pca_df = pd.DataFrame(pca_features, columns=['PCA Komponent 1', 'PCA Komponent 2'])
    pca_df['Kluster'] = kluster_etiketter
    fig = px.scatter(
        pca_df,
        x='PCA Komponent 1',
        y='PCA Komponent 2',
        color=pca_df['Kluster'].astype(str),
        title='Kundsegmentering med KMeans (2D PCA)',
        labels={'color': 'Kluster'},
        template="plotly_white"
    )
    fig.update_traces(marker=dict(size=10, opacity=0.7))
    fig.show()

def tilldela_kluster_till_data(data, kluster_etiketter):
    """Tilldelar klusteretiketter till den ursprungliga datan."""
    data['Kluster'] = kluster_etiketter
    return data

def huvudfunktion():
    """Huvudflödet för klustringsanalys."""
    # Ladda och förbered data
    data = ladda_data(DATAFILSVÄG, AVSKILJARE)
    skalade_features = förbered_features(data, FEATURES)

    # Beräkna klustermått
    wcss, silhouette_scores = beräkna_kluster_mått(skalade_features, KLUSTERSPAN)
    plotta_elbow_metod(KLUSTERSPAN, wcss)
    plotta_silhouette_scores(KLUSTERSPAN, silhouette_scores)

    # Klustring med standard antal kluster
    kluster_etiketter = kör_kmeans(skalade_features, STANDARD_ANTAL_KLUSTER)
    pca_features = reducera_med_pca(skalade_features)

    # Visualisera kluster
    plotta_kluster_2d(pca_features, kluster_etiketter)

    # Tilldela kluster till ursprunglig data
    klustrad_data = tilldela_kluster_till_data(data, kluster_etiketter)
    return klustrad_data

if __name__ == "__main__":
    klustrad_data = huvudfunktion()
