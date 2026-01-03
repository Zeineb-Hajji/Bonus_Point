import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

# Page configuration
st.set_page_config(page_title="ClusterLab - Multi-Perspective", layout="wide")

st.title(" Multi-Perspective ClusterLab")
st.markdown("### K-Means & Agglomerative Clustering on Multi-Perspective Data")

# Data loading
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('merged_clusters.csv')
        df['Income'] = df['Income'].fillna(df['Income'].median())
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df_raw = load_data()

if df_raw is None:
    st.stop()

# Feature definition per perspective
value_features = ['Customer Lifetime Value', 'Income', 'LoyaltyScore', 'clv_per_month_log']
behaviour_features = ['AvgFlightsPerMonth', 'TotalPointsRedeemed', 'AvgRedeemRate', 'AvgDistancePerFlight', 'tenure_months']
demo_features = ['EducationLevelCode', 'ProvinceActivity', 'LoyaltyScore']

# Sidebar for parameters
st.sidebar.header("⚙️ Clustering Parameters")

st.sidebar.subheader("1. Value Perspective")
n_clusters_value = st.sidebar.slider("Clusters (Value)", 2, 10, 4)

st.sidebar.subheader("2. Behaviour Perspective")
n_clusters_behaviour = st.sidebar.slider("Clusters (Behaviour)", 2, 10, 3)

st.sidebar.subheader("3. Demographic Perspective")
n_clusters_demo = st.sidebar.slider("Clusters (Demographics)", 2, 10, 6)

st.sidebar.subheader("4. Merged Perspective")
n_final_clusters = st.sidebar.slider("Final Clusters", 2, 12, 6)

# Helper function for PCA visualization
def plot_pca_step(X_scaled, labels, title, hover_data=None):
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(components, columns=['PCA1', 'PCA2'])
    df_pca['Cluster'] = labels.astype(str)
    if hover_data is not None:
        df_pca['ID'] = hover_data
    
    fig = px.scatter(df_pca, x='PCA1', y='PCA2', color='Cluster', 
                     hover_data=['ID'] if hover_data is not None else None,
                     title=title,
                     color_discrete_sequence=px.colors.qualitative.Safe)
    fig.update_layout(template="plotly_white", height=400)
    return fig

# Button to run clustering
if st.sidebar.button("Run Clustering"):
    df = df_raw.copy()
    
    # 1. Value Clustering
    scaler_value = StandardScaler()
    X_value_scaled = scaler_value.fit_transform(df[value_features])
    kmeans_value = KMeans(n_clusters=n_clusters_value, init="k-means++", n_init=20, random_state=42)
    df["cluster_value"] = kmeans_value.fit_predict(X_value_scaled)
    
    # 2. Behaviour Clustering
    scaler_behaviour = StandardScaler()
    X_behaviour_scaled = scaler_behaviour.fit_transform(df[behaviour_features])
    kmeans_behaviour = KMeans(n_clusters=n_clusters_behaviour, init="k-means++", n_init=20, random_state=42)
    df["cluster_behaviour"] = kmeans_behaviour.fit_predict(X_behaviour_scaled)
    
    # 3. Demographic Clustering
    scaler_demo = StandardScaler()
    X_demo_scaled = scaler_demo.fit_transform(df[demo_features])
    hc_demo = AgglomerativeClustering(n_clusters=n_clusters_demo, linkage="ward")
    df["cluster_demo"] = hc_demo.fit_predict(X_demo_scaled)
    
    # 4. Merging Perspectives
    all_features = list(set(value_features + behaviour_features + demo_features))
    scaler_merge = StandardScaler()
    X_merge_scaled = scaler_merge.fit_transform(df[all_features])
    
    df_merge_space = pd.DataFrame(X_merge_scaled, columns=all_features, index=df.index)
    df_merge_space["cluster_value"] = df["cluster_value"]
    df_merge_space["cluster_behaviour"] = df["cluster_behaviour"]
    df_merge_space["cluster_demo"] = df["cluster_demo"]
    
    df_centroids = df_merge_space.groupby(['cluster_value', 'cluster_behaviour', 'cluster_demo'])[all_features].mean()
    hclust_final = AgglomerativeClustering(linkage='ward', n_clusters=n_final_clusters)
    final_labels = hclust_final.fit_predict(df_centroids)
    
    centroid_to_cluster = {idx: label for idx, label in zip(df_centroids.index, final_labels)}
    df['final_cluster'] = df.apply(lambda x: centroid_to_cluster[(x['cluster_value'], x['cluster_behaviour'], x['cluster_demo'])], axis=1)
    
    # Metrics
    cluster_centers = df.groupby('final_cluster')[all_features].transform('mean')
    r2 = r2_score(df[all_features], cluster_centers)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Final R2 score", f"{r2:.3f}")
    col_m2.metric("Total Data Points", f"{len(df)}")
    col_m3.metric("Final Clusters", f"{n_final_clusters}")

    st.divider()
    
    # Main Tabs
    tabs = st.tabs(["🎯 Final Results", "📐 Perspective Breakdown", "🔍 Cluster Profiling"])
    
    with tabs[0]:
        st.subheader("Final Merged Clustering")
        fig_final = plot_pca_step(X_merge_scaled, df['final_cluster'], "Final 2D PCA Projection", df['Loyalty#'])
        st.plotly_chart(fig_final, use_container_width=True)
        
        col_b1, col_b2 = st.columns([1, 1])
        with col_b1:
            st.subheader("Cluster Distribution")
            cluster_counts = df['final_cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Count']
            cluster_counts['Cluster'] = cluster_counts['Cluster'].astype(str)
            fig_bar = px.bar(cluster_counts, x='Cluster', y='Count', color='Cluster',
                             color_discrete_sequence=px.colors.qualitative.Set1)
            st.plotly_chart(fig_bar, use_container_width=True)
        with col_b2:
            st.subheader("Download Data")
            st.write("Export the final clustered dataset to CSV.")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results (CSV)", csv, "clustered_data.csv", "text/csv")

    with tabs[1]:
        st.subheader("Individual Perspective Visualizations")
        st.write("See how each perspective clusters the data before they are merged.")
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            fig_v = plot_pca_step(X_value_scaled, df['cluster_value'], "Value Perspective Clusters")
            st.plotly_chart(fig_v, use_container_width=True)
        with col_p2:
            fig_b = plot_pca_step(X_behaviour_scaled, df['cluster_behaviour'], "Behaviour Perspective Clusters")
            st.plotly_chart(fig_b, use_container_width=True)
            
        col_p3, col_p4 = st.columns(2)
        with col_p3:
            fig_d = plot_pca_step(X_demo_scaled, df['cluster_demo'], "Demographic Perspective Clusters")
            st.plotly_chart(fig_d, use_container_width=True)
        with col_p4:
            st.info("""
            **How it works:**
            1. Each perspective is clustered independently.
            2. The results are combined into a 'meta-space'.
            3. A final clustering is performed on the centroids of this combined space.
            """)

    with tabs[2]:
        st.subheader("Cluster Profiling (Averages)")
        analysis_cols = ['Customer Lifetime Value', 'Income', 'AvgFlightsPerMonth', 'TotalPointsRedeemed', 'LoyaltyScore']
        cluster_analysis = df.groupby('final_cluster')[analysis_cols].mean().reset_index()
        
        st.write("Cluster Characteristics (Radar Chart)")
        cluster_norm = (cluster_analysis[analysis_cols] - cluster_analysis[analysis_cols].min()) / (cluster_analysis[analysis_cols].max() - cluster_analysis[analysis_cols].min())
        
        fig_radar = go.Figure()
        for i in range(len(cluster_analysis)):
            fig_radar.add_trace(go.Scatterpolar(
                r=cluster_norm.iloc[i].values,
                theta=analysis_cols,
                fill='toself',
                name=f'Cluster {cluster_analysis.iloc[i]["final_cluster"]}'
            ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
        st.plotly_chart(fig_radar, use_container_width=True)
        
        st.write("Raw Data Averages:")
        st.dataframe(cluster_analysis.style.background_gradient(cmap='YlGnBu'))

    

else:
    st.info("Adjust the parameters in the sidebar and click 'Run Clustering' to begin.")
    st.image("https://raw.githubusercontent.com/streamlit/docs/main/public/images/tutorials/visualizing-data.png", caption="Visualization Example")
