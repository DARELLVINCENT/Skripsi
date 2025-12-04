##Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils import calculate_distance_matrix
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from datetime import datetime
from functools import reduce
from itertools import combinations
import random
from typing import List, cast
import plotly.graph_objects as go
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
#buat set centroid tetap sama setiap run
random.seed(42) 

#________#
st.title("Clustering ")

if "df" in st.session_state:
    df = st.session_state.df
    st.dataframe(df.head())
else:
    st.warning("Please upload your CSV file on the Home page first.")

if "df" in st.session_state:
    # Drop unnamed extra columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    try:
        df["tanggal"] = pd.to_datetime(df["tanggal"], format='mixed')
        df["qty"] = df["qty"].astype(str).str.replace(',', '').astype(int)
        df["harga"] = df["harga"].astype(str).str.replace(',', '').astype(float)
        df["total_harga"] = df["total_harga"].astype(str).str.replace(',', '').astype(float)
        
    except Exception as e:
        st.error(f"Data preprocessing failed: {e}")
        st.stop()

    # Rename
    df.rename(columns={"nama_customer": "customer_id"}, inplace=True)

    # --- Select Method ---
    st.subheader("Select Method and Fetures")
    method = st.selectbox("Select Clustering Method", ["K-Means", "K-Medoids", "DBSCAN", "GMM"])
    features = st.selectbox("Select Features", ["CLV", "MLRFM"])

        # 👉 Simpan pilihan user ke session_state
    st.session_state["last_method"] = method
    st.session_state["last_features"] = features

    # ---- placeholders agar nama selalu ada & Pylance tidak protes ----
    if 'cluster_source' not in st.session_state:
        st.session_state['cluster_source'] = None  # akan diisi DataFrame ['customer_id','Cluster']

    # untuk static analyzer; akan di-overwrite saat branch jalan
    clv_df = None
    mlrfm_scaled_df = None

    if method == "K-Means" and features == "CLV":
        # --- CLV ---
        st.subheader("K-Means Clustering using CLV")

            # CLV - Agregasi per customer (mirip versi sebelumnya)
        latest_date = df["tanggal"].max()

        customer_df = (
            df.groupby("customer_id")
            .agg(
                Lifespan=("tanggal", lambda x: (latest_date - x.min()).days + 1),  # hari aktif
                Frequency=("qty", "count"),                                        # jumlah transaksi
                Total_Value=("total_harga", "sum"),                                # total nilai belanja
            )
            .reset_index()
        )

        # Turunan metrik
        customer_df["Average"] = customer_df["Total_Value"] / customer_df["Frequency"]  # AOV
        profit_margin = 0.08
        customer_df["CLV"] = (customer_df["Average"] * profit_margin * customer_df["Frequency"] * customer_df["Lifespan"])  

        # Kolom yang ingin ditampilkan
        cols = ["customer_id", "Average", "Lifespan", "Frequency", "CLV"]

        st.dataframe(customer_df[cols].head(10))

        # --- Yeo-Johnson + MinMax ---
        st.subheader("Applying Yeo-Johnson Transformation and MinMax Scaling")

        clv = customer_df[['Average', 'Lifespan', 'Frequency', 'CLV']]
        pt = PowerTransformer(method='yeo-johnson')
        clv_transformed = pt.fit_transform(clv)
        scaler = MinMaxScaler()
        clv_scaled = scaler.fit_transform(clv_transformed)

        st.write("Data for K-Means Clustering")
        st.dataframe(pd.DataFrame(clv_scaled, columns=['Average', 'Lifespan', 'Frequency', 'CLV']).head(10))

        st.subheader("Select Elbow Method")
        # Buat DataFrame kerja untuk clustering & visualisasi
        clv_df = pd.DataFrame(
            np.column_stack((customer_df['customer_id'].values, clv_scaled)),
            columns=['customer_id', 'Average', 'Lifespan', 'Frequency', 'CLV']
        )
        # set maximum k for evaluation
        max_k = st.slider("Maximum number of clusters to test", min_value=3, max_value=15, value=10)

        inertias = []
        X = clv_scaled  

        K_range = range(2, max_k + 1)
        for k in K_range:
            model = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = model.fit_predict(X)
            inertias.append(model.inertia_)

        fig_elbow, ax = plt.subplots()
        ax.plot(K_range, inertias, marker='o')
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("SSE score")
        ax.set_title("Elbow Method for Optimal k")
        st.pyplot(fig_elbow)

        st.write("### Elbow method SSE")
        scores_df = pd.DataFrame({'k': list(K_range), 'SSE Score': inertias})
        st.dataframe(scores_df.style.format(precision=3))

        # --- K-Means Clustering ---
        st.subheader("K-Means Clustering")
        n_clusters = st.slider("Choose number of clusters", 2, 10, 3)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        clv_df['Cluster'] = kmeans.fit_predict(X)
        st.session_state["clv_cluster_df"] = clv_df.copy()
        st.session_state["cluster_source"] = "CLV"
        st.session_state["clv_method"] = "K-Means"
        

        # --- Show clustered data ---
        st.subheader("Clustered Customer Table")
        st.dataframe(clv_df)

        # --- Calculate evaluation metrics for the current k ---
        sil_score = silhouette_score(clv_scaled, clv_df['Cluster'])
        db_score = davies_bouldin_score(clv_scaled, clv_df['Cluster'])
        chi_score = calinski_harabasz_score(clv_scaled, clv_df['Cluster'])
      
        # --- Show Evaluation Metrics ---
        st.subheader("Evaluation Metrics for K-Means")
        st.success(f"**Silhouette Score:** {sil_score:.3f}")
        st.success(f"**Davies-Bouldin Score:** {db_score:.3f}")
        st.success(f"**Calinski-Harabasz Score:** {chi_score:3f}")

        st.subheader("Silhouette Plot Visualization")
        # Hitung silhouette per sample
        sample_silhouette_values = silhouette_samples(clv_scaled, clv_df['Cluster'])

        # Bungkus ke DataFrame supaya boolean indexing lebih jelas buat Pylance
        sil_df = pd.DataFrame({
            "Cluster": clv_df["Cluster"].astype(int),
            "silhouette": sample_silhouette_values
        })

        fig_sil, ax1 = plt.subplots(figsize=(8, 4))

        y_lower = 10  # posisi awal di sumbu y

        # Colormap
        cmap = cm.get_cmap("viridis")

        for i in range(n_clusters):
            # Ambil nilai silhouette utk cluster ke-i, sort, dan jadikan numpy array
            cluster_vals = (
                sil_df
                .loc[sil_df["Cluster"] == i, "silhouette"]
                .sort_values()
                .to_numpy()
            )

            size_cluster_i = len(cluster_vals)
            y_upper = y_lower + size_cluster_i

            # Warna per cluster
            color = cmap(float(i) / max(n_clusters - 1, 1))

            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_vals,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label cluster di tengah blok
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # update y_lower utk cluster berikutnya
            y_lower = y_upper + 10  # jarak antar cluster

        ax1.set_title("Silhouette Plot for the Various Clusters")
        ax1.set_xlabel("Silhouette Coefficient Values")
        ax1.set_ylabel("Cluster Label")

        # Garis vertikal rata-rata silhouette score
        ax1.axvline(x=float(sil_score), linestyle="--", linewidth=1)

        # Hilangkan ticks di sumbu y (karena kita sudah pakai label teks)
        ax1.set_yticks([])

        # Batas x
        ax1.set_xlim(-0.2, 1.0)

        st.pyplot(fig_sil)
        
        # --- Visualisasi 3D Klaster ---

        st.markdown("### Visualisasi Klaster (3D)")

        # daftar fitur untuk dijelajahi
        fitur = ["Average", "Lifespan", "Frequency", "CLV"]

        # buat semua kombinasi 3 fitur dan mapping label -> tuple fitur
        tiga_dim = list(combinations(fitur, 3))
        opsi_kombinasi = {f"{a} · {b} · {c}": (a, b, c) for a, b, c in tiga_dim}

        # pilih kombinasi melalui selectbox
        label_terpilih = st.selectbox("Pilih 3 fitur untuk scatter 3D", list(opsi_kombinasi.keys()))

        # ambil nama kolom sumbu
        sx, sy, sz = opsi_kombinasi[label_terpilih]

        # siapkan figure
        fig = px.scatter_3d(
            data_frame=clv_df,
            x=sx,
            y=sy,
            z=sz,
            color="Cluster",
            hover_name=clv_df.index.astype(str) if clv_df.index.name or clv_df.index.is_unique else None,
            hover_data={col: True for col in clv_df.columns},
            title=f"Scatter 3D | {sx} × {sy} × {sz}",
            height=620
        )

        # tweak marker & layout agar berbeda dari default
        fig.update_traces(marker=dict(size=4, opacity=0.85))
        fig.update_layout(
            legend_title_text="Kelompok",
            scene=dict(
                xaxis_title=sx,
                yaxis_title=sy,
                zaxis_title=sz
            ),
            margin=dict(l=10, r=10, t=60, b=10)
        )

        # render di Streamlit
        st.plotly_chart(fig, use_container_width=True)

        
        # --- Segmentation profile ---
        avg_aov = clv_df['Average'].mean()
        avg_lifespan = clv_df['Lifespan'].mean()
        avg_frequency = clv_df['Frequency'].mean()
        avg_clv = clv_df['CLV'].mean()

        st.subheader("Average CLV Values")
        global_avg = {
               
            'Average':  clv_df['Average'].mean(),
            'Lifespan': clv_df['Lifespan'].mean(),
            'Frequency': clv_df['Frequency'].mean(),
            'CLV':      clv_df['CLV'].mean(),
        }
        st.table(pd.DataFrame.from_dict(global_avg, orient='index', columns=['Value']).round(2))

        cluster_avg = clv_df.groupby('Cluster')[['Average', 'Lifespan', 'Frequency', 'CLV']].mean().reset_index()

        # --- Determine if feature is higher
        for col in ['Average', 'Lifespan', 'Frequency', 'CLV']:
            cluster_avg[f'{col}_high'] = cluster_avg[col] > global_avg[col]
        
        # Segment mapping
        segment_map = {
            (True,  True,  True,  True):  "High-value loyal customers",
            (False, True,  True,  True):  "High-value new customers",
            (True,  False, True,  True):  "Potential loyal customers",
            (False, False, True,  True):  "High-value lost customers",
            (True,  True,  False, True):  "Platinum customers",
            (False, True,  False, True):  "Consuming promotional customers",
            (True,  False, False, True):  "Potential consuming customers",
            (False, False, False, True):  "Consuming churn customers",
            (True,  True,  True,  False): "High frequency customers",
            (False, True,  True,  False): "Frequency promotional customers",
            (True,  False, True,  False): "Potential frequency customers",
            (False, False, True,  False): "Frequency churn customers",
            (True,  True,  False, False): "Low-cost consuming customers",
            (False, True,  False, False): "Uncertain new customers",
            (True,  False, False, False): "High-cost consuming customers",
            (False, False, False, False): "Uncertain lost customers",
        }

        # Assign segment
        cluster_avg['Segment'] = cluster_avg[['Average_high', 'Lifespan_high', 'Frequency_high', 'CLV_high']].apply(
            lambda x: segment_map[tuple(x)], axis=1
        )

        # Segment
        st.subheader("Cluster Segment Classification")
        st.dataframe(cluster_avg[['Cluster', 'Average', 'Lifespan', 'Frequency', 'CLV', 'Segment']])
        st.bar_chart(clv_df['Cluster'].value_counts())

        # --- Download result ---
        csv = clv_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Clustered Data as CSV", csv, "K-Means_clustered_customers.csv", "text/csv")
    

    # --- Kmeans MLRFM ---
    elif method == "K-Means" and features == "MLRFM":

        st.subheader("K-Means Clustering using MLRFM")
        max_date = df['tanggal'].max()
        df.rename(columns={"customer_id": "nama_customer"}, inplace=True)
        recency_df = df.groupby('nama_customer')['tanggal'].max().reset_index()
        recency_df['Recency'] = (max_date - recency_df['tanggal']).dt.days
        recency_df = recency_df[['nama_customer', 'Recency']]
        
        #Periods
        periods = {'365d': 365, '730d': 730, 'All': None}
        # periods = {'180d': 180, '365d': 365, '730d': 730}
        rfm_list = []

        for label, days in periods.items():
            if days is not None:
                start_date = max_date - pd.Timedelta(days=days)
                df_period = df[df['tanggal'] > start_date]
            else:
                df_period = df.copy()  # All data
                
            frequency = df_period.groupby('nama_customer')['no_invoice'].nunique().reset_index()
            monetary = df_period.groupby('nama_customer')['total_harga'].sum().reset_index()

            rfm = frequency.merge(monetary, on='nama_customer', how='outer')
            rfm = rfm.rename(columns={
                'no_invoice': f'Frequency_{label}',
                'total_harga': f'Monetary_{label}'
            })

            rfm_list.append(rfm)

        mlrfm = reduce(lambda left, right: pd.merge(left, right, on='nama_customer', how='outer'), rfm_list)
        final_mlrfm = recency_df.merge(mlrfm, on='nama_customer', how='outer')

        st.dataframe(final_mlrfm.head(10))    

        final_mlrfm.fillna({
            'Frequency_365d': 0, 'Monetary_365d': 0,
            'Frequency_730d': 0, 'Monetary_730d': 0,
            'Frequency_All': 0, 'Monetary_All': 0,
            'Recency': 999
        }, inplace=True)

        
        # final_mlrfm.fillna({
        #     'Frequency_180d': 0, 'Monetary_180d': 0,
        #     'Frequency_365d': 0, 'Monetary_365d': 0,
        #     'Frequency_730d': 0, 'Monetary_730d': 0,
        #     'Recency': 999
        # }, inplace=True)


        
        st.subheader("Yeo-Johnson Transformation & MinMax Scaling")

        mlrfm_scaled_df = final_mlrfm.copy()
        features_to_scale = ['Recency', 'Frequency_365d', 'Frequency_730d', 'Frequency_All',
                            'Monetary_365d', 'Monetary_730d', 'Monetary_All']
        # features_to_scale = ['Recency', 'Frequency_180d', 'Frequency_365d', 'Frequency_730d',
        #                     'Monetary_180d', 'Monetary_365d', 'Monetary_730d']

        # Apply Yeo-Johnson
        pt = PowerTransformer(method='yeo-johnson')
        mlrfm_scaled_df[features_to_scale] = pt.fit_transform(mlrfm_scaled_df[features_to_scale])

        # Apply MinMaxScaler
        scaler = MinMaxScaler()
        mlrfm_scaled_df[features_to_scale] = scaler.fit_transform(mlrfm_scaled_df[features_to_scale])

        st.dataframe(mlrfm_scaled_df.head(10))  

        st.subheader("Set Weights for Multi-Layer RFM")

        st.markdown("#### Frequency Weights")
        w_f_365 = st.number_input("Weight for Frequency_365d", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        w_f_730 = st.number_input("Weight for Frequency_730d", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        w_f_All = st.number_input("Weight for Frequency_All", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

        st.markdown("#### Monetary Weights")
        w_m_365 = st.number_input("Weight for Monetary_365d", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        w_m_730 = st.number_input("Weight for Monetary_730d", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        w_m_All = st.number_input("Weight for Monetary_All", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

        # Normalize weights
        total_f = w_f_365 + w_f_730 + w_f_All
        w_f_365 /= total_f
        w_f_730 /= total_f
        w_f_All /= total_f

        total_m = w_m_365 + w_m_730 + w_m_All
        w_m_365 /= total_m
        w_m_730 /= total_m
        w_m_All /= total_m

        # Compute combined Frequency & Monetary
        mlrfm_scaled_df['Multi_Layer_Frequency'] = (
            w_f_365 * mlrfm_scaled_df['Frequency_365d'] +
            w_f_730 * mlrfm_scaled_df['Frequency_730d'] +
            w_f_All * mlrfm_scaled_df['Frequency_All']
        )

        mlrfm_scaled_df['Multi_Layer_Monetary'] = (
            w_m_365 * mlrfm_scaled_df['Monetary_365d'] +
            w_m_730 * mlrfm_scaled_df['Monetary_730d'] +
            w_m_All * mlrfm_scaled_df['Monetary_All']
        )

        # w_f_180 = st.number_input("Weight for Frequency_180d", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        # w_f_365 = st.number_input("Weight for Frequency_365d", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        # w_f_730 = st.number_input("Weight for Frequency_730d", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

        # # === Monetary Weights (180, 365, 730) ===
        # st.markdown("#### Monetary Weights")
        # w_m_180 = st.number_input("Weight for Monetary_180d", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        # w_m_365 = st.number_input("Weight for Monetary_365d", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        # w_m_730 = st.number_input("Weight for Monetary_730d", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

        # # Normalize weights
        # total_f = w_f_180 + w_f_365 + w_f_730
        # w_f_180 /= total_f
        # w_f_365 /= total_f
        # w_f_730 /= total_f

        # total_m = w_m_180 + w_m_365 + w_m_730
        # w_m_180 /= total_m
        # w_m_365 /= total_m
        # w_m_730 /= total_m

        # # === Multi Layer Frequency & Monetary pakai 180/365/730 ===
        # mlrfm_scaled_df['Multi_Layer_Frequency'] = (
        #     w_f_180 * mlrfm_scaled_df['Frequency_180d'] +
        #     w_f_365 * mlrfm_scaled_df['Frequency_365d'] +
        #     w_f_730 * mlrfm_scaled_df['Frequency_730d']
        # )

        # mlrfm_scaled_df['Multi_Layer_Monetary'] = (
        #     w_m_180 * mlrfm_scaled_df['Monetary_180d'] +
        #     w_m_365 * mlrfm_scaled_df['Monetary_365d'] +
        #     w_m_730 * mlrfm_scaled_df['Monetary_730d']
        # )


        mlrfm_scaled_df = mlrfm_scaled_df[['nama_customer','Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']]
        st.dataframe(mlrfm_scaled_df.head(10))  

        # Bins
        bins = [0.0, 0.10, 0.30, 0.50, 0.70, 1.00]
        labels = [1, 2, 3, 4, 5]

        # Note: Recency Reverse
        mlrfm_scaled_df['Recency'] = pd.cut(mlrfm_scaled_df['Recency'], bins=bins, labels=labels[::-1], include_lowest=True).astype(int) * 0.33
        mlrfm_scaled_df['Multi_Layer_Frequency'] = pd.cut(mlrfm_scaled_df['Multi_Layer_Frequency'], bins=bins, labels=labels, include_lowest=True).astype(int) * 0.33
        mlrfm_scaled_df['Multi_Layer_Monetary'] = pd.cut(mlrfm_scaled_df['Multi_Layer_Monetary'], bins=bins, labels=labels, include_lowest=True).astype(int) * 0.33

        st.subheader("Final MLRFM Data")
        st.dataframe(mlrfm_scaled_df)  

        K_range = range(2, 11)

        st.subheader("Select Elbow Method")

        # Allow user to define max k
        max_k = st.slider("Maximum number of clusters to test (Elbow)", min_value=3, max_value=15, value=10)

        inertias = []
        X_feats = mlrfm_scaled_df[['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']].values

        K_range = range(2, max_k + 1)
        for k in K_range:
            model = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = model.fit_predict(X_feats)
            inertias.append(model.inertia_)
        
        # Plot Elbow
        fig_elbow, ax = plt.subplots()
        ax.plot(K_range, inertias, marker='o')
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("SSE Score")
        ax.set_title("Elbow Method For Optimal k")
        st.pyplot(fig_elbow)

        # Display scores for each k
        scores_df = pd.DataFrame({
            'k': list(K_range),
            'SSE Score': inertias,
        })
        st.dataframe(scores_df.style.format(precision=3))

        #--- K-Means Cluster ---
        st.subheader("K-Means Clustering")
        n_clusters = st.slider("Choose number of clusters", 2, 10, 5)
        X = mlrfm_scaled_df[['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']].values
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        mlrfm_scaled_df['Cluster'] = kmeans.fit_predict(X_feats)

        st.session_state["mlrfm_cluster_df"] = mlrfm_scaled_df.copy()
        st.session_state["cluster_source"] = "MLRFM"
        st.session_state["mlrfm_method"] = "K-Means"

        st.subheader("Clustered Data Table")
        st.dataframe(mlrfm_scaled_df[['nama_customer', 'Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Cluster']])

        # --- Calculate evaluation metrics for the current k ---
        sil_score = silhouette_score(mlrfm_scaled_df[['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Cluster']], mlrfm_scaled_df['Cluster'])
        db_score = davies_bouldin_score(mlrfm_scaled_df[['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Cluster']], mlrfm_scaled_df['Cluster'])
        chi_score = calinski_harabasz_score(mlrfm_scaled_df[['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Cluster']], mlrfm_scaled_df['Cluster'])
        
        # ---Silhouette, Davies-Bouldin ---
        st.subheader("Evaluation Metrics for K-Means")
        st.success(f"**Silhouette Score:** {sil_score:.3f}")
        st.success(f"**Davies-Bouldin Score:** {db_score:.3f}")
        st.success(f"**Calinski-Harabasz Score:** {chi_score:3f}")

        st.subheader("Silhouette Plot for K-Means Clustering")

        # Tetap pakai Cluster sebagai fitur (sesuai permintaan kamu)
        X_for_sil = mlrfm_scaled_df[
            ['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Cluster']
        ].to_numpy(dtype=float)

        labels_arr = mlrfm_scaled_df['Cluster'].to_numpy()

        # Hitung nilai silhouette per-sample, lalu pastikan jadi numpy array float
        silhouette_vals = silhouette_samples(X_for_sil, labels_arr)
        silhouette_vals = np.asarray(silhouette_vals, dtype=float)

        fig_sil, ax1 = plt.subplots(figsize=(8, 6))
        y_lower = 10

        # Hindari akses langsung plt.cm.nipy_spectral (Pylance nggak kenal),
        # pakai get_cmap supaya type checker senang
        cmap = plt.get_cmap("nipy_spectral")

        for i in range(n_clusters):
            # Mask cluster i
            mask = (labels_arr == i)

            # Ambil nilai silhouette cluster i
            ith_cluster_sil_vals = silhouette_vals[mask]

            # Pylance kadang nggak suka .sort(), jadi pakai np.sort
            ith_cluster_sil_vals = np.sort(ith_cluster_sil_vals)

            size_cluster_i = int(ith_cluster_sil_vals.shape[0])
            y_upper = y_lower + size_cluster_i

            color = cmap(float(i) / float(n_clusters))

            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0.0,
                ith_cluster_sil_vals,
                facecolor=color,
                edgecolor=color,
                alpha=0.7
            )

            # Label nomor cluster di tengah bar
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10  # spasi antar cluster

        # Garis vertikal rata-rata silhouette score
        ax1.axvline(x=float(sil_score), linestyle="--")

        ax1.set_xlabel("Silhouette coefficient")
        ax1.set_ylabel("Cluster")
        ax1.set_title("Silhouette Plot for K-Means Clustering")

        ax1.set_yticks([])

        # Pylance komplain kalau dikasih list, jadi kirim dua argumen float
        ax1.set_xlim(-0.1, 1.0)

        st.pyplot(fig_sil)

        st.markdown("### Visualisasi 3D Segmen Pelanggan")

        # Opsi interaktif yang tidak ada di versi awal
        ukuran_marker = st.slider("Ukuran titik", min_value=3, max_value=12, value=6, step=1)
        df_vis = mlrfm_scaled_df.rename(columns={
            "Recency": "R",
            "Multi_Layer_Frequency": "MLF",
            "Multi_Layer_Monetary": "MLM"
        })

        # Pilih cluster mana yang ingin ditampilkan
        daftar_cluster = sorted(df_vis["Cluster"].unique().tolist())
        pilihan_cluster = st.multiselect(
            "Tampilkan cluster",
            options=daftar_cluster,
            default=daftar_cluster
        )

        fig = go.Figure()

        # Tambahkan trace per cluster agar warna/legenda jelas
        for c in pilihan_cluster:
            subset = df_vis[df_vis["Cluster"] == c]
            # hovertext kustom agar berbeda dari default hover_data
            hovertext = (
                "Customer: " + subset["nama_customer"].astype(str) +
                "<br>R: " + subset["R"].round(2).astype(str) +
                "<br>MLF: " + subset["MLF"].round(2).astype(str) +
                "<br>MLM: " + subset["MLM"].round(2).astype(str)
            )

            fig.add_trace(
                go.Scatter3d(
                    x=subset["R"],
                    y=subset["MLF"],
                    z=subset["MLM"],
                    mode="markers",
                    name=f"Cluster {c}",
                    text=hovertext,
                    hovertemplate="%{text}<extra></extra>",
                    marker=dict(size=ukuran_marker, opacity=0.85)
                )
            )

        fig.update_layout(
            title="3D Customer Segmentation",
            legend_title_text="Cluster",
            scene=dict(
                xaxis_title="Recency (R)",
                yaxis_title="Multi-Layer Frequency (MLF)",
                zaxis_title="Multi-Layer Monetary (MLM)"
            ),
            margin=dict(l=10, r=10, t=60, b=10),
            height=640
        )

        st.plotly_chart(fig, use_container_width=True)


        # --- Segmentation profile ---
        avg_recency = mlrfm_scaled_df['Recency'].mean()
        avg_frequency = mlrfm_scaled_df['Multi_Layer_Frequency'].mean()
        avg_monetary = mlrfm_scaled_df['Multi_Layer_Monetary'].mean()

        st.subheader("Average MLRFM Values")
        global_avg = {
            'Recency': mlrfm_scaled_df['Recency'].mean(),
            'Multi_Layer_Frequency': mlrfm_scaled_df['Multi_Layer_Frequency'].mean(),
            'Multi_Layer_Monetary': mlrfm_scaled_df['Multi_Layer_Monetary'].mean()
        }

        st.table(pd.DataFrame.from_dict(global_avg, orient='index', columns=['Average']).round(2))

        cluster_avg = mlrfm_scaled_df.groupby('Cluster')[[ 'Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']].mean().reset_index()

        # --- Determine if feature is higher
        for col in ['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']:
            cluster_avg[f'{col}_high'] = cluster_avg[col] > global_avg[col]

        # segment_map = {
        # (True,  True,  True):  "Loyal Customers",              # R↑ F↑ M↑ — paling aktif & berbelanja tinggi
        # (True,  True,  False): "Potential loyal customers",    # R↑ F↑ M↓ — sering & baru-baru ini, nilai belanja masih kecil
        # (True,  False, True):  "Promising customers",          # R↑ F↓ M↑ — baru-baru ini, nilai tinggi tapi belum sering
        # (True,  False, False): "New customers",                # R↑ F↓ M↓ — baru beli & masih rendah
        # (False, True,  True):  "At risk (high value)",         # R↓ F↑ M↑ — dulu aktif & bernilai tinggi, sekarang mulai tidak aktif
        # (False, True,  False): "About to sleep",               # R↓ F↑ M↓ — dulu sering tapi nilai kecil, mulai tidak aktif
        # (False, False, True):  "Churned high spenders",        # R↓ F↓ M↑ — pernah belanja besar tapi sudah tidak aktif
        # (False, False, False): "Lost customers",               # R↓ F↓ M↓ — tidak aktif & nilai rendah
        # }
        segment_map = {
            (True,  True,  True):  "Loyal customers",          # R↑ F↑ M↑
            (True,  False, True):  "Promising customers",      # R↑ F↓ M↑
            (True,  False, False): "New customers",            # R↑ F↓ M↓
            (False, False, False): "Lost customers",           # R↓ F↓ M↓
            (False, True,  True):  "Lost customers 'Churned'",           # R↓ F↑ M↑ 
            (False, False, True): "Lost Customers 'High Value'",           # R↓ F↓ M↑ 
            (False, True,  False): "Lost Customers 'At Risk'"
        }
        
        # Assign segment
        cluster_avg['Segment'] = cluster_avg[['Recency_high', 'Multi_Layer_Frequency_high', 'Multi_Layer_Monetary_high']].apply(
            lambda x: segment_map[tuple(x)], axis=1
        )
        # Segment
        st.subheader("Cluster Segment Classification")
        st.dataframe(cluster_avg[['Cluster', 'Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Segment']])
        st.bar_chart(mlrfm_scaled_df['Cluster'].value_counts())

        # --- Download ---
        csv = mlrfm_scaled_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Clustered Data as CSV", csv, "mlrfm_clustered.csv", "text/csv")
            
    
    elif method == "K-Medoids" and features == "CLV":
        st.subheader("K-Mediods Clustering using CLV")
        
        latest_date = df["tanggal"].max()
        
        customer_df = (
            df.groupby("customer_id")
            .agg(
                Lifespan=("tanggal", lambda x: (latest_date - x.min()).days + 1),  # hari aktif
                Frequency=("qty", "count"),                                        # jumlah transaksi
                Total_Value=("total_harga", "sum"),                                # total nilai belanja
            )
            .reset_index()
        )

        # Turunan metrik
        customer_df["Average"] = customer_df["Total_Value"] / customer_df["Frequency"]  # AOV
        profit_margin = 0.08
        customer_df["CLV"] = (customer_df["Average"] * profit_margin * customer_df["Frequency"] * customer_df["Lifespan"])  

        cols = ["customer_id", "Average", "Lifespan", "Frequency", "CLV"]
        st.dataframe(customer_df[cols].head(10))

        st.subheader("Applying Yeo-Johnson Transformation and MinMax Scaling")

        clv = customer_df[['Average', 'Lifespan', 'Frequency', 'CLV']]
        pt = PowerTransformer(method='yeo-johnson')
        clv_transformed = pt.fit_transform(clv)

        scaler = MinMaxScaler()
        clv_scaled = scaler.fit_transform(clv_transformed)

        st.write("Data for K-Medoids Clustering ")
        st.dataframe(pd.DataFrame(clv_scaled, columns=['Average', 'Lifespan', 'Frequency', 'CLV']).head(10))

        st.subheader("Select Distance Metric for K-Medoids (CLV)")
        metric_options = {
            "Euclidean": "euclidean",
            "Manhattan (L1)": "manhattan",
            "Cosine": "cosine"
        }
        metric_name = st.selectbox(
            "Distance metric",
            list(metric_options.keys()),
            key="kmedoids_clv_metric"
        )
        metric = metric_options[metric_name]

        st.subheader("Select Elbow Method")
        clv_df = pd.DataFrame(
            np.column_stack((customer_df['customer_id'].values, clv_scaled)),
            columns=['customer_id', 'Average', 'Lifespan', 'Frequency', 'CLV']
        )
        st.dataframe(clv_df.head(10))
    
        max_k = st.slider("Maximum number of clusters to test", min_value=3, max_value=15, value=10)

        distance_matrix = pairwise_distances(clv_scaled, metric=metric)

        inertias = []
        K_range = range(2, max_k + 1)

        for k in K_range:
            try:
                initial_medoids_k = random.sample(range(len(clv_scaled)), k)
                kmedoids_eval = kmedoids(distance_matrix, initial_medoids_k, data_type='distance_matrix')
                kmedoids_eval.process()

                raw_clusters = kmedoids_eval.get_clusters()
                clusters_k: List[List[int]] = cast(List[List[int]], raw_clusters)

                raw_medoids = kmedoids_eval.get_medoids()
                medoids_k: List[int] = cast(List[int], raw_medoids)

                labels_k = np.zeros(len(clv_scaled), dtype=int)
                for cid, clus in enumerate(clusters_k):
                    for idx in clus:
                        labels_k[idx] = cid

                inertia_k = 0.0
                for cid, clus in enumerate(clusters_k):
                    medoid_idx = medoids_k[cid]
                    inertia_k += sum(
                        np.linalg.norm(clv_scaled[i] - clv_scaled[medoid_idx]) 
                        for i in clus
                    )

                inertias.append(inertia_k)

            except Exception as e:
                inertias.append(None)
                st.warning(f"Clustering failed for k={k}: {e}")

        # --- Plot Elbow ---
        st.write("### Elbow Method (SSE)")
        fig_elbow, ax = plt.subplots()
        ax.plot(K_range, inertias, marker='o', color='blue')
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("SSE Score")
        ax.set_title("Elbow Method for Optimal k")
        st.pyplot(fig_elbow)

        # --- Show Evaluation Table ---
        st.write("### SSE Score")
        scores_df = pd.DataFrame({
            'k': list(K_range),
            'SSE Score': inertias
        })
        st.dataframe(scores_df.style.format(precision=3))

        # --- K-Medoids ---
        st.subheader("K-Medoids Clustering")
        n_clusters = st.slider("Choose number of clusters", 2, 10, 3)

        # Select initial medoids 
        initial_medoids = list(range(n_clusters))
        kmedoids_instance = kmedoids(distance_matrix, initial_medoids, data_type='distance_matrix')
        kmedoids_instance.process()

        raw_clusters = kmedoids_instance.get_clusters()
        clusters: List[List[int]] = cast(List[List[int]], raw_clusters)

        labels = np.zeros(len(clv_scaled), dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for index in cluster:
                labels[index] = cluster_id

        clv_df['Cluster'] = labels
        st.session_state["clv_cluster_df"] = clv_df.copy()
        st.session_state["cluster_source"] = "CLV"
        st.session_state["clv_method"] = "K-Medoids"


        # --- Show clustered data ---
        st.subheader("Clustered Customer Table")
        st.dataframe(clv_df)

        

        # --- Visualisasi 3D Klaster ---

        st.markdown("### Visualisasi Klaster (3D)")

        # daftar fitur untuk dijelajahi
        fitur = ["Average", "Lifespan", "Frequency", "CLV"]

        # buat semua kombinasi 3 fitur dan mapping label -> tuple fitur
        tiga_dim = list(combinations(fitur, 3))
        opsi_kombinasi = {f"{a} · {b} · {c}": (a, b, c) for a, b, c in tiga_dim}

        # pilih kombinasi melalui selectbox
        label_terpilih = st.selectbox("Pilih 3 fitur untuk scatter 3D", list(opsi_kombinasi.keys()))

        # ambil nama kolom sumbu
        sx, sy, sz = opsi_kombinasi[label_terpilih]

        # siapkan figure
        fig = px.scatter_3d(
            data_frame=clv_df,
            x=sx,
            y=sy,
            z=sz,
            color="Cluster",
            hover_name=clv_df.index.astype(str) if clv_df.index.name or clv_df.index.is_unique else None,
            hover_data={col: True for col in clv_df.columns},
            title=f"Scatter 3D | {sx} × {sy} × {sz}",
            height=620
        )

        # tweak marker & layout agar berbeda dari default
        fig.update_traces(marker=dict(size=4, opacity=0.85))
        fig.update_layout(
            legend_title_text="Kelompok",
            scene=dict(
                xaxis_title=sx,
                yaxis_title=sy,
                zaxis_title=sz
            ),
            margin=dict(l=10, r=10, t=60, b=10)
        )

        # render di Streamlit
        st.plotly_chart(fig, use_container_width=True)


        # --- Calculate evaluation metrics for the current k ---
        sil_score = silhouette_score(clv_scaled, clv_df['Cluster'])
        db_score = davies_bouldin_score(clv_scaled, clv_df['Cluster'])
        ch_score = calinski_harabasz_score(clv_scaled, clv_df['Cluster'])

        # --- Show Evaluation Metrics ---
        st.subheader("Evaluation Metrics for K-Medoids")
        st.success(f"**Silhouette Score:** {sil_score:.3f}")
        st.success(f"**Davies-Bouldin Score:** {db_score:.3f}")
        st.success(f"**Calinski-Harabasz Score:** {ch_score:3f}")

        st.subheader("Silhouette Plot Visualization")
        # Hitung silhouette per sample
      
        # Hitung silhouette per sample
        sample_silhouette_values = silhouette_samples(clv_scaled, clv_df['Cluster'])

        # Bungkus ke DataFrame supaya boolean indexing lebih jelas buat Pylance
        sil_df = pd.DataFrame({
            "Cluster": clv_df["Cluster"].astype(int),
            "silhouette": sample_silhouette_values
        })

        fig_sil, ax1 = plt.subplots(figsize=(8, 4))

        y_lower = 10  # posisi awal di sumbu y

        # Colormap
        cmap = cm.get_cmap("viridis")

        for i in range(n_clusters):
            # Ambil nilai silhouette utk cluster ke-i, sort, dan jadikan numpy array
            cluster_vals = (
                sil_df
                .loc[sil_df["Cluster"] == i, "silhouette"]
                .sort_values()
                .to_numpy()
            )

            size_cluster_i = len(cluster_vals)
            y_upper = y_lower + size_cluster_i

            # Warna per cluster
            color = cmap(float(i) / max(n_clusters - 1, 1))

            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_vals,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label cluster di tengah blok
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # update y_lower utk cluster berikutnya
            y_lower = y_upper + 10  # jarak antar cluster

        ax1.set_title("Silhouette Plot for the Various Clusters")
        ax1.set_xlabel("Silhouette Coefficient Values")
        ax1.set_ylabel("Cluster Label")

        # Garis vertikal rata-rata silhouette score
        ax1.axvline(x=float(sil_score), linestyle="--", linewidth=1)

        # Hilangkan ticks di sumbu y (karena kita sudah pakai label teks)
        ax1.set_yticks([])

        # Batas x
        ax1.set_xlim(-0.2, 1.0)

        st.pyplot(fig_sil)
        
        # --- Segmentation profile ---
        avg_aov = clv_df['Average'].mean()
        avg_lifespan = clv_df['Lifespan'].mean()
        avg_frequency = clv_df['Frequency'].mean()
        avg_clv = clv_df['CLV'].mean()

        st.subheader("Average CLV Values")
        global_avg = {
            'Average': clv_df['Average'].mean(),
            'Lifespan': clv_df['Lifespan'].mean(),
            'Frequency': clv_df['Frequency'].mean(),
            'CLV': clv_df['CLV'].mean()
        }

        st.table(pd.DataFrame.from_dict(global_avg, orient='index', columns=['Average']).round(2))

        cluster_avg = clv_df.groupby('Cluster')[['Average', 'Lifespan', 'Frequency', 'CLV']].mean().reset_index()

        # --- Determine if feature is higher
        for col in ['Average', 'Lifespan', 'Frequency', 'CLV']:
            cluster_avg[f'{col}_high'] = cluster_avg[col] > global_avg[col]
        
        
        # Segment mapping
        segment_map = {
            (True,  True,  True,  True):  "High-value loyal customers",
            (False, True,  True,  True):  "High-value new customers",
            (True,  False, True,  True):  "Potential loyal customers",
            (False, False, True,  True):  "High-value lost customers",
            (True,  True,  False, True):  "Platinum customers",
            (False, True,  False, True):  "Consuming promotional customers",
            (True,  False, False, True):  "Potential consuming customers",
            (False, False, False, True):  "Consuming churn customers",
            (True,  True,  True,  False): "High frequency customers",
            (False, True,  True,  False): "Frequency promotional customers",
            (True,  False, True,  False): "Potential frequency customers",
            (False, False, True,  False): "Frequency churn customers",
            (True,  True,  False, False): "Low-cost consuming customers",
            (False, True,  False, False): "Uncertain new customers",
            (True,  False, False, False): "High-cost consuming customers",
            (False, False, False, False): "Uncertain lost customers",
        }

        # Assign segment
        cluster_avg['Segment'] = cluster_avg[['Average_high', 'Lifespan_high', 'Frequency_high', 'CLV_high']].apply(
            lambda x: segment_map[tuple(x)], axis=1
        )
        # Segment
        st.subheader("Cluster Segment Classification")
        st.dataframe(cluster_avg[['Cluster', 'Average', 'Lifespan', 'Frequency', 'CLV', 'Segment']])
        st.bar_chart(clv_df['Cluster'].value_counts())

        # --- Download result ---
        csv = clv_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Clustered Data as CSV", csv, "K-Medoids_LRFM_clustered_customers.csv", "text/csv")


    elif method == "K-Medoids" and features == "MLRFM":  
        st.subheader("K-Medoids Clustering using MLRFM")

        max_date = df['tanggal'].max()
        df.rename(columns={"customer_id": "nama_customer"}, inplace=True)
        recency_df = df.groupby('nama_customer')['tanggal'].max().reset_index()
        recency_df['Recency'] = (max_date - recency_df['tanggal']).dt.days
        recency_df = recency_df[['nama_customer', 'Recency']]
        
        #Periods
        periods = {'365d': 365, '730d': 730, 'All': None}

        rfm_list = []

        for label, days in periods.items():
            if days is not None:
                start_date = max_date - pd.Timedelta(days=days)
                df_period = df[df['tanggal'] > start_date]
            else:
                df_period = df.copy()  # All data
                
            frequency = df_period.groupby('nama_customer')['no_invoice'].nunique().reset_index()
            monetary = df_period.groupby('nama_customer')['total_harga'].sum().reset_index()

            rfm = frequency.merge(monetary, on='nama_customer', how='outer')
            rfm = rfm.rename(columns={
                'no_invoice': f'Frequency_{label}',
                'total_harga': f'Monetary_{label}'
            })

            rfm_list.append(rfm)

        mlrfm = reduce(lambda left, right: pd.merge(left, right, on='nama_customer', how='outer'), rfm_list)
        final_mlrfm = recency_df.merge(mlrfm, on='nama_customer', how='outer')

        st.dataframe(final_mlrfm.head(10))    

        final_mlrfm.fillna({
            'Frequency_365d': 0, 'Monetary_365d': 0,
            'Frequency_730d': 0, 'Monetary_730d': 0,
            'Frequency_All': 0, 'Monetary_All': 0,
            'Recency': 999
        }, inplace=True)

        
        st.subheader("Yeo-Johnson Transformation & MinMax Scaling")

        mlrfm_scaled_df = final_mlrfm.copy()
        features_to_scale = ['Recency', 'Frequency_365d', 'Frequency_730d', 'Frequency_All',
                            'Monetary_365d', 'Monetary_730d', 'Monetary_All']

        # Apply Yeo-Johnson
        pt = PowerTransformer(method='yeo-johnson')
        mlrfm_scaled_df[features_to_scale] = pt.fit_transform(mlrfm_scaled_df[features_to_scale])

        # Apply MinMaxScaler
        scaler = MinMaxScaler()
        mlrfm_scaled_df[features_to_scale] = scaler.fit_transform(mlrfm_scaled_df[features_to_scale])

        st.dataframe(mlrfm_scaled_df.head(10))  

        st.subheader("Set Weights for Multi-Layer RFM")

        st.markdown("#### Frequency Weights")
        w_f_365 = st.number_input("Weight for Frequency_365d", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        w_f_730 = st.number_input("Weight for Frequency_730d", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        w_f_All = st.number_input("Weight for Frequency_All", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

        st.markdown("#### Monetary Weights")
        w_m_365 = st.number_input("Weight for Monetary_365d", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        w_m_730 = st.number_input("Weight for Monetary_730d", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        w_m_All = st.number_input("Weight for Monetary_All", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

        # Normalize weights
        total_f = w_f_365 + w_f_730 + w_f_All
        w_f_365 /= total_f
        w_f_730 /= total_f
        w_f_All /= total_f

        total_m = w_m_365 + w_m_730 + w_m_All
        w_m_365 /= total_m
        w_m_730 /= total_m
        w_m_All /= total_m

        # Compute combined Frequency & Monetary
        mlrfm_scaled_df['Multi_Layer_Frequency'] = (
            w_f_365 * mlrfm_scaled_df['Frequency_365d'] +
            w_f_730 * mlrfm_scaled_df['Frequency_730d'] +
            w_f_All * mlrfm_scaled_df['Frequency_All']
        )

        mlrfm_scaled_df['Multi_Layer_Monetary'] = (
            w_m_365 * mlrfm_scaled_df['Monetary_365d'] +
            w_m_730 * mlrfm_scaled_df['Monetary_730d'] +
            w_m_All * mlrfm_scaled_df['Monetary_All']
        )
        mlrfm_scaled_df = mlrfm_scaled_df[['nama_customer','Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']]
        st.dataframe(mlrfm_scaled_df.head(10))  

        # Bins
        bins = [0.0, 0.10, 0.30, 0.50, 0.70, 1.00]
        labels = [1, 2, 3, 4, 5]

        # Note: Recency Reverse
        mlrfm_scaled_df['Recency'] = pd.cut(mlrfm_scaled_df['Recency'], bins=bins, labels=labels[::-1], include_lowest=True).astype(int) * 0.33
        mlrfm_scaled_df['Multi_Layer_Frequency'] = pd.cut(mlrfm_scaled_df['Multi_Layer_Frequency'], bins=bins, labels=labels, include_lowest=True).astype(int) * 0.33
        mlrfm_scaled_df['Multi_Layer_Monetary'] = pd.cut(mlrfm_scaled_df['Multi_Layer_Monetary'], bins=bins, labels=labels, include_lowest=True).astype(int) * 0.33

        st.subheader("Final MLRFM Data")
        st.dataframe(mlrfm_scaled_df)  

        st.subheader("Select Distance Metric for K-Medoids (MLRFM)")
        metric_options = {
            "Euclidean": "euclidean",
            "Manhattan (L1)": "manhattan",
            "Cosine": "cosine"
        }
        metric_name = st.selectbox(
            "Distance metric",
            list(metric_options.keys()),
            key="kmedoids_mlrfm_metric"
        )
        metric = metric_options[metric_name]

        st.subheader("Select Elbow Method")
        features = ['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']

        # Pakai numpy array, bukan DataFrame
        X_scaled = mlrfm_scaled_df[features].to_numpy()

        # 👉 pakai pairwise_distances dengan metric pilihan
        distance_matrix = pairwise_distances(X_scaled, metric=metric)

        # --- Elbow Method ---
        max_k = st.slider("Maximum number of clusters to test (Elbow)", min_value=3, max_value=15, value=10)
        K_range = range(2, max_k + 1)

        inertias = []

        for k in K_range:
            try:
                initial_medoids_k = random.sample(range(len(X_scaled)), k)
                kmedoids_eval = kmedoids(distance_matrix, initial_medoids_k, data_type='distance_matrix')
                kmedoids_eval.process()

                raw_clusters = kmedoids_eval.get_clusters()
                clusters_k: List[List[int]] = cast(List[List[int]], raw_clusters)

                raw_medoids = kmedoids_eval.get_medoids()
                medoids: List[int] = cast(List[int], raw_medoids)

                inertia_k = 0.0
                for cid, idx_list in enumerate(clusters_k):
                    m = medoids[cid]
                    diffs = X_scaled[idx_list] - X_scaled[m]
                    dists = np.linalg.norm(diffs, axis=1)
                    inertia_k += float(dists.sum())

                inertias.append(inertia_k)

            except Exception as e:
                inertias.append(None)
                st.warning(f"Clustering failed for k={k}: {e}")

        # --- Plot Elbow ---
        fig_elbow, ax = plt.subplots()
        ax.plot(K_range, inertias, marker='o')
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("SSE Score")
        ax.set_title("Elbow Method For Optimal k")
        st.pyplot(fig_elbow)

        scores_df = pd.DataFrame({'k': list(K_range), 'SSE Score': inertias})
        st.dataframe(scores_df.style.format(precision=3))

        # --- K-Medoids final ---
        st.subheader("K-Medoids Clustering")
        n_clusters = st.slider("Choose number of clusters", 2, 10, 5)

        initial_medoids = random.sample(range(len(X_scaled)), n_clusters)
        kmedoids_instance = kmedoids(distance_matrix, initial_medoids, data_type='distance_matrix')
        kmedoids_instance.process()

        raw_clusters = kmedoids_instance.get_clusters()
        if not isinstance(raw_clusters, list):
            st.error("Clustering failed: Invalid cluster format")
            st.stop()

        clusters: List[List[int]] = cast(List[List[int]], raw_clusters)

        cluster_labels = np.zeros(len(X_scaled), dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for index in cluster:
                cluster_labels[index] = cluster_id

        mlrfm_scaled_df['Cluster'] = cluster_labels
        st.session_state["mlrfm_cluster_df"] = mlrfm_scaled_df.copy()
        st.session_state["cluster_source"] = "MLRFM"
        st.session_state["mlrfm_method"] = "K-Medoids"


        st.subheader("Clustered Data Table")
        st.dataframe(mlrfm_scaled_df[['nama_customer', 'Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Cluster']])

        # --- Calculate evaluation metrics for the current k ---
        sil_score = silhouette_score(mlrfm_scaled_df[['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Cluster']], mlrfm_scaled_df['Cluster'])
        db_score = davies_bouldin_score(mlrfm_scaled_df[['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Cluster']], mlrfm_scaled_df['Cluster'])
        chi_score = calinski_harabasz_score(mlrfm_scaled_df[['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Cluster']], mlrfm_scaled_df['Cluster'])

        # ---Silhouette, Davies-Bouldin ---
        st.subheader("Evaluation Metrics for K-Medoids")
        st.success(f"**Silhouette Score:** {sil_score:.3f}")
        st.success(f"**Davies-Bouldin Score:** {db_score:.3f}")
        st.success(f"**Calinski-Harabasz Score:** {chi_score:3f}")

        # --- Silhouette Plot for K-Medoids ---
        st.subheader("Silhouette Plot for K-Medoids Clustering")

        # Tetap pakai Cluster sebagai fitur (sesuai permintaan kamu)
        X_for_sil = mlrfm_scaled_df[
            ['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Cluster']
        ].to_numpy(dtype=float)

        labels_arr = mlrfm_scaled_df['Cluster'].to_numpy()

        # Hitung nilai silhouette per-sample, lalu pastikan jadi numpy array float
        silhouette_vals = silhouette_samples(X_for_sil, labels_arr)
        silhouette_vals = np.asarray(silhouette_vals, dtype=float)

        fig_sil, ax1 = plt.subplots(figsize=(8, 6))
        y_lower = 10

        # Hindari akses langsung plt.cm.nipy_spectral (Pylance nggak kenal),
        # pakai get_cmap supaya type checker senang
        cmap = plt.get_cmap("nipy_spectral")

        for i in range(n_clusters):
            # Mask cluster i
            mask = (labels_arr == i)

            # Ambil nilai silhouette cluster i
            ith_cluster_sil_vals = silhouette_vals[mask]

            # Pylance kadang nggak suka .sort(), jadi pakai np.sort
            ith_cluster_sil_vals = np.sort(ith_cluster_sil_vals)

            size_cluster_i = int(ith_cluster_sil_vals.shape[0])
            y_upper = y_lower + size_cluster_i

            color = cmap(float(i) / float(n_clusters))

            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0.0,
                ith_cluster_sil_vals,
                facecolor=color,
                edgecolor=color,
                alpha=0.7
            )

            # Label nomor cluster di tengah bar
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10  # spasi antar cluster

        # Garis vertikal rata-rata silhouette score
        ax1.axvline(x=float(sil_score), linestyle="--")

        ax1.set_xlabel("Silhouette coefficient")
        ax1.set_ylabel("Cluster")
        ax1.set_title("Silhouette Plot for K-Medoids Clustering")

        ax1.set_yticks([])

        # Pylance komplain kalau dikasih list, jadi kirim dua argumen float
        ax1.set_xlim(-0.1, 1.0)

        st.pyplot(fig_sil)
            

        st.markdown("### Visualisasi 3D Segmen Pelanggan")

        # Opsi interaktif yang tidak ada di versi awal
        ukuran_marker = st.slider("Ukuran titik", min_value=3, max_value=12, value=6, step=1)
        df_vis = mlrfm_scaled_df.rename(columns={
            "Recency": "R",
            "Multi_Layer_Frequency": "MLF",
            "Multi_Layer_Monetary": "MLM"
        })

        # Pilih cluster mana yang ingin ditampilkan
        daftar_cluster = sorted(df_vis["Cluster"].unique().tolist())
        pilihan_cluster = st.multiselect(
            "Tampilkan cluster",
            options=daftar_cluster,
            default=daftar_cluster
        )

        fig = go.Figure()

        # Tambahkan trace per cluster agar warna/legenda jelas
        for c in pilihan_cluster:
            subset = df_vis[df_vis["Cluster"] == c]
            # hovertext kustom agar berbeda dari default hover_data
            hovertext = (
                "Customer: " + subset["nama_customer"].astype(str) +
                "<br>R: " + subset["R"].round(2).astype(str) +
                "<br>MLF: " + subset["MLF"].round(2).astype(str) +
                "<br>MLM: " + subset["MLM"].round(2).astype(str)
            )

            fig.add_trace(
                go.Scatter3d(
                    x=subset["R"],
                    y=subset["MLF"],
                    z=subset["MLM"],
                    mode="markers",
                    name=f"Cluster {c}",
                    text=hovertext,
                    hovertemplate="%{text}<extra></extra>",
                    marker=dict(size=ukuran_marker, opacity=0.85)
                )
            )

        fig.update_layout(
            title="3D Customer Segmentation",
            legend_title_text="Cluster",
            scene=dict(
                xaxis_title="Recency (R)",
                yaxis_title="Multi-Layer Frequency (MLF)",
                zaxis_title="Multi-Layer Monetary (MLM)"
            ),
            margin=dict(l=10, r=10, t=60, b=10),
            height=640
        )

        st.plotly_chart(fig, use_container_width=True)


        # --- Segmentation profile ---
        avg_recency = mlrfm_scaled_df['Recency'].mean()
        avg_frequency = mlrfm_scaled_df['Multi_Layer_Frequency'].mean()
        avg_monetary = mlrfm_scaled_df['Multi_Layer_Monetary'].mean()

        st.subheader("Average MLRFM Values")
        global_avg = {
            'Recency': mlrfm_scaled_df['Recency'].mean(),
            'Multi_Layer_Frequency': mlrfm_scaled_df['Multi_Layer_Frequency'].mean(),
            'Multi_Layer_Monetary': mlrfm_scaled_df['Multi_Layer_Monetary'].mean()
        }

        st.table(pd.DataFrame.from_dict(global_avg, orient='index', columns=['Average']).round(2))

        cluster_avg = mlrfm_scaled_df.groupby('Cluster')[[ 'Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']].mean().reset_index()

        # --- Determine if feature is higher
        for col in ['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']:
            cluster_avg[f'{col}_high'] = cluster_avg[col] > global_avg[col]

        # segment_map = {
        # (True,  True,  True):  "Loyal Customers",              # R↑ F↑ M↑ — paling aktif & berbelanja tinggi
        # (True,  True,  False): "Potential loyal customers",    # R↑ F↑ M↓ — sering & baru-baru ini, nilai belanja masih kecil
        # (True,  False, True):  "Promising customers",          # R↑ F↓ M↑ — baru-baru ini, nilai tinggi tapi belum sering
        # (True,  False, False): "New customers",                # R↑ F↓ M↓ — baru beli & masih rendah
        # (False, True,  True):  "At risk (high value)",         # R↓ F↑ M↑ — dulu aktif & bernilai tinggi, sekarang mulai tidak aktif
        # (False, True,  False): "About to sleep",               # R↓ F↑ M↓ — dulu sering tapi nilai kecil, mulai tidak aktif
        # (False, False, True):  "Churned high spenders",        # R↓ F↓ M↑ — pernah belanja besar tapi sudah tidak aktif
        # (False, False, False): "Lost customers",               # R↓ F↓ M↓ — tidak aktif & nilai rendah
        # }

        segment_map = {
            (True,  True,  True):  "Loyal customers",          # R↑ F↑ M↑
            (True,  False, True):  "Promising customers",      # R↑ F↓ M↑
            (True,  False, False): "New customers",            # R↑ F↓ M↓
            (False, False, False): "Lost customers",           # R↓ F↓ M↓
            (False, True,  True):  "Lost customers 'Churned'",           # R↓ F↑ M↑ 
            (False, False, True): "Lost Customers 'High Value'"           # R↓ F↓ M↑ 
        }
        

        # Assign segment
        cluster_avg['Segment'] = cluster_avg[['Recency_high', 'Multi_Layer_Frequency_high', 'Multi_Layer_Monetary_high']].apply(
            lambda x: segment_map[tuple(x)], axis=1
        )
        # Segment
        st.subheader("Cluster Segment Classification")
        st.dataframe(cluster_avg[['Cluster', 'Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Segment']])
        st.bar_chart(mlrfm_scaled_df['Cluster'].value_counts())

        # --- Download ---
        csv = mlrfm_scaled_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Clustered Data as CSV", csv, "K-Medoids_mlrfm_clustered.csv", "text/csv")

    elif method == "DBSCAN" and features == "CLV":  
        
        st.subheader("DBSCAN Clustering untuk CLV")

        latest_date = df["tanggal"].max()
        
        customer_df = (
        df.groupby("customer_id")
            .agg(
                Lifespan=("tanggal", lambda x: (latest_date - x.min()).days + 1),  # hari aktif
                Frequency=("qty", "count"),                                # jumlah transaksi
                Total_Value=("total_harga", "sum"),                                  # total nilai belanja
            )
            .reset_index()
        )

        # Turunan metrik
        customer_df["Average"] = customer_df["Total_Value"] / customer_df["Frequency"]  # AOV
        profit_margin = 0.08
        customer_df["CLV"] = (customer_df["Average"] * profit_margin * customer_df["Frequency"] * customer_df["Lifespan"])  


        
        st.dataframe(customer_df.head(10))

    
        st.subheader("Applying Yeo-Johnson Transformation and MinMax Scaling")

        clv_dbscan = customer_df[['Average', 'Lifespan', 'Frequency', 'CLV']]
        pt = PowerTransformer(method='yeo-johnson')
        clvdbs_transform = pt.fit_transform(clv_dbscan)

        scaler = MinMaxScaler()
        clvdbs_scaled = scaler.fit_transform(clvdbs_transform)
        st.dataframe(pd.DataFrame(clvdbs_scaled, columns=['Average', 'Lifespan', 'Frequency', 'CLV']).head(10))
       
        st.write("Data for DBSCAN Clustering")

        clv_df = pd.DataFrame(
            np.column_stack((customer_df['customer_id'].values, clvdbs_scaled)),
            columns=['customer_id', 'Average', 'Lifespan', 'Frequency', 'CLV']
        )
        st.dataframe(clv_df.head(10))

        # --- DBSCAN ---
        st.subheader("DBSCAN Clustering")

        # ==========================
        # 1. PILIH DISTANCE METRIC
        # ==========================
        st.markdown("#### Distance Metric")
        metric_label = st.selectbox(
            "Pilih distance metric untuk DBSCAN & Silhouette",
            ["Euclidean", "Manhattan", "Cosine"]
        )

        metric_map = {
            "Euclidean": "euclidean",
            "Manhattan": "manhattan",
            "Cosine": "cosine"
        }
        distance_metric = metric_map[metric_label]

        # ==========================
        # 2. K-Distance (Elbow Method for eps)
        # ==========================
        st.subheader("Elbow Method for Optimal eps (k-distance plot)")

        eps = st.slider("Set epsilon (eps)", min_value=0.01, max_value=2.0, value=0.16, step=0.01)
        min_samples = st.slider("Set min_samples", min_value=1, max_value=20, value=6)

        k = min_samples
        nn = NearestNeighbors(n_neighbors=k, metric=distance_metric)
        nn_fit = nn.fit(clvdbs_scaled)
        distances, _ = nn_fit.kneighbors(clvdbs_scaled)
        distances = np.sort(distances[:, -1])  # sort k-th NN distances

        fig_eps, ax_eps = plt.subplots()
        ax_eps.plot(distances)
        ax_eps.set_title(f"k-Distance Graph (k={k}, metric={distance_metric})")
        ax_eps.set_xlabel("Data Points sorted by distance")
        ax_eps.set_ylabel(f"{k}-NN Distance")
        st.pyplot(fig_eps)

        # ==========================
        # 3. DBSCAN dengan metric terpilih
        # ==========================
        db = DBSCAN(eps=eps, min_samples=min_samples, metric=distance_metric)
        labels = db.fit_predict(clvdbs_scaled)

        clv_df['Cluster'] = labels

        st.session_state["clv_cluster_df"] = clv_df.copy()
        st.session_state["cluster_source"] = "CLV"
        st.session_state["clv_method"] = "DBSCAN"

        # --- Visualization ---
        st.subheader("Cluster Visualization")
        features_visualitazion = ['Average', 'Lifespan', 'Frequency', 'CLV']
        combinations_3d = list(combinations(features_visualitazion, 3))

        # selectbox
        combination_labels = [f"{x[0]}, {x[1]}, {x[2]}" for x in combinations_3d]
        selected_label = st.selectbox("Pick Combination for 3D Scatter Plot", combination_labels)

        # pick combination
        selected_features = selected_label.split(", ")  

        # Plot 3D 
        fig = px.scatter_3d(
            clv_df,
            x=selected_features[0],
            y=selected_features[1],
            z=selected_features[2],
            color='Cluster',
            hover_data=clv_df.columns,
            title=f"3D Scatter Plot by Feature: {selected_features[0]} vs {selected_features[1]} vs {selected_features[2]}"
        )

        st.plotly_chart(fig)

        # --- Evaluation Scores (ignore noise label -1)
        valid_labels = clv_df['Cluster'].unique()
        valid_labels = valid_labels[valid_labels != -1]
        n_clusters = len(valid_labels)

        st.subheader("Clustered Customer Table")
        st.dataframe(clv_df)

        st.write("### Evaluation Metrics")
        if n_clusters > 1:
            mask = clv_df['Cluster'] != -1

            X_valid = clvdbs_scaled[mask]
            labels_valid = np.asarray(clv_df.loc[mask, 'Cluster'])

            # Silhouette pakai metric yang dipilih
            sil_score = silhouette_score(X_valid, labels_valid, metric=distance_metric)

            st.markdown(f"**Number of Clusters (excluding noise):** {n_clusters}")
            st.success(f"**Silhouette Score (metric = {distance_metric}):** {sil_score:.4f}")

            # DBI & CH hanya logis untuk Euclidean di sklearn
            if distance_metric == "euclidean":
                db_score = davies_bouldin_score(X_valid, labels_valid)
                chi_score = calinski_harabasz_score(X_valid, labels_valid)

                st.success(f"**Davies-Bouldin Score (euclidean):** {db_score:.4f}")
                st.success(f"**Calinski-Harabasz Score (euclidean):** {chi_score:.4f}")
            else:
                st.info(
                    "Davies-Bouldin dan Calinski-Harabasz di sklearn dihitung berbasis Euclidean distance, "
                    "jadi di sini hanya ditampilkan Silhouette Score untuk metric selain Euclidean."
                )

            # Silhouette plot
            st.subheader("Silhouette Plot for DBSCAN (excluding noise)")

            sample_silhouette_values = silhouette_samples(X_valid, labels_valid, metric=distance_metric)

            sil_df = pd.DataFrame({
                "Cluster": labels_valid,
                "silhouette": sample_silhouette_values
            })

            fig_sil, ax1 = plt.subplots(figsize=(8, 4))

            y_lower = 10
            cmap = cm.get_cmap("viridis")

            cluster_ids = sorted(sil_df["Cluster"].unique())
            n_valid_clusters = len(cluster_ids)

            for idx, cluster_id in enumerate(cluster_ids):
                vals = np.asarray(
                    sil_df.loc[sil_df["Cluster"] == cluster_id, "silhouette"]
                )
                cluster_vals = np.sort(vals)

                size_cluster_i = len(cluster_vals)
                y_upper = y_lower + size_cluster_i

                color = cmap(float(idx) / max(n_valid_clusters - 1, 1))

                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    cluster_vals,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster_id))
                y_lower = y_upper + 10

            ax1.set_title(f"Silhouette Plot for DBSCAN Clusters (metric={distance_metric}, Noise Excluded)")
            ax1.set_xlabel("Silhouette Coefficient Values")
            ax1.set_ylabel("Cluster Label")

            ax1.axvline(x=float(sil_score), linestyle="--", linewidth=1)
            ax1.set_yticks([])
            ax1.set_xlim(-0.2, 1.0)

            st.pyplot(fig_sil)

        else:
            st.warning("Not enough clusters found (excluding noise) to calculate evaluation metrics.")

            
        # --- Segmentation profile ---
        avg_aov = clv_df['Average'].mean()
        avg_lifespan = clv_df['Lifespan'].mean()
        avg_frequency = clv_df['Frequency'].mean()
        avg_clv = clv_df['CLV'].mean()

        st.subheader("Average CLV Values")
        global_avg = {
            'Average': clv_df['Average'].mean(),
            'Lifespan': clv_df['Lifespan'].mean(),
            'Frequency': clv_df['Frequency'].mean(),
            'CLV': clv_df['CLV'].mean()
        }

        st.table(pd.DataFrame.from_dict(global_avg, orient='index', columns=['Average']).round(2))

        cluster_avg = clv_df.groupby('Cluster')[['Average', 'Lifespan', 'Frequency', 'CLV']].mean().reset_index()

        # --- Determine if feature is higher
        for col in ['Average', 'Lifespan', 'Frequency', 'CLV']:
            cluster_avg[f'{col}_high'] = cluster_avg[col] > global_avg[col]
        
        
        # Segment mapping
        segment_map = {
            (True,  True,  True,  True):  "High-value loyal customers",
            (False, True,  True,  True):  "High-value new customers",
            (True,  False, True,  True):  "Potential loyal customers",
            (False, False, True,  True):  "High-value lost customers",
            (True,  True,  False, True):  "Platinum customers",
            (False, True,  False, True):  "Consuming promotional customers",
            (True,  False, False, True):  "Potential consuming customers",
            (False, False, False, True):  "Consuming churn customers",
            (True,  True,  True,  False): "High frequency customers",
            (False, True,  True,  False): "Frequency promotional customers",
            (True,  False, True,  False): "Potential frequency customers",
            (False, False, True,  False): "Frequency churn customers",
            (True,  True,  False, False): "Low-cost consuming customers",
            (False, True,  False, False): "Uncertain new customers",
            (True,  False, False, False): "High-cost consuming customers",
            (False, False, False, False): "Uncertain lost customers",
        }

        # Assign segment
        cluster_avg['Segment'] = cluster_avg[['Average_high', 'Lifespan_high', 'Frequency_high', 'CLV_high']].apply(
            lambda x: segment_map[tuple(x)], axis=1
        )
        # Segment
        st.subheader("Cluster Segment Classification")
        st.dataframe(cluster_avg[['Cluster', 'Average', 'Lifespan', 'Frequency', 'CLV', 'Segment']])
        st.bar_chart(clv_df['Cluster'].value_counts())  

        # --- Download result
        st.write("### Download")
        csv = clv_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Clustered Data as CSV", csv, "clustered_customers_dbscan.csv", "text/csv")

    elif method == "DBSCAN" and features == "MLRFM":  
        st.subheader("DBSCAN Clustering using MLRFM")
        max_date = df['tanggal'].max()
        df.rename(columns={"customer_id": "nama_customer"}, inplace=True)
        recency_df = df.groupby('nama_customer')['tanggal'].max().reset_index()
        recency_df['Recency'] = (max_date - recency_df['tanggal']).dt.days
        recency_df = recency_df[['nama_customer', 'Recency']]
        
        #Periods
        periods = {'365d': 365, '730d': 730, 'All': None}

        rfm_list = []

        for label, days in periods.items():
            if days is not None:
                start_date = max_date - pd.Timedelta(days=days)
                df_period = df[df['tanggal'] > start_date]
            else:
                df_period = df.copy()  # All data
                
            frequency = df_period.groupby('nama_customer')['no_invoice'].nunique().reset_index()
            monetary = df_period.groupby('nama_customer')['total_harga'].sum().reset_index()

            rfm = frequency.merge(monetary, on='nama_customer', how='outer')
            rfm = rfm.rename(columns={
                'no_invoice': f'Frequency_{label}',
                'total_harga': f'Monetary_{label}'
            })

            rfm_list.append(rfm)

        mlrfm = reduce(lambda left, right: pd.merge(left, right, on='nama_customer', how='outer'), rfm_list)
        final_mlrfm = recency_df.merge(mlrfm, on='nama_customer', how='outer')

        st.dataframe(final_mlrfm.head(10))    

        final_mlrfm.fillna({
            'Frequency_365d': 0, 'Monetary_365d': 0,
            'Frequency_730d': 0, 'Monetary_730d': 0,
            'Frequency_All': 0, 'Monetary_All': 0,
            'Recency': 999
        }, inplace=True)

        
        st.subheader("Yeo-Johnson Transformation & MinMax Scaling")

        mlrfm_scaled_df = final_mlrfm.copy()
        features_to_scale = ['Recency', 'Frequency_365d', 'Frequency_730d', 'Frequency_All',
                            'Monetary_365d', 'Monetary_730d', 'Monetary_All']

        # Apply Yeo-Johnson
        pt = PowerTransformer(method='yeo-johnson')
        mlrfm_scaled_df[features_to_scale] = pt.fit_transform(mlrfm_scaled_df[features_to_scale])
 
        # Apply MinMaxScaler
        scaler = MinMaxScaler()
        mlrfm_scaled_df[features_to_scale] = scaler.fit_transform(mlrfm_scaled_df[features_to_scale])

        st.dataframe(mlrfm_scaled_df.head(10))  

        st.subheader("Set Weights for Multi-Layer RFM")

        st.markdown("#### Frequency Weights")
        w_f_365 = st.number_input("Weight for Frequency_365d", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        w_f_730 = st.number_input("Weight for Frequency_730d", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        w_f_All = st.number_input("Weight for Frequency_All", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

        st.markdown("#### Monetary Weights")
        w_m_365 = st.number_input("Weight for Monetary_365d", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        w_m_730 = st.number_input("Weight for Monetary_730d", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        w_m_All = st.number_input("Weight for Monetary_All", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

        # Normalize weights
        total_f = w_f_365 + w_f_730 + w_f_All
        w_f_365 /= total_f
        w_f_730 /= total_f
        w_f_All /= total_f

        total_m = w_m_365 + w_m_730 + w_m_All
        w_m_365 /= total_m
        w_m_730 /= total_m
        w_m_All /= total_m

        # Compute combined Frequency & Monetary
        mlrfm_scaled_df['Multi_Layer_Frequency'] = (
            w_f_365 * mlrfm_scaled_df['Frequency_365d'] +
            w_f_730 * mlrfm_scaled_df['Frequency_730d'] +
            w_f_All * mlrfm_scaled_df['Frequency_All']
        )

        mlrfm_scaled_df['Multi_Layer_Monetary'] = (
            w_m_365 * mlrfm_scaled_df['Monetary_365d'] +
            w_m_730 * mlrfm_scaled_df['Monetary_730d'] +
            w_m_All * mlrfm_scaled_df['Monetary_All']
        )
        mlrfm_scaled_df = mlrfm_scaled_df[['nama_customer','Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']]
        st.dataframe(mlrfm_scaled_df.head(10))  

        # # Bins
        # bins = [0.0, 0.10, 0.30, 0.50, 0.70, 1.00]
        # labels = [1, 2, 3, 4, 5]

        # # Note: Recency Reverse
        # mlrfm_scaled_df['Recency'] = pd.cut(mlrfm_scaled_df['Recency'], bins=bins, labels=labels[::-1], include_lowest=True).astype(int) * 0.33
        # mlrfm_scaled_df['Multi_Layer_Frequency'] = pd.cut(mlrfm_scaled_df['Multi_Layer_Frequency'], bins=bins, labels=labels, include_lowest=True).astype(int) * 0.33
        # mlrfm_scaled_df['Multi_Layer_Monetary'] = pd.cut(mlrfm_scaled_df['Multi_Layer_Monetary'], bins=bins, labels=labels, include_lowest=True).astype(int) * 0.33

        # mlrfm_scaled_df['Recency'] = mlrfm_scaled_df['Recency'] *  0.33
        # mlrfm_scaled_df['Multi_Layer_Frequency'] = mlrfm_scaled_df['Multi_Layer_Frequency'] * 0.33
        # mlrfm_scaled_df['Multi_Layer_Monetary'] = mlrfm_scaled_df['Multi_Layer_Monetary'] *0.33

        mlrfm_scaled_df['Recency'] = 1 - mlrfm_scaled_df['Recency'] #command if use bin

        st.subheader("Final MLRFM Data")
        st.dataframe(mlrfm_scaled_df) 

        st.subheader("DBSCAN Clustering")
        X = mlrfm_scaled_df[['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']]

        #--- Pilih Distance Metric ---
        st.markdown("#### Distance Metric")
        metric_label = st.selectbox(
            "Pilih distance metric untuk DBSCAN & Silhouette",
            ["Euclidean", "Manhattan", "Cosine"]
        )

        metric_map = {
            "Euclidean": "euclidean",
            "Manhattan": "manhattan",
            "Cosine": "cosine"
        }
        distance_metric = metric_map[metric_label]

        min_samples = st.slider("Set min_samples", 2, 15, 6)
        epsilon = st.slider("Set epsilon (eps)", min_value=0.01, max_value=2.0, value=0.16, step=0.01)
        # Epsilon Estimation via k-distance graph
        neighbors = NearestNeighbors(
            n_neighbors=min_samples,
            metric=distance_metric
        )
        neighbors_fit = neighbors.fit(X)
        distances, indices = neighbors_fit.kneighbors(X)
        distances = np.sort(distances[:, min_samples - 1])

        fig_kdist, ax = plt.subplots()
        ax.plot(distances)
        ax.set_title("K-Distance Graph (Pick ε from elbow)")
        ax.set_xlabel("Data Points (sorted)")
        ax.set_ylabel(f"{min_samples}-NN Distance")
        st.pyplot(fig_kdist)

        db = DBSCAN(
            eps=epsilon,
            min_samples=min_samples,
            metric=distance_metric
        )
        mlrfm_scaled_df['Cluster'] = db.fit_predict(X)

        st.session_state["mlrfm_cluster_df"] = mlrfm_scaled_df.copy()
        st.session_state["cluster_source"] = "MLRFM"
        st.session_state["mlrfm_method"] = "DBSCAN"


     # --- Evaluation (ignoring noise cluster -1)
        st.write("### Evaluation Metrics")
        valid_clusters = mlrfm_scaled_df[mlrfm_scaled_df['Cluster'] != -1]
        
        if len(valid_clusters['Cluster'].unique()) > 1:

            
            # Silhouette pakai metric terpilih
            sil = silhouette_score(
                valid_clusters[['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']],
                valid_clusters['Cluster'],
                metric=distance_metric
            )
            st.success(f"Silhouette Score (metric = {distance_metric} | excluding noise): {sil:.3f}")

            # DBI & CH hanya valid untuk Euclidean (implementasi sklearn)
            if distance_metric == "euclidean":
                dbi = davies_bouldin_score(
                    valid_clusters[['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']],
                    valid_clusters['Cluster']
                )
                chi = calinski_harabasz_score(
                    valid_clusters[['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']],
                    valid_clusters['Cluster']
                )
                st.success(f"Davies-Bouldin Index (euclidean, excluding noise): {dbi:.3f}")
                st.success(f"Calinski-Harabasz Score (euclidean, excluding noise): {chi:.3f}")
            else:
                st.info("Davies-Bouldin & Calinski-Harabasz di sklearn hanya menggunakan Euclidean distance, "
                        "jadi di sini hanya ditampilkan Silhouette Score.")

            # Konversi ke numpy supaya Pylance tau tipenya jelas
            X_valid = valid_clusters[['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']].to_numpy(dtype=float)
            labels_valid = valid_clusters['Cluster'].to_numpy()

            silhouette_vals = silhouette_samples(
                X_valid,
                labels_valid,
                metric=distance_metric
            )
            silhouette_vals = np.asarray(silhouette_vals, dtype=float)

            unique_labels = np.unique(labels_valid)
            n_clusters_db = int(unique_labels.shape[0])

            fig_sil, ax1 = plt.subplots(figsize=(8, 6))
            y_lower = 10
            cmap = plt.get_cmap("nipy_spectral")

            for idx, c in enumerate(unique_labels):
                mask = (labels_valid == c)
                ith_cluster_sil_vals = silhouette_vals[mask]
                ith_cluster_sil_vals = np.sort(ith_cluster_sil_vals)

                size_cluster_i = int(ith_cluster_sil_vals.shape[0])
                y_upper = y_lower + size_cluster_i

                color = cmap(float(idx) / float(max(n_clusters_db, 1)))

                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0.0,
                    ith_cluster_sil_vals,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7
                )
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(c))
                y_lower = y_upper + 10

            ax1.axvline(x=float(sil), linestyle="--")
            ax1.set_xlabel(f"Silhouette coefficient (metric = {distance_metric})")
            ax1.set_ylabel("Cluster")
            ax1.set_title("Silhouette Plot for DBSCAN Clustering (excluding noise)")
            ax1.set_yticks([])
            ax1.set_xlim(-0.1, 1.0)

            st.pyplot(fig_sil)

        else:
            st.warning("Not enough clusters to compute Silhouette (need >1 cluster excluding noise).")


        st.markdown("### Visualisasi 3D Segmen Pelanggan")

        # Pengatur ukuran marker
        ukuran_marker = st.slider("Ukuran titik", min_value=3, max_value=12, value=6, step=1)

        # Ganti nama kolom agar ringkas (opsional, hanya untuk tampilan)
        df_vis = mlrfm_scaled_df.rename(columns={
            "Recency": "R",
            "Multi_Layer_Frequency": "MLF",
            "Multi_Layer_Monetary": "MLM"
        })

        # Pilih cluster yang ingin ditampilkan
        daftar_cluster = sorted(df_vis["Cluster"].unique().tolist())
        pilihan_cluster = st.multiselect(
            "Tampilkan cluster",
            options=daftar_cluster,
            default=daftar_cluster
        )

        # Bangun figure 3D per-cluster
        fig = go.Figure()
        for c in pilihan_cluster:
            subset = df_vis[df_vis["Cluster"] == c]
            hovertext = (
                "Customer: " + subset["nama_customer"].astype(str) +
                "<br>R: " + subset["R"].round(2).astype(str) +
                "<br>MLF: " + subset["MLF"].round(2).astype(str) +
                "<br>MLM: " + subset["MLM"].round(2).astype(str)
            )

            fig.add_trace(
                go.Scatter3d(
                    x=subset["R"],
                    y=subset["MLF"],
                    z=subset["MLM"],
                    mode="markers",
                    name=f"Cluster {c}",
                    text=hovertext,
                    hovertemplate="%{text}<extra></extra>",
                    marker=dict(size=ukuran_marker, opacity=0.85)
                )
            )

        fig.update_layout(
            title="DBSCAN Clusters (3D)",
            legend_title_text="Cluster",
            scene=dict(
                xaxis_title="Recency (R)",
                yaxis_title="Multi-Layer Frequency (MLF)",
                zaxis_title="Multi-Layer Monetary (MLM)"
            ),
            margin=dict(l=10, r=10, t=60, b=10),
            height=640
        )

        st.plotly_chart(fig, use_container_width=True)

        # Tabel data terklaster (ikut filter cluster pilihan)
        st.subheader("Clustered Data Table")
        df_tabel = mlrfm_scaled_df[
            mlrfm_scaled_df["Cluster"].isin(pilihan_cluster)
        ][['nama_customer', 'Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Cluster']]

        st.dataframe(df_tabel, use_container_width=True)
        
        
   
        # --- Segmentation profile ---
        avg_recency = mlrfm_scaled_df['Recency'].mean()
        avg_frequency = mlrfm_scaled_df['Multi_Layer_Frequency'].mean()
        avg_monetary = mlrfm_scaled_df['Multi_Layer_Monetary'].mean()

        st.subheader("Average MLRFM Values")
        global_avg = {
            'Recency': mlrfm_scaled_df['Recency'].mean(),
            'Multi_Layer_Frequency': mlrfm_scaled_df['Multi_Layer_Frequency'].mean(),
            'Multi_Layer_Monetary': mlrfm_scaled_df['Multi_Layer_Monetary'].mean()
        }

        st.table(pd.DataFrame.from_dict(global_avg, orient='index', columns=['Average']).round(2))

        cluster_avg = mlrfm_scaled_df.groupby('Cluster')[[ 'Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']].mean().reset_index()

        # --- Determine if feature is higher
        for col in ['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']:
            cluster_avg[f'{col}_high'] = cluster_avg[col] > global_avg[col]

        
        # segment_map = {
        # (True,  True,  True):  "Loyal Customers",              # R↑ F↑ M↑ — paling aktif & berbelanja tinggi
        # (True,  True,  False): "Potential loyal customers",    # R↑ F↑ M↓ — sering & baru-baru ini, nilai belanja masih kecil
        # (True,  False, True):  "Promising customers",          # R↑ F↓ M↑ — baru-baru ini, nilai tinggi tapi belum sering
        # (True,  False, False): "New customers",                # R↑ F↓ M↓ — baru beli & masih rendah
        # (False, True,  True):  "At risk (high value)",         # R↓ F↑ M↑ — dulu aktif & bernilai tinggi, sekarang mulai tidak aktif
        # (False, True,  False): "About to sleep",               # R↓ F↑ M↓ — dulu sering tapi nilai kecil, mulai tidak aktif
        # (False, False, True):  "Churned high spenders",        # R↓ F↓ M↑ — pernah belanja besar tapi sudah tidak aktif
        # (False, False, False): "Lost customers",               # R↓ F↓ M↓ — tidak aktif & nilai rendah
        # }

        segment_map = {
            (True,  True,  True):  "Loyal customers",          # R↑ F↑ M↑
            (True,  False, True):  "Promising customers",      # R↑ F↓ M↑
            (True,  False, False): "New customers",            # R↑ F↓ M↓
            (False, False, False): "Lost customers",           # R↓ F↓ M↓
            (False, True,  True):  "Lost customers 'Churned'",           # R↓ F↑ M↑ 
            (False, False, True): "Lost Customers 'High Value'"           # R↓ F↓ M↑ 
        }
         

        # Assign segment
        cluster_avg['Segment'] = cluster_avg[['Recency_high', 'Multi_Layer_Frequency_high', 'Multi_Layer_Monetary_high']].apply(
            lambda x: segment_map[tuple(x)], axis=1
        )
        # Segment
        st.subheader("Cluster Segment Classification")
        st.dataframe(cluster_avg[['Cluster', 'Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Segment']])
        st.bar_chart(mlrfm_scaled_df['Cluster'].value_counts())

        # --- Download ---
        csv = mlrfm_scaled_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Clustered Data as CSV", csv, "mlrfm_clustered.csv", "text/csv")

    

    elif method == "GMM" and features == "CLV":
        st.subheader("GMM Clustering using CLV")

        # --- Agregasi CLV (sama seperti bagian K-Means CLV) ---
        latest_date = df["tanggal"].max()

        customer_df = (
            df.groupby("customer_id")
            .agg(
                Lifespan=("tanggal", lambda x: (latest_date - x.min()).days + 1),
                Frequency=("qty", "count"),
                Total_Value=("total_harga", "sum"),
            )
            .reset_index()
        )

        customer_df["Average"] = customer_df["Total_Value"] / customer_df["Frequency"]
        profit_margin = 0.08
        customer_df["CLV"] = (
            customer_df["Average"] * profit_margin * customer_df["Frequency"] * customer_df["Lifespan"]
        )

        st.dataframe(customer_df[["customer_id", "Average", "Lifespan", "Frequency", "CLV"]].head(10))

        st.subheader("Yeo-Johnson Transformation & MinMax Scaling")

        clv = customer_df[['Average', 'Lifespan', 'Frequency', 'CLV']]
        pt = PowerTransformer(method='yeo-johnson')
        clv_transformed = pt.fit_transform(clv)
        scaler = MinMaxScaler()
        clv_scaled = scaler.fit_transform(clv_transformed)
        X = clv_scaled

        clv_df = pd.DataFrame(
            np.column_stack((customer_df['customer_id'].values, clv_scaled)),
            columns=['customer_id', 'Average', 'Lifespan', 'Frequency', 'CLV']
        )

        st.write("Data for GMM Clustering")
        st.dataframe(clv_df.head(10))

        # --- Cari jumlah komponen pakai BIC ---
        st.subheader("BIC untuk Menentukan Jumlah Komponen GMM")

        max_comp = st.slider(
            "Maximum number of components to test (BIC)",
            min_value=2,
            max_value=15,
            value=8
        )

        components_range = range(2, max_comp + 1)
        bics = []

        for k in components_range:
            gmm_tmp = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=42,
                n_init=5,
            )
            gmm_tmp.fit(X)
            bics.append(gmm_tmp.bic(X))

        fig_bic, ax = plt.subplots()
        ax.plot(list(components_range), bics, marker="o")
        ax.set_xlabel("Number of components")
        ax.set_ylabel("BIC")
        ax.set_title("BIC vs Number of Components (GMM)")
        st.pyplot(fig_bic)

        st.write(
            "Nilai **BIC lebih kecil** biasanya lebih baik. "
            "Pilih jumlah komponen yang menurut kamu paling pas."
        )

        # --- Fit GMM final ---
        st.subheader("GMM Clustering")
        n_components = st.slider("Choose number of clusters/components", 2, 10, 3)

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=42,
            n_init=10,
        )
        labels = gmm.fit_predict(X)

        clv_df["Cluster"] = labels.astype(int)

        st.session_state["clv_cluster_df"] = clv_df.copy()
        st.session_state["cluster_source"] = "CLV"
        st.session_state["clv_method"] = "GMM"

        st.subheader("Clustered Customer Table")
        st.dataframe(clv_df)

        # --- Evaluation metrics ---
        sil_score = silhouette_score(X, labels)
        db_score = davies_bouldin_score(X, labels)
        chi_score = calinski_harabasz_score(X, labels)

        st.subheader("Evaluation Metrics for GMM (CLV)")
        st.success(f"**Silhouette Score:** {sil_score:.3f}")
        st.success(f"**Davies-Bouldin Score:** {db_score:.3f}")
        st.success(f"**Calinski-Harabasz Score:** {chi_score:3f}")

        # --- Silhouette Plot ---
        st.subheader("Silhouette Plot for GMM (CLV)")

        sample_silhouette_values = silhouette_samples(X, labels)
        sil_df = pd.DataFrame({
            "Cluster": labels.astype(int),
            "silhouette": sample_silhouette_values
        })

        fig_sil, ax1 = plt.subplots(figsize=(8, 4))
        y_lower = 10
        cmap = cm.get_cmap("viridis")

        for i in range(n_components):
            cluster_vals = (
                sil_df.loc[sil_df["Cluster"] == i, "silhouette"]
                .sort_values()
                .to_numpy()
            )
            size_cluster_i = len(cluster_vals)
            y_upper = y_lower + size_cluster_i

            color = cmap(float(i) / max(n_components - 1, 1))
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_vals,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        ax1.set_title("Silhouette Plot for GMM (CLV)")
        ax1.set_xlabel("Silhouette Coefficient Values")
        ax1.set_ylabel("Cluster Label")
        ax1.axvline(x=float(sil_score), linestyle="--", linewidth=1)
        ax1.set_yticks([])
        ax1.set_xlim(-0.2, 1.0)
        st.pyplot(fig_sil)

        # --- 3D Visualisation ---
        st.markdown("### Visualisasi Klaster (3D)")

        fitur = ["Average", "Lifespan", "Frequency", "CLV"]
        tiga_dim = list(combinations(fitur, 3))
        opsi_kombinasi = {f"{a} · {b} · {c}": (a, b, c) for a, b, c in tiga_dim}

        label_terpilih = st.selectbox("Pilih 3 fitur untuk scatter 3D (GMM CLV)", list(opsi_kombinasi.keys()))
        sx, sy, sz = opsi_kombinasi[label_terpilih]

        fig3d = px.scatter_3d(
            data_frame=clv_df,
            x=sx,
            y=sy,
            z=sz,
            color="Cluster",
            hover_data={col: True for col in clv_df.columns},
            title=f"GMM Scatter 3D | {sx} × {sy} × {sz}",
            height=620,
        )
        fig3d.update_traces(marker=dict(size=4, opacity=0.85))
        fig3d.update_layout(
            legend_title_text="Cluster",
            scene=dict(
                xaxis_title=sx,
                yaxis_title=sy,
                zaxis_title=sz
            ),
            margin=dict(l=10, r=10, t=60, b=10)
        )
        st.plotly_chart(fig3d, use_container_width=True)

        st.subheader("Average CLV Values (Global)")

        global_avg = {
            'Average':  clv_df['Average'].mean(),
            'Lifespan': clv_df['Lifespan'].mean(),
            'Frequency': clv_df['Frequency'].mean(),
            'CLV':      clv_df['CLV'].mean(),
        }
        st.table(pd.DataFrame.from_dict(global_avg, orient='index', columns=['Value']).round(2))

        cluster_avg = clv_df.groupby('Cluster')[['Average', 'Lifespan', 'Frequency', 'CLV']].mean().reset_index()

        for col in ['Average', 'Lifespan', 'Frequency', 'CLV']:
            cluster_avg[f'{col}_high'] = cluster_avg[col] > global_avg[col]

        segment_map = {
            (True,  True,  True,  True):  "High-value loyal customers",
            (False, True,  True,  True):  "High-value new customers",
            (True,  False, True,  True):  "Potential loyal customers",
            (False, False, True,  True):  "High-value lost customers",
            (True,  True,  False, True):  "Platinum customers",
            (False, True,  False, True):  "Consuming promotional customers",
            (True,  False, False, True):  "Potential consuming customers",
            (False, False, False, True):  "Consuming churn customers",
            (True,  True,  True,  False): "High frequency customers",
            (False, True,  True,  False): "Frequency promotional customers",
            (True,  False, True,  False): "Potential frequency customers",
            (False, False, True,  False): "Frequency churn customers",
            (True,  True,  False, False): "Low-cost consuming customers",
            (False, True,  False, False): "Uncertain new customers",
            (True,  False, False, False): "High-cost consuming customers",
            (False, False, False, False): "Uncertain lost customers",
        }

        cluster_avg['Segment'] = cluster_avg[
            ['Average_high', 'Lifespan_high', 'Frequency_high', 'CLV_high']
        ].apply(lambda x: segment_map[tuple(x)], axis=1)

        st.subheader("Cluster Segment Classification (GMM CLV)")
        st.dataframe(cluster_avg[['Cluster', 'Average', 'Lifespan', 'Frequency', 'CLV', 'Segment']])
        st.bar_chart(clv_df['Cluster'].value_counts())

        # ======================
        # 9. Download hasil
        # ======================
        csv = clv_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download GMM CLV Clustered Data as CSV",
            csv,
            "GMM_CLV_clustered_customers.csv",
            "text/csv"
        )

    elif method == "GMM" and features == "MLRFM":
        st.subheader("GMM Clustering using MLRFM")

        # --- Hitung MLRFM (sama konsep dengan bagian K-Means MLRFM) ---
        max_date = df['tanggal'].max()
        df.rename(columns={"customer_id": "nama_customer"}, inplace=True)

        recency_df = df.groupby('nama_customer')['tanggal'].max().reset_index()
        recency_df['Recency'] = (max_date - recency_df['tanggal']).dt.days
        recency_df = recency_df[['nama_customer', 'Recency']]

        periods = {'365d': 365, '730d': 730, 'All': None}
        rfm_list = []

        for label, days in periods.items():
            if days is not None:
                start_date = max_date - pd.Timedelta(days=days)
                df_period = df[df['tanggal'] > start_date]
            else:
                df_period = df.copy()

            frequency = df_period.groupby('nama_customer')['no_invoice'].nunique().reset_index()
            monetary = df_period.groupby('nama_customer')['total_harga'].sum().reset_index()

            rfm = frequency.merge(monetary, on='nama_customer', how='outer')
            rfm = rfm.rename(columns={
                'no_invoice': f'Frequency_{label}',
                'total_harga': f'Monetary_{label}'
            })
            rfm_list.append(rfm)

        mlrfm = reduce(lambda left, right: pd.merge(left, right, on='nama_customer', how='outer'), rfm_list)
        final_mlrfm = recency_df.merge(mlrfm, on='nama_customer', how='outer')

        final_mlrfm.fillna({
            'Frequency_365d': 0, 'Monetary_365d': 0,
            'Frequency_730d': 0, 'Monetary_730d': 0,
            'Frequency_All': 0, 'Monetary_All': 0,
            'Recency': 999
        }, inplace=True)

        st.dataframe(final_mlrfm.head(10))

        st.subheader("Yeo-Johnson Transformation & MinMax Scaling")

        mlrfm_scaled_df = final_mlrfm.copy()
        features_to_scale = [
            'Recency',
            'Frequency_365d', 'Frequency_730d', 'Frequency_All',
            'Monetary_365d', 'Monetary_730d', 'Monetary_All'
        ]

        pt = PowerTransformer(method='yeo-johnson')
        mlrfm_scaled_df[features_to_scale] = pt.fit_transform(mlrfm_scaled_df[features_to_scale])

        scaler = MinMaxScaler()
        mlrfm_scaled_df[features_to_scale] = scaler.fit_transform(mlrfm_scaled_df[features_to_scale])

        st.dataframe(mlrfm_scaled_df.head(10))

        st.subheader("Set Weights for Multi-Layer RFM")

        st.markdown("#### Frequency Weights")
        w_f_365 = st.number_input("Weight for Frequency_365d", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        w_f_730 = st.number_input("Weight for Frequency_730d", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        w_f_All = st.number_input("Weight for Frequency_All", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

        st.markdown("#### Monetary Weights")
        w_m_365 = st.number_input("Weight for Monetary_365d", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        w_m_730 = st.number_input("Weight for Monetary_730d", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        w_m_All = st.number_input("Weight for Monetary_All", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

        # normalisasi bobot
        total_f = w_f_365 + w_f_730 + w_f_All
        w_f_365 /= total_f
        w_f_730 /= total_f
        w_f_All /= total_f

        total_m = w_m_365 + w_m_730 + w_m_All
        w_m_365 /= total_m
        w_m_730 /= total_m
        w_m_All /= total_m

        mlrfm_scaled_df['Multi_Layer_Frequency'] = (
            w_f_365 * mlrfm_scaled_df['Frequency_365d'] +
            w_f_730 * mlrfm_scaled_df['Frequency_730d'] +
            w_f_All * mlrfm_scaled_df['Frequency_All']
        )

        mlrfm_scaled_df['Multi_Layer_Monetary'] = (
            w_m_365 * mlrfm_scaled_df['Monetary_365d'] +
            w_m_730 * mlrfm_scaled_df['Monetary_730d'] +
            w_m_All * mlrfm_scaled_df['Monetary_All']
        )

        mlrfm_scaled_df = mlrfm_scaled_df[['nama_customer', 'Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']]
        st.dataframe(mlrfm_scaled_df.head(10))

        # Recency dibalik (semakin kecil = makin lama)
        mlrfm_scaled_df['Recency'] = 1 - mlrfm_scaled_df['Recency']

        st.subheader("Final MLRFM Data")
        st.dataframe(mlrfm_scaled_df)

        # --- GMM BIC ---
        X_feats = mlrfm_scaled_df[['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']].values

        st.subheader("BIC untuk Menentukan Jumlah Komponen (GMM MLRFM)")
        max_comp = st.slider(
            "Maximum number of components to test (BIC) - MLRFM",
            min_value=2,
            max_value=15,
            value=8
        )

        components_range = range(2, max_comp + 1)
        bics = []

        for k in components_range:
            gmm_tmp = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=42,
                n_init=5,
            )
            gmm_tmp.fit(X_feats)
            bics.append(gmm_tmp.bic(X_feats))

        fig_bic, ax = plt.subplots()
        ax.plot(list(components_range), bics, marker="o")
        ax.set_xlabel("Number of components")
        ax.set_ylabel("BIC")
        ax.set_title("BIC vs Number of Components (GMM MLRFM)")
        st.pyplot(fig_bic)

        st.write("Pilih jumlah komponen berdasarkan BIC (semakin kecil semakin baik).")

        # --- GMM final ---
        st.subheader("GMM Clustering (MLRFM)")
        n_components = st.slider("Choose number of clusters/components (MLRFM)", 2, 10, 5)

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=42,
            n_init=10,
        )
        labels = gmm.fit_predict(X_feats)

        mlrfm_scaled_df['Cluster'] = labels.astype(int)

        st.session_state["mlrfm_cluster_df"] = mlrfm_scaled_df.copy()
        st.session_state["cluster_source"] = "MLRFM"
        st.session_state["mlrfm_method"] = "GMM"

        st.subheader("Clustered Data Table (MLRFM GMM)")
        st.dataframe(mlrfm_scaled_df[['nama_customer', 'Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Cluster']])

        X_eval = mlrfm_scaled_df[['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Cluster']].to_numpy(dtype=float)
        y_labels = mlrfm_scaled_df['Cluster'].to_numpy()

        # --- Metrics ---
        sil_score = silhouette_score(X_eval, y_labels)
        db_score = davies_bouldin_score(X_eval, y_labels)
        chi_score = calinski_harabasz_score(X_eval, y_labels)
        
        st.subheader("Evaluation Metrics for GMM (MLRFM)")
        st.success(f"**Silhouette Score:** {sil_score:.3f}")
        st.success(f"**Davies-Bouldin Score:** {db_score:.3f}")
        st.success(f"**Calinski-Harabasz Score:** {chi_score:3f}")

        # --- Silhouette plot ---
        st.subheader("Silhouette Plot for GMM (MLRFM)")

        silhouette_vals = silhouette_samples(X_eval, y_labels)
        sil_df = pd.DataFrame({
            "Cluster": labels.astype(int),
            "silhouette": silhouette_vals
        })

        fig_sil, ax1 = plt.subplots(figsize=(8, 6))
        y_lower = 10
        cmap = plt.get_cmap("nipy_spectral")

        for i in range(n_components):
            vals = sil_df.loc[sil_df["Cluster"] == i, "silhouette"].values
            vals = np.sort(vals)
            size_cluster_i = len(vals)
            y_upper = y_lower + size_cluster_i

            color = cmap(float(i) / float(max(n_components - 1, 1)))
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0.0,
                vals,
                facecolor=color,
                edgecolor=color,
                alpha=0.7
            )
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        ax1.axvline(x=float(sil_score), linestyle="--")
        ax1.set_xlabel("Silhouette coefficient")
        ax1.set_ylabel("Cluster")
        ax1.set_title("Silhouette Plot for GMM (MLRFM)")
        ax1.set_yticks([])
        ax1.set_xlim(-0.1, 1.0)
        st.pyplot(fig_sil)

        # --- 3D Visualisation ---
        st.markdown("### Visualisasi 3D Segmen Pelanggan (GMM MLRFM)")

        ukuran_marker = st.slider("Ukuran titik (GMM MLRFM)", 3, 12, 6, 1)

        df_vis = mlrfm_scaled_df.rename(columns={
            "Recency": "R",
            "Multi_Layer_Frequency": "MLF",
            "Multi_Layer_Monetary": "MLM"
        })

        daftar_cluster = sorted(df_vis["Cluster"].unique().tolist())
        pilihan_cluster = st.multiselect(
            "Tampilkan cluster (GMM MLRFM)",
            options=daftar_cluster,
            default=daftar_cluster
        )

        fig3d = go.Figure()
        for c in pilihan_cluster:
            subset = df_vis[df_vis["Cluster"] == c]
            hovertext = (
                "Customer: " + subset["nama_customer"].astype(str) +
                "<br>R: " + subset["R"].round(2).astype(str) +
                "<br>MLF: " + subset["MLF"].round(2).astype(str) +
                "<br>MLM: " + subset["MLM"].round(2).astype(str)
            )

            fig3d.add_trace(
                go.Scatter3d(
                    x=subset["R"],
                    y=subset["MLF"],
                    z=subset["MLM"],
                    mode="markers",
                    name=f"Cluster {c}",
                    text=hovertext,
                    hovertemplate="%{text}<extra></extra>",
                    marker=dict(size=ukuran_marker, opacity=0.85)
                )
            )

        fig3d.update_layout(
            title="GMM Customer Segmentation (MLRFM)",
            legend_title_text="Cluster",
            scene=dict(
                xaxis_title="Recency (R)",
                yaxis_title="Multi-Layer Frequency (MLF)",
                zaxis_title="Multi-Layer Monetary (MLM)"
            ),
            margin=dict(l=10, r=10, t=60, b=10),
            height=640
        )
        st.plotly_chart(fig3d, use_container_width=True)

        # ========================
        # 9. Segment Profiling (optional, mirip K-Means MLRFM)
        # ========================
        st.subheader("Average MLRFM Values")
        global_avg = {
            'Recency': mlrfm_scaled_df['Recency'].mean(),
            'Multi_Layer_Frequency': mlrfm_scaled_df['Multi_Layer_Frequency'].mean(),
            'Multi_Layer_Monetary': mlrfm_scaled_df['Multi_Layer_Monetary'].mean()
        }
        st.table(pd.DataFrame.from_dict(global_avg, orient='index', columns=['Average']).round(2))

        cluster_avg = mlrfm_scaled_df.groupby('Cluster')[
            ['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']
        ].mean().reset_index()

        for col in ['Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary']:
            cluster_avg[f'{col}_high'] = cluster_avg[col] > global_avg[col]

        segment_map = {
            (True,  True,  True):  "Loyal customers",          # R↑ F↑ M↑
            (True,  False, True):  "Promising customers",      # R↑ F↓ M↑
            (True,  False, False): "New customers",            # R↑ F↓ M↓
            (False, False, False): "Lost customers",           # R↓ F↓ M↓
            (False, True,  True):  "Lost customers 'Churned'", # R↓ F↑ M↑
            (False, False, True):  "Lost customers 'High Value'"  # R↓ F↓ M↑
        }

        cluster_avg['Segment'] = cluster_avg[
            ['Recency_high', 'Multi_Layer_Frequency_high', 'Multi_Layer_Monetary_high']
        ].apply(lambda x: segment_map.get(tuple(x), "Others"), axis=1)

        st.subheader("Cluster Segment Classification")
        st.dataframe(cluster_avg[['Cluster', 'Recency', 'Multi_Layer_Frequency', 'Multi_Layer_Monetary', 'Segment']])
        st.bar_chart(mlrfm_scaled_df['Cluster'].value_counts())

        # ========================
        # 10. Download hasil
        # ========================
        

        # --- Download ---
        csv = mlrfm_scaled_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download GMM MLRFM Clustered CSV", csv, "GMM_MLRFM_clustered.csv", "text/csv")



        
#Segmentation Analysis
                # ===============================
        # --- Segment Profiling (Universal)
        # ===============================
        # ===============================
# --- Segment Profiling (fixed) ---
# ===============================
    # st.title("Segment Profiling")

    # # --- Resolver universal: cari tabel berisi ['customer_id','Cluster'] ---
    # def resolve_cluster_source() -> pd.DataFrame | None:
    #     for name in ('clv_df', 'mlrfm_scaled_df'):
    #         obj = globals().get(name, None)
    #         if isinstance(obj, pd.DataFrame) and 'Cluster' in obj.columns:
    #             tmp = obj.copy()
    #             if 'customer_id' not in tmp.columns and 'nama_customer' in tmp.columns:
    #                 tmp = tmp.rename(columns={'nama_customer': 'customer_id'})
    #             if {'customer_id','Cluster'}.issubset(tmp.columns):
    #                 return tmp[['customer_id','Cluster']].dropna(subset=['Cluster'])
    #     return None

    # # versi aman untuk rata-rata hari antar transaksi (tanpa .days, tanpa pembagian timedelta)
    # def avg_days_between(dates: pd.Series):
    #     dates = pd.to_datetime(dates, errors='coerce').dropna().sort_values()
    #     if len(dates) < 2:
    #         return np.nan
    #     diffs = dates.diff().dropna()
    #     # konversi Timedelta -> hari (float)
    #     return float((diffs.dt.total_seconds() / 86400).mean())

    # # siapkan df transaksi
    # seg_df = df.copy()
    # if 'customer_id' not in seg_df.columns and 'nama_customer' in seg_df.columns:
    #     seg_df = seg_df.rename(columns={'nama_customer': 'customer_id'})

    # # AMBIL label cluster SEKALI SAJA dari resolver (JANGAN ditimpa lagi)
    # cluster_source = resolve_cluster_source()
    # if cluster_source is None:
    #     st.error("Label cluster belum tersedia. Jalankan salah satu proses clustering terlebih dahulu.")
    #     st.stop()

    # # merge label ke transaksi
    # df_trans = seg_df.merge(cluster_source, on='customer_id', how='left')

    # # turunan waktu
    # df_trans['year'] = df_trans['tanggal'].dt.year
    # df_trans['month'] = df_trans['tanggal'].dt.month
    # df_trans['day'] = df_trans['tanggal'].dt.date
    # df_trans['day_of_week'] = df_trans['tanggal'].dt.day_name()

    # # pilih cluster
    # listed_clusters = sorted([c for c in df_trans['Cluster'].dropna().unique()])
    # if len(listed_clusters) == 0:
    #     st.warning("Belum ada label cluster pada data transaksi.")
    #     st.stop()
    # selected = st.selectbox("Select Cluster", listed_clusters)

    # filtered_df = df_trans[df_trans['Cluster'] == selected].copy()
    # st.subheader(f"Customer Analysis — Cluster {selected}")

    # # KPI ringkas
    # total_pelanggan = filtered_df['customer_id'].nunique()
    # total_invoice = filtered_df['no_invoice'].nunique() if 'no_invoice' in filtered_df.columns else filtered_df.groupby(['customer_id', 'tanggal']).ngroups
    # total_revenue = float(filtered_df['total_harga'].sum()) if 'total_harga' in filtered_df.columns else 0.0
    # aov_cluster = (filtered_df.groupby('no_invoice')['total_harga'].sum().mean()
    #             if 'no_invoice' in filtered_df.columns else None)
    # avg_qty_per_tx = (filtered_df.groupby('no_invoice')['qty'].sum().mean()
    #                 if 'no_invoice' in filtered_df.columns else None)

    # col1, col2, col3 = st.columns(3)
    # with col1:
    #     st.metric("👥 Customers", value=f"{total_pelanggan:,}")
    # with col2:
    #     st.metric("🧾 Transactions", value=f"{total_invoice:,}")
    # with col3:
    #     st.metric("💵 Revenue (sum)", value=f"Rp {total_revenue:,.0f}")
    # if aov_cluster is not None:
    #     st.metric("🛒 AOV (avg revenue/transaction)", value=f"Rp {aov_cluster:,.0f}")
    # if avg_qty_per_tx is not None:
    #     st.metric("📦 Basket Size (avg qty/tx)", value=f"{avg_qty_per_tx:.2f} item")

    # # Preferensi barang
    # if 'nama_barang' in filtered_df.columns:
    #     st.subheader("Preferensi Barang yang Dibeli")
    #     preferensi_barang = (
    #         filtered_df.groupby('nama_barang')
    #         .agg(jumlah_beli=('qty','sum'),
    #             transaksi=('no_invoice','nunique'),
    #             revenue=('total_harga','sum'))
    #         .reset_index()
    #         .sort_values(['revenue','jumlah_beli'], ascending=False)
    #     )
    #     st.dataframe(preferensi_barang, use_container_width=True)

    # # 💰/📦 & interval
    # if {'no_invoice','total_harga'}.issubset(filtered_df.columns):
    #     uang_per_transaksi = filtered_df.groupby(['customer_id','no_invoice'])['total_harga'].sum().reset_index()
    #     avg_uang_per_customer = uang_per_transaksi.groupby('customer_id')['total_harga'].mean().mean()

    #     produk_per_transaksi = filtered_df.groupby(['customer_id','no_invoice'])['qty'].sum().reset_index()
    #     avg_qty_per_customer = produk_per_transaksi.groupby('customer_id')['qty'].mean().mean()

    #     tanggal_per_trans = (
    #         filtered_df.sort_values('tanggal')
    #         .drop_duplicates(subset=['customer_id','no_invoice'])[['customer_id','tanggal']]
    #     )
    #     avg_rentang_per_customer = (
    #         tanggal_per_trans.groupby('customer_id')['tanggal']
    #         .apply(avg_days_between)
    #         .dropna()
    #         .mean()
    #     )

    #     st.markdown(f"**💰 Average Money Spent per Transaction per Customer:** Rp {avg_uang_per_customer:,.0f}")
    #     st.markdown(f"**📦 Average Product Sold per Transaction per Customer:** {avg_qty_per_customer:.2f} item")
    #     if pd.notnull(avg_rentang_per_customer):
    #         st.markdown(f"**⏱️ Average Days Between Each Transaction per Customer:** {avg_rentang_per_customer:.2f} hari")

    # # 📆 per tahun
    # st.subheader("📆 Total Transaction per Year")
    # tahun_df = filtered_df.groupby('year')['no_invoice'].nunique().reset_index(name='jumlah_transaksi')
    # st.dataframe(tahun_df, use_container_width=True)
    # st.bar_chart(tahun_df.set_index('year'))

    # # 📅 per bulan (all years)
    # st.subheader("📅 Total Transaction for Each Month (All Years)")
    # bulan_df = filtered_df.groupby('month')['no_invoice'].nunique().reset_index(name='jumlah_transaksi')
    # bulan_df['month_name'] = bulan_df['month'].apply(lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))
    # bulan_order = ['January','February','March','April','May','June','July','August','September','October','November','December']
    # bulan_df['month_name'] = pd.Categorical(bulan_df['month_name'], categories=bulan_order, ordered=True)
    # bulan_df = bulan_df.sort_values('month_name')
    # st.dataframe(bulan_df[['month_name','jumlah_transaksi']].rename(columns={'month_name':'Bulan'}), use_container_width=True)
    # st.bar_chart(bulan_df.set_index('month_name')['jumlah_transaksi'])

    # # 📆 per hari (Mon–Sun)
    # st.subheader("📅 Total Transaction per Day (Monday - Sunday)")
    # hari_df = filtered_df.groupby('day_of_week')['no_invoice'].nunique().reset_index(name='jumlah_transaksi')
    # order_hari = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    # hari_df['day_of_week'] = pd.Categorical(hari_df['day_of_week'], categories=order_hari, ordered=True)
    # hari_df = hari_df.sort_values('day_of_week')
    # hari_df['Hari'] = hari_df['day_of_week']
    # st.dataframe(hari_df[['Hari','jumlah_transaksi']], use_container_width=True)
    # st.bar_chart(hari_df.set_index('Hari')['jumlah_transaksi'])

    # # ⏳ Distribusi Recency/Lifespan (pakai sumber yang benar)
    # st.subheader("⏳ Distribusi Waktu (Recency/Lifespan)")
    # if features == "MLRFM" and 'mlrfm_scaled_df' in globals() and isinstance(mlrfm_scaled_df, pd.DataFrame):
    #     tmp = mlrfm_scaled_df.copy()
    #     if 'customer_id' not in tmp.columns and 'nama_customer' in tmp.columns:
    #         tmp = tmp.rename(columns={'nama_customer':'customer_id'})
    #     rec_map = tmp[tmp['customer_id'].isin(filtered_df['customer_id'].unique())]
    #     if 'Recency' in rec_map.columns:
    #         fig_r, ax_r = plt.subplots()
    #         ax_r.hist(rec_map['Recency'].dropna(), bins=20)
    #         ax_r.set_xlabel("Recency (scaled/bin)")
    #         ax_r.set_ylabel("Count")
    #         st.pyplot(fig_r)
    # elif features == "CLV" and 'Lifespan' in filtered_df.columns:
    #     fig_l, ax_l = plt.subplots()
    #     ax_l.hist(filtered_df['Lifespan'].dropna(), bins=20)
    #     ax_l.set_xlabel("Lifespan (days)")
    #     ax_l.set_ylabel("Count")
    #     st.pyplot(fig_l)

    # st.info("Tip: gunakan tabel di atas untuk membuat kampanye yang disesuaikan per cluster (promosi, retensi, win-back, dll).")

