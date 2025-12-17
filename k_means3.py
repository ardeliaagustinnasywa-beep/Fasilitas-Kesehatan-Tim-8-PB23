import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(layout="wide", page_title="Analisis Faskes 2022-2024")

# --- 2. FUNGSI LOAD DATA ---
@st.cache_data
def load_and_combine_data():
    files = {2022: "fasilitas2022.csv", 2023: "fasilitas2023.csv", 2024: "fasilitas2024.csv"}
    combined_list = []
    
    for year, path in files.items():
        try:
            # Deteksi pemisah otomatis dan bersihkan nama kolom
            df = pd.read_csv(path, sep=None, engine='python')
            df.columns = df.columns.str.strip().str.upper()
            df['TAHUN'] = year
            combined_list.append(df)
        except Exception as e:
            st.error(f"Gagal memuat {path}: {e}")
            
    if not combined_list:
        return pd.DataFrame()
    return pd.concat(combined_list, ignore_index=True)

# --- 3. ALUR UTAMA ---
def main():
    st.title("Dashboard Analisis Cluster Fasilitas Kesehatan Kab/Kota di Jawa Barat (2022-2024)")
    st.markdown("Pengelompokkan wilayah berdasarkan ketersediaan sarana kesehatan menggunakan algoritma K-Means.")

    df = load_and_combine_data()
    if df.empty:
        st.warning("Data tidak tersedia. Pastikan file CSV sudah benar.")
        st.stop()

    # Variabel fitur yang digunakan
    features = ['JUMLAH RUMAH SAKIT', 'JUMLAH PUSKESMAS', 'JUMLAH POSYANDU', 'JUMLAH APOTEK', 'JUMLAH KLINIK']
    available_features = [f for f in features if f in df.columns]

    # --- PRE-PROCESSING ---
    # Mengisi nilai kosong dengan 0 dan Standarisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[available_features].fillna(0))

    # --- SIDEBAR ---
    st.sidebar.header("Pengaturan Model")
    
    # Menampilkan Metode Elbow sebagai panduan
    st.sidebar.subheader("1. Analisis Elbow")
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    fig_elbow, ax = plt.subplots(figsize=(5,4))
    ax.plot(range(1, 11), wcss, marker='o', color='#1f77b4')
    ax.set_title("Metode Elbow")
    ax.set_xlabel("Jumlah Cluster")
    ax.set_ylabel("WCSS")
    st.sidebar.pyplot(fig_elbow)
    
    # Pilihan Cluster dibatasi 3 dan 4
    st.sidebar.subheader("2. Pilih Jumlah Cluster")
    k_selected = st.sidebar.radio("Gunakan pengaturan untuk memilih:", options=[3, 4], index=0)

    # --- EKSEKUSI K-MEANS ---
    model = KMeans(n_clusters=k_selected, init='k-means++', random_state=42, n_init=10)
    df['CLUSTER'] = model.fit_predict(X_scaled)

    # --- Tampilan Tab ---
    tab_pca, tab_perbandingan, tab_data = st.tabs(["Visualisasi PCA", "Perbandingan Tahunan", "Data Keseluruhan"])

    # --- TAB 1: VISUALISASI PCA ---
    with tab_pca:
        st.subheader(f"Peta Sebaran Cluster (K={k_selected})")
        st.write("Menggunakan PCA untuk memetakan 5 fitur kesehatan ke dalam bidang 2 dimensi.")
        
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(X_scaled)
        df_pca = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
        df_pca['CLUSTER'] = df['CLUSTER'].astype(str)
        df_pca['KABUPATEN'] = df['NAMA KABUPATEN']
        df_pca['TAHUN'] = df['TAHUN'].astype(str)

        fig_scatter = px.scatter(df_pca, x='PC1', y='PC2', color='CLUSTER', 
                                 symbol='TAHUN', hover_name='KABUPATEN',
                                 color_discrete_sequence=px.colors.qualitative.Set1,
                                 title=f"Visualisasi PCA Cluster {k_selected}")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Tampilkan Karakteristik Cluster
        st.subheader("Karakteristik Fasilitas per Cluster (Rata-rata)")
        centroid = df.groupby('CLUSTER')[available_features].mean()
        st.dataframe(centroid.style.background_gradient(cmap='Blues').format("{:.2f}"))

    # --- TAB 2: PERBANDINGAN SETIAP TAHUN ---
    with tab_perbandingan:
        st.subheader("Perubahan Status Cluster Kabupaten/Kota")
        st.write("Tabel di bawah menunjukkan perpindahan cluster setiap tahunnya. Baris berwarna biru menunjukkan adanya perubahan status fasilitas.")
        
        # Pivot table untuk melihat perbandingan tahunan
        pivot_df = df.pivot(index='NAMA KABUPATEN', columns='TAHUN', values='CLUSTER')
        
        def highlight_changes(row):
            # Highlight jika nilai antar tahun tidak sama (ada perpindahan cluster)
            color = 'background-color: #000080' if row.nunique() > 1 else ''
            return [color] * len(row)

        st.dataframe(pivot_df.style.apply(highlight_changes, axis=1), use_container_width=True)
        
        # Grafik Tren Anggota Cluster
        st.subheader("Tren Jumlah Anggota per Cluster")
        trend_data = df.groupby(['TAHUN', 'CLUSTER']).size().reset_index(name='JUMLAH_WILAYAH')
        trend_data['CLUSTER'] = trend_data['CLUSTER'].astype(str)
        fig_trend = px.line(trend_data, x='TAHUN', y='JUMLAH_WILAYAH', color='CLUSTER', markers=True,
                            title="Pertumbuhan/Penurunan Jumlah Anggota Cluster per Tahun")
        st.plotly_chart(fig_trend, use_container_width=True)

    # --- TAB 3: DATA KESELURUHAN ---
    with tab_data:
        st.subheader("Semua Data Fasilitas Kesehatan (2022-2024)")
        # Tambahkan filter tahun di tabel data
        year_filter = st.multiselect("Filter Tahun:", options=[2022, 2023, 2024], default=[2022, 2023, 2024])
        filtered_df = df[df['TAHUN'].isin(year_filter)]
        
        st.dataframe(filtered_df[['TAHUN', 'NAMA KABUPATEN', 'CLUSTER'] + available_features].sort_values(['TAHUN', 'NAMA KABUPATEN']),
                     use_container_width=True)
        
        # Download Button
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data Hasil Cluster (CSV)", data=csv, file_name="hasil_clustering_faskes.csv", mime="text/csv")

if __name__ == "__main__":
    main()