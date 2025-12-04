import streamlit as st
import pandas as pd

# ------------------------------------------------------
# KONFIGURASI HALAMAN
# ------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide",
    page_icon="📊"
)

st.title("📊 PT. XYZ Customer Segmentation Dashboard")
st.caption("Upload data transaksi pelanggan, lalu jelajahi hasil clustering & profiling segmen di halaman berikutnya.")

# Langkah-langkah singkat
st.markdown("""
1. 📁 **Upload file CSV transaksi pelanggan**
2. 🧮 **Lakukan clustering & segmentasi di halaman berikutnya**
3. 🔍 **Analisis profil tiap segmen dan temukan insight bisnis**
""")

# ------------------------------------------------------
# SESSION STATE UNTUK DATA
# ------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

# Layout dua kolom: kiri untuk upload, kanan untuk info/tips
col_left, col_right = st.columns([2, 1])

with col_left:
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Simpan data ke session state
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("✅ Data uploaded successfully!")

        # Quick preview & ringkasan data
        with st.expander("👀 Quick preview of your data"):
            st.dataframe(df.head())

        # Metrik ringkas
        n_rows, n_cols = df.shape
        

        m1, m2 = st.columns(2)
        m1.metric("Number of rows", n_rows)
        m2.metric("Number of columns", n_cols)
       

        st.info("Data siap dipakai. Kamu akan diarahkan ke halaman **Clustering**.")

        # Link secara langsung ke halaman clustering setelah upload
        st.switch_page("pages/1_Clustering.py")

    else:
        if st.session_state.df is not None:
            df = st.session_state.df
            st.write("✅ Data already loaded from previous session.")
            with st.expander("👀 Quick preview of existing data"):
                st.dataframe(df.head())

            n_rows, n_cols = df.shape
           

            m1, m2 = st.columns(2)
            m1.metric("Number of rows", n_rows)
            m2.metric("Number of columns", n_cols)
            
        else:
            st.info("Silakan upload file CSV untuk memulai.")

with col_right:
    st.markdown("### 💡 Tips data")
    st.markdown(
        """
        - Disarankan ada kolom seperti:
          - `customer_id` / `nama_customer`
          - `tanggal` transaksi  
          - `jumlah` / `nilai_transaksi`
        - Pastikan format tanggal sudah benar (YYYY-MM-DD).
        - Semakin lengkap data, semakin bagus hasil segmentasinya.
        """
    )

    st.markdown("### ℹ️ Info")
    st.markdown(
        """
        - File yang didukung: **CSV**  
        - Jika data sudah pernah di-upload, kamu bisa langsung buka halaman **Clustering** di sidebar.
        """
    )
