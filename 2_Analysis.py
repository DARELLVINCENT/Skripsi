import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------
# KONFIGURASI HALAMAN
# ------------------------------------------------------
st.set_page_config(page_title="Segmentation Profiling", layout="wide")
st.title("📊 Segmentation Profiling")

# ------------------------------------------------------
# 1. CEK HASIL CLUSTERING DI SESSION_STATE
# ------------------------------------------------------
sources: list[tuple[str, str]] = []

# CLV
if "clv_cluster_df" in st.session_state:
    clv_method = st.session_state.get("clv_method", "Unknown")
    label = f"CLV-based Clustering ({clv_method})"
    sources.append(("clv", label))

# MLRFM
if "mlrfm_cluster_df" in st.session_state:
    mlrfm_method = st.session_state.get("mlrfm_method", "Unknown")
    label = f"MLRFM-based Clustering ({mlrfm_method})"
    sources.append(("mlrfm", label))

if not sources:
    st.warning(
        "Belum ada hasil clustering di sesi ini.\n\n"
        "Silakan jalankan dulu halaman *Clustering* dan pastikan hasilnya "
        "disimpan ke `st.session_state['clv_cluster_df']` atau "
        "`st.session_state['mlrfm_cluster_df']`."
    )
    st.stop()

labels = [lbl for _, lbl in sources]
choice_label = st.selectbox("Pilih sumber hasil clustering", labels)

# "clv" atau "mlrfm"
source_code = next(code for code, lbl in sources if lbl == choice_label)


# ------------------------------------------------------
# 2. FUNGSI HELPER: GLOBAL vs CLUSTER (tanpa heatmap/radar)
# ------------------------------------------------------
def show_global_vs_cluster(
    global_avg: pd.Series,
    cluster_avg: pd.DataFrame,
    metric_cols: list[str],
    segment_col: str | None = None,
    extra_cols: list[str] | None = None,
) -> None:
    """
    Tampilkan:
      - Rata-rata global
      - Profil tiap cluster
      - Bar chart jumlah customer per cluster
    """
    st.subheader("🔍 Ringkasan Global")

    # global_avg: Series -> dict of float
    global_avg = global_avg.astype(float)
    global_avg_rounded = global_avg.round(3)
    global_avg_dict: dict[str, float] = {
        str(k): float(v) for k, v in global_avg_rounded.items()
    }

    global_df = pd.DataFrame.from_dict(
        global_avg_dict, orient="index", columns=["Global Average"]
    )
    st.table(global_df)

    # ---------------- Profil tiap cluster ----------------
    st.subheader("📦 Profil Tiap Cluster")
    cols_to_show: list[str] = ["Cluster"] + list(metric_cols)

    if segment_col and segment_col in cluster_avg.columns:
        cols_to_show.append(segment_col)

    if extra_cols:
        for c in extra_cols:
            if c in cluster_avg.columns:
                cols_to_show.append(c)

    st.dataframe(cluster_avg[cols_to_show].round(3), use_container_width=True)

    # ---------------- Jumlah customer per cluster ----------------
    st.subheader("👥 Jumlah Customer per Cluster")
    if "Count" in cluster_avg.columns:
        cluster_counts = cluster_avg.set_index("Cluster")["Count"]
        st.bar_chart(cluster_counts)
    else:
        st.info("Kolom 'Count' tidak ditemukan di cluster_avg.")


# ------------------------------------------------------
# 2b. HELPER: RINGKASAN TRANSAKSI PER CLUSTER + GRAFIK
# ------------------------------------------------------
def show_cluster_transaction_details(
    cluster_assign_df: pd.DataFrame,
    cust_col_candidates: list[str],
    title_suffix: str,
) -> None:
    """
    Menampilkan:
      - Transaksi per tahun (+ grafik bar)
      - Transaksi per bulan (+ line chart)
      - Barang favorit (Top 10) per cluster
      - Rata-rata nominal & jumlah barang per customer per cluster
      - Ringkasan rata-rata tersebut untuk semua cluster

    cluster_assign_df: DataFrame yang punya kolom customer + Cluster
    cust_col_candidates: kemungkinan nama kolom customer di cluster_assign_df
    """
    base_df = st.session_state.get("df")
    if not isinstance(base_df, pd.DataFrame):
        st.info("Data transaksi detail (df) tidak tersedia di session_state.")
        return

    if "tanggal" not in base_df.columns:
        st.info("Kolom 'tanggal' tidak ditemukan di data transaksi.")
        return

    df_trx = base_df.copy()

    # --- cari kolom customer di data transaksi ---
    base_cust_col: str | None = None
    for c in ["customer_id", "nama_customer"]:
        if c in df_trx.columns:
            base_cust_col = c
            break
    if base_cust_col is None:
        base_cust_col = df_trx.columns[0]

    # --- cari kolom customer di data cluster ---
    cluster_cust_col: str | None = None
    for c in cust_col_candidates:
        if c in cluster_assign_df.columns:
            cluster_cust_col = c
            break
    if cluster_cust_col is None:
        cluster_cust_col = cluster_assign_df.columns[0]

    # --- mapping customer -> cluster ---
    mapping_df = (
        cluster_assign_df[[cluster_cust_col, "Cluster"]]
        .drop_duplicates()
        .rename(columns={cluster_cust_col: "CustomerKey"})
    )

    df_trx = df_trx.rename(columns={base_cust_col: "CustomerKey"})
    df_trx = df_trx.merge(mapping_df, on="CustomerKey", how="left")
    df_trx = df_trx.dropna(subset=["Cluster"])

    if df_trx.empty:
        st.info("Tidak ada transaksi yang berhasil dipetakan ke cluster.")
        return

    df_trx["Cluster"] = df_trx["Cluster"].astype(int)
    df_trx["tanggal"] = pd.to_datetime(df_trx["tanggal"], errors="coerce")
    df_trx = df_trx.dropna(subset=["tanggal"])

    df_trx["Year"] = df_trx["tanggal"].dt.year # type: ignore[reportAttributeAccessIssue]
    df_trx["Month"] = df_trx["tanggal"].dt.to_period("M").astype(str) # type: ignore[reportAttributeAccessIssue]

    trx_year = (
        df_trx.groupby(["Cluster", "Year"])
        .size()
        .reset_index(name="Transactions")
    )
    trx_month = (
        df_trx.groupby(["Cluster", "Month"])
        .size()
        .reset_index(name="Transactions")
    )

    # --- cari kolom nama barang ---
    item_col: str | None = None
    for cand in ["nama_barang", "nama_produk", "product", "item", "barang"]:
        for col in df_trx.columns:
            if cand in col.lower():
                item_col = col
                break
        if item_col:
            break

    qty_col = "qty" if "qty" in df_trx.columns else None

    st.subheader(f"🧾 Ringkasan Transaksi per Cluster ({title_suffix})")

    # untuk summary rata-rata per customer per cluster
    summary_rows: list[dict[str, float | int | None]] = []

    for cluster_id in sorted(df_trx["Cluster"].unique()):
        df_c = df_trx[df_trx["Cluster"] == cluster_id].copy()

        # ----------------- RATA-RATA PER CUSTOMER -----------------
        avg_spend = None
        avg_qty = None
        if "total_harga" in df_c.columns:
            group = df_c.groupby("CustomerKey")

            if qty_col:
                cust_stats = group.agg(
                    total_spend=("total_harga", "sum"),
                    total_qty=(qty_col, "sum"),
                )
            else:
                # fallback: pakai jumlah transaksi kalau kolom qty tidak ada
                cust_stats = group.agg(
                    total_spend=("total_harga", "sum"),
                    total_qty=("CustomerKey", "size"),
                )

            if not cust_stats.empty:
                avg_spend = float(cust_stats["total_spend"].mean())
                avg_qty = float(cust_stats["total_qty"].mean())

        summary_rows.append(
            {
                "Cluster": cluster_id,
                "Avg_Spend_per_Customer": avg_spend,
                "Avg_Qty_per_Customer": avg_qty,
            }
        )

        with st.expander(f"Cluster {cluster_id} - Aktivitas Transaksi"):
            # tampilkan rata-rata di atas
            if avg_spend is not None:
                st.markdown(
                    f"**Rata-rata nominal pembelian per customer:** "
                    f"Rp {avg_spend:,.0f}"
                )
                satuan = "unit barang" if qty_col else "transaksi"
                st.markdown(
                    f"**Rata-rata jumlah barang per customer:** "
                    f"{avg_qty:,.2f} {satuan}"
                )
            else:
                st.info(
                    "Tidak bisa menghitung rata-rata nominal/qty karena "
                    "kolom `total_harga` tidak ditemukan."
                )

            st.markdown("---")

            # ---------- TAHUN ----------
            sub_year = trx_year[trx_year["Cluster"] == cluster_id][
                ["Year", "Transactions"]
            ].sort_values("Year")

            st.markdown("**Transaksi per Tahun**")
            st.dataframe(sub_year, use_container_width=True)

            if not sub_year.empty:
                fig_y = px.bar(
                    sub_year,
                    x="Year",
                    y="Transactions",
                    title="Transaksi per Tahun",
                )
                fig_y.update_layout(
                    xaxis_title="Tahun",
                    yaxis_title="Jumlah Transaksi",
                )
                st.plotly_chart(fig_y, use_container_width=True)

            # ---------- BULAN ----------
            sub_month = trx_month[trx_month["Cluster"] == cluster_id][
                ["Month", "Transactions"]
            ].sort_values("Month")

            st.markdown("**Transaksi per Bulan**")
            st.dataframe(sub_month, use_container_width=True)

            if not sub_month.empty:
                fig_m = px.line(
                    sub_month,
                    x="Month",
                    y="Transactions",
                    title="Transaksi per Bulan",
                    markers=True,
                )
                fig_m.update_layout(
                    xaxis_title="Periode (YYYY-MM)",
                    yaxis_title="Jumlah Transaksi",
                )
                st.plotly_chart(fig_m, use_container_width=True)

            # ---------- BARANG FAVORIT ----------
            if item_col is not None:
                if qty_col:
                    fav = (
                        df_c.groupby(item_col)[qty_col]
                        .sum()
                        .reset_index(name="Total_Qty")
                        .sort_values("Total_Qty", ascending=False)
                        .head(10)  # TOP 10
                    )
                else:
                    fav = (
                        df_c.groupby(item_col)
                        .size()
                        .reset_index(name="Transaksi")
                        .sort_values("Transaksi", ascending=False)
                        .head(10)  # TOP 10
                    )

                st.markdown("**Barang Favorit (Top 10)**")
                st.dataframe(fav, use_container_width=True)
            else:
                st.info(
                    "Kolom nama barang tidak ditemukan, tidak bisa menghitung barang favorit."
                )

    # --------- RINGKASAN RATA-RATA PER CUSTOMER UNTUK SEMUA CLUSTER ---------
    if summary_rows:
        st.markdown("**Ringkasan Rata-rata per Customer untuk Semua Cluster**")
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(
            summary_df.round(2),
            use_container_width=True,
        )


# ------------------------------------------------------
# 3. PROFILING: CLV-BASED CLUSTERING
# ------------------------------------------------------
if source_code == "clv":
    st.markdown("### 🔐 Sumber: CLV-based Clustering")

    clv_df = st.session_state["clv_cluster_df"].copy()

    metric_cols_clv: list[str] = ["Average", "Lifespan", "Frequency", "CLV"]
    for col in metric_cols_clv:
        clv_df[col] = pd.to_numeric(clv_df[col], errors="coerce")

    # Rata-rata global
    global_avg_clv: pd.Series = clv_df[metric_cols_clv].mean()

    # Rata-rata per cluster
    cluster_avg_clv = (
        clv_df.groupby("Cluster")[metric_cols_clv].mean().reset_index()
    )

    # Jumlah customer per cluster
    cluster_counts_clv = (
        clv_df["Cluster"]
        .value_counts()
        .rename_axis("Cluster")
        .reset_index(name="Count")
    )
    cluster_avg_clv = cluster_avg_clv.merge(
        cluster_counts_clv, on="Cluster", how="left"
    )

    # Flag high/low per metrik
    global_avg_clv_dict: dict[str, float] = {
        str(k): float(v) for k, v in global_avg_clv.items()
    }
    for col in metric_cols_clv:
        threshold = float(global_avg_clv_dict[col])
        cluster_avg_clv[f"{col}_high"] = cluster_avg_clv[col] > threshold

    # Segment map CLV
    segment_map_clv: dict[tuple[bool, bool, bool, bool], str] = {
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

    def map_segment_clv(row: pd.Series) -> str:
        key = (
            bool(row["Average_high"]),
            bool(row["Lifespan_high"]),
            bool(row["Frequency_high"]),
            bool(row["CLV_high"]),
        )
        return segment_map_clv.get(key, "Unmapped segment")

    cluster_avg_clv["Segment"] = cluster_avg_clv.apply(
        map_segment_clv, axis=1
    )

    # Pola A↑ L↑ F↑ C↑
    def pattern_clv(row: pd.Series) -> str:
        a = "A↑" if bool(row["Average_high"]) else "A↓"
        l = "L↑" if bool(row["Lifespan_high"]) else "L↓"
        f = "F↑" if bool(row["Frequency_high"]) else "F↓"
        c = "C↑" if bool(row["CLV_high"]) else "C↓"
        return f"{a} {l} {f} {c}"

    cluster_avg_clv["Pattern_CLV"] = cluster_avg_clv.apply(pattern_clv, axis=1)

    # Tampilkan ringkasan & visualisasi (tabel + bar chart)
    show_global_vs_cluster(
        global_avg=global_avg_clv,
        cluster_avg=cluster_avg_clv,
        metric_cols=metric_cols_clv,
        segment_col="Segment",
        extra_cols=["Pattern_CLV"],
    )

    # Keterangan pola CLV
    st.markdown(
        "**Keterangan pola A / L / F / C:**  \n"
        "- `A` = **Average Transaction Value** (rata-rata nilai transaksi)  \n"
        "- `L` = **Lifespan** (lama pelanggan aktif)  \n"
        "- `F` = **Frequency** (jumlah transaksi)  \n"
        "- `C` = **CLV Score** (nilai seumur hidup pelanggan)  \n"
        "Tanda panah **↑** berarti di atas rata-rata global, **↓** berarti di bawah rata-rata."
    )

    # Rekomendasi sederhana berbasis nama segment
    st.subheader("💡 Rekomendasi Umum per Segment (CLV)")
       
    reco_clv = {
        "High-value loyal customers": (
            "- Pertahankan dengan loyalty program eksklusif (VIP, points, tiering).\n"
            "- Prioritaskan layanan (CS prioritas, fast response).\n"
            "- Dorong advocacy: ajak mereka beri review & referral."
        ),
        "High-value new customers": (
            "- Segera follow-up setelah transaksi pertama (welcome series).\n"
            "- Berikan voucher pembelian berikutnya agar cepat repeat.\n"
            "- Edukasi value produk dan ajak join program loyalti."
        ),
        "Potential loyal customers": (
            "- Frekuensi sudah bagus, tingkatkan nilai belanja per transaksi.\n"
            "- Tawarkan bundle / paket hemat dengan AOV lebih tinggi.\n"
            "- Kirim rekomendasi produk pelengkap (cross-sell)."
        ),
        "High-value lost customers": (
            "- Buat kampanye win-back khusus (diskon besar 1x untuk comeback).\n"
            "- Kirim pesan personal yang mengingatkan histori belanja mereka.\n"
            "- Tanyakan feedback kenapa berhenti dan perbaiki hambatannya."
        ),
        "Platinum customers": (
            "- Mereka sering & CLV tinggi, jadikan champion segment.\n"
            "- Undang ke event eksklusif / early access produk baru.\n"
            "- Pertahankan dengan benefit non-diskon (privilege, layanan khusus)."
        ),
        "Consuming promotional customers": (
            "- Sering belanja tapi margin tipis, biasanya sensitif promo.\n"
            "- Sesuaikan promo: batasi diskon langsung, perbanyak bundling & poin.\n"
            "- Naikkan value dengan add-on ber-margin tinggi."
        ),
        "Potential consuming customers": (
            "- Baru beberapa kali transaksi, bisa diarahkan jadi loyal.\n"
            "- Beri edukasi produk + konten how-to supaya makin percaya.\n"
            "- Coba upsell paket lebih besar dengan bonus kecil."
        ),
        "Consuming churn customers": (
            "- Dulu suka promo, sekarang sudah tidak aktif.\n"
            "- Kirim promo spesial comeback dengan batas waktu jelas.\n"
            "- Uji coba harga / bundling yang lebih sesuai kemampuan mereka."
        ),
        "High frequency customers": (
            "- Sangat sering beli, tapi CLV relatif tidak setinggi platinum.\n"
            "- Fokus ke efisiensi: paket langganan, auto-repeat order.\n"
            "- Pastikan stok & ketersediaan produk selalu aman."
        ),
        "Frequency promotional customers": (
            "- Sering beli saat promo saja.\n"
            "- Atur kalender promo terencana (payday, tanggal spesial) dan arahkan ke mereka.\n"
            "- Coba dorong pembelian non-promo dengan benefit lain (gratis ongkir, poin)."
        ),
        "Potential frequency customers": (
            "- Masih jarang beli tapi potensial jadi sering.\n"
            "- Gunakan reminder rutin (WA/email) saat mereka biasanya belanja.\n"
            "- Tawarkan paket hemat untuk pembelian berkala."
        ),
        "Frequency churn customers": (
            "- Dulunya sering, sekarang hampir tidak aktif.\n"
            "- Kirim pesan personal: \"Kami kangen, ada yang bisa kami bantu?\".\n"
            "- Beri opsi downgrade paket / nominal agar lebih terjangkau."
        ),
        "Low-cost consuming customers": (
            "- Sering beli nilai kecil, kontribusi margin rendah.\n"
            "- Arahkan ke produk dengan margin lebih tinggi tapi masih terjangkau.\n"
            "- Bundling beberapa item kecil jadi satu paket."
        ),
        "Uncertain new customers": (
            "- Masih tahap coba-coba, datanya belum stabil.\n"
            "- Pastikan pengalaman pertama sangat bagus (pengiriman, CS, kualitas).\n"
            "- Kirim konten onboarding: cara pakai, rekomendasi produk lanjutan."
        ),
        "High-cost consuming customers": (
            "- CLV belum tinggi tapi cost untuk melayani mereka besar (banyak komplain/retur/promo berat).\n"
            "- Evaluasi apakah segmen ini layak di-push atau cukup dipertahankan saja.\n"
            "- Standarkan aturan promo & layanan agar cost lebih terkendali."
        ),
        "Uncertain lost customers": (
            "- Aktivitas & nilai belanja rendah, dan sekarang sudah tidak aktif.\n"
            "- Jangan fokus terlalu besar di segmen ini.\n"
            "- Bisa tetap disertakan di broadcast massal, tapi jangan diberi promo mahal."
        ),
    }

    for _, r in cluster_avg_clv.iterrows():
        seg = r["Segment"]
        with st.expander(f"Cluster {int(r['Cluster'])}: {seg}"):
            st.markdown(reco_clv.get(seg, "Belum ada rekomendasi spesifik untuk segmen ini."))

    # ------------------------------------------------------
    # Tabel customer & cluster (CLV)
    # ------------------------------------------------------
    st.subheader("📋 Daftar Customer dan Cluster (CLV)")

    # Cari nama kolom customer secara fleksibel
    cust_col = "customer_id"
    if cust_col not in clv_df.columns:
        possible = [c for c in clv_df.columns if "customer" in c.lower()]
        cust_col = possible[0] if possible else clv_df.columns[0]

    clv_customer_table = clv_df[[cust_col, "Cluster"]].copy()
    clv_customer_table = clv_customer_table.merge(
        cluster_avg_clv[["Cluster", "Segment", "Pattern_CLV"]],
        on="Cluster",
        how="left",
    )

    st.dataframe(
        clv_customer_table.rename(
            columns={
                cust_col: "Customer",
                "Cluster": "Cluster ID",
                "Segment": "Segment",
                "Pattern_CLV": "Pola (A/L/F/C)",
            }
        ),
        use_container_width=True,
    )

    # ------------------------------------------------------
    # Ringkasan transaksi per cluster (CLV)
    # ------------------------------------------------------
    show_cluster_transaction_details(
        cluster_assign_df=clv_df,
        cust_col_candidates=[cust_col, "customer_id", "nama_customer"],
        title_suffix="CLV",
    )


# ------------------------------------------------------
# 4. PROFILING: MLRFM-BASED CLUSTERING
# ------------------------------------------------------
elif source_code == "mlrfm":
    st.markdown("### 🔐 Sumber: MLRFM-based Clustering")

    mlrfm_df = st.session_state["mlrfm_cluster_df"].copy()

    metric_cols_m: list[str] = [
        "Recency",
        "Multi_Layer_Frequency",
        "Multi_Layer_Monetary",
    ]
    for col in metric_cols_m:
        mlrfm_df[col] = pd.to_numeric(mlrfm_df[col], errors="coerce")

    # Rata-rata global
    global_avg_m: pd.Series = mlrfm_df[metric_cols_m].mean()

    # Rata-rata per cluster
    cluster_avg_m = (
        mlrfm_df.groupby("Cluster")[metric_cols_m].mean().reset_index()
    )

    # Jumlah customer per cluster
    cluster_counts_m = (
        mlrfm_df["Cluster"]
        .value_counts()
        .rename_axis("Cluster")
        .reset_index(name="Count")
    )
    cluster_avg_m = cluster_avg_m.merge(
        cluster_counts_m, on="Cluster", how="left"
    )

    # Flag high/low per metrik
    global_avg_m_dict: dict[str, float] = {
        str(k): float(v) for k, v in global_avg_m.items()
    }
    for col in metric_cols_m:
        threshold = float(global_avg_m_dict[col])
        cluster_avg_m[f"{col}_high"] = cluster_avg_m[col] > threshold

    # Segment map MLRFM
    segment_map_mlrfm: dict[tuple[bool, bool, bool], str] = {
        (True,  True,  True):  "Loyal customers",                    # R↑ F↑ M↑
        (True,  False, True):  "Promising customers",                # R↑ F↓ M↑
        (True,  False, False): "New customers",                      # R↑ F↓ M↓
        (False, False, False): "Lost customers",                     # R↓ F↓ M↓
        (False, True,  True):  "Lost customers 'Churned'",           # R↓ F↑ M↑ 
        (False, False, True): "Lost Customers 'High Value'" , 
        (False, True , False): "Lost Customers 'At Risk'"                # R↓ F↓ M↑
    }

    # Tambahkan kolom pola RFM: R↑ / R↓, F↑ / F↓, M↑ / M↓
    def pattern_rfm(row: pd.Series) -> str:
        r = "R↑" if bool(row["Recency_high"]) else "R↓"
        f = "F↑" if bool(row["Multi_Layer_Frequency_high"]) else "F↓"
        m = "M↑" if bool(row["Multi_Layer_Monetary_high"]) else "M↓"
        return f"{r} {f} {m}"

    # Urutan boolean: (Recency_high, Multi_Layer_Frequency_high, Multi_Layer_Monetary_high)
    def map_segment_mlrfm(row: pd.Series) -> str:
        key = (
            bool(row["Recency_high"]),
            bool(row["Multi_Layer_Frequency_high"]),
            bool(row["Multi_Layer_Monetary_high"]),
        )
        return segment_map_mlrfm.get(key, "Unmapped segment")

    cluster_avg_m["Segment"] = cluster_avg_m.apply(
        map_segment_mlrfm, axis=1
    )
    cluster_avg_m["Pattern_RFM"] = cluster_avg_m.apply(pattern_rfm, axis=1)

    # Tampilkan ringkasan & visualisasi (tabel + bar chart)
    show_global_vs_cluster(
        global_avg=global_avg_m,
        cluster_avg=cluster_avg_m,
        metric_cols=metric_cols_m,
        segment_col="Segment",
        extra_cols=["Pattern_RFM"],
    )

    # Keterangan pola R / F / M
    st.markdown(
        "**Keterangan pola R / F / M:**  \n"
        "- `R` = **Recency score** (kebaruan transaksi; semakin tinggi berarti transaksi lebih baru/masih aktif)  \n"
        "- `F` = **Multi Layer Frequency** (gabungan frekuensi transaksi beberapa periode)  \n"
        "- `M` = **Multi Layer Monetary** (gabungan nilai belanja beberapa periode)  \n"
        "Tanda panah **↑** berarti di atas rata-rata global, **↓** berarti di bawah rata-rata."
    )

    # --------------------------------------------------
    # Rekomendasi tindakan per segmen (MLRFM)
    # --------------------------------------------------
    st.subheader("💡 Rekomendasi Tindakan per Segment (MLRFM)")

    reco_mlrfm: dict[str, str] = {
    "Loyal customers": (
        "- Fokus retensi dan apresiasi (loyalty program, poin, hadiah).\n"
        "- Beri akses prioritas ke produk baru & promo eksklusif.\n"
        "- Mintakan review/testimoni untuk memperkuat social proof."
    ),
    "Promising customers": (
        "- Mereka baru, tapi nilai belanja sudah tinggi.\n"
        "- Kirim ucapan terima kasih + voucher repeat order.\n"
        "- Follow-up cepat setelah pembelian untuk menjaga pengalaman positif."
    ),
    "New customers": (
        "- Bangun kebiasaan repeat buying (welcome series via email/WA).\n"
        "- Beri promo khusus untuk 2–3 transaksi pertama.\n"
        "- Sederhanakan proses pemesanan & pembayaran."
    ),
    "Lost customers": (
        "- Jangan habiskan terlalu banyak biaya, tapi boleh tetap kirim broadcast promo massal.\n"
        "- Gunakan mereka sebagai kontrol saat menguji kampanye baru.\n"
        "- Fokus effort utama ke segmen dengan nilai lebih tinggi."
    ),
    "Lost customers 'Churned'": (
        "- Mereka dulu sering dan/atau banyak belanja, tapi sekarang hilang.\n"
        "- Kirim kampanye win-back agresif (diskon spesial atau bundle favorit mereka).\n"
        "- Personalisasi pesan dengan menyebut histori belanja & tawarkan alasan kuat untuk kembali.\n"
        "- Minta feedback singkat kenapa berhenti belanja (survey/form/chat CS)."
    ),
    "Lost Customers 'High Value'": (
        "- Prioritaskan sebagai target comeback bernilai tinggi.\n"
        "- Buat penawaran sangat personal & eksklusif (VIP comeback offer sekali pakai).\n"
        "- Tawarkan produk premium/koleksi terbaru yang sesuai histori belanja besar mereka.\n"
        "- Jika memungkinkan, lakukan follow-up 1:1 (WA/telepon) untuk memahami kebutuhan & hambatan."
    ),
}

    for _, r in cluster_avg_m.iterrows():
        seg = r["Segment"]
        pattern = r["Pattern_RFM"]
        cluster_id = int(r["Cluster"])

        with st.expander(f"Cluster {cluster_id}: {seg} ({pattern})"):
            st.markdown(f"**Pola RFM:** `{pattern}`")
            st.markdown(reco_mlrfm.get(seg, "Belum ada rekomendasi spesifik untuk segmen ini."))

    # ------------------------------------------------------
    # Tabel customer & cluster (MLRFM)
    # ------------------------------------------------------
    st.subheader("📋 Daftar Customer dan Cluster (MLRFM)")

    cust_col_m = "nama_customer"
    if cust_col_m not in mlrfm_df.columns:
        possible = [c for c in mlrfm_df.columns if "customer" in c.lower()]
        cust_col_m = possible[0] if possible else mlrfm_df.columns[0]

    mlrfm_customer_table = mlrfm_df[[cust_col_m, "Cluster"]].copy()
    mlrfm_customer_table = mlrfm_customer_table.merge(
        cluster_avg_m[["Cluster", "Segment", "Pattern_RFM"]],
        on="Cluster",
        how="left",
    )

    st.dataframe(
        mlrfm_customer_table.rename(
            columns={
                cust_col_m: "Customer",
                "Cluster": "Cluster ID",
                "Segment": "Segment",
                "Pattern_RFM": "Pola (R/F/M)",
            }
        ),
        use_container_width=True,
    )

    # ------------------------------------------------------
    # Ringkasan transaksi per cluster (MLRFM)
    # ------------------------------------------------------
    show_cluster_transaction_details(
        cluster_assign_df=mlrfm_df,
        cust_col_candidates=[cust_col_m, "nama_customer", "customer_id"],
        title_suffix="MLRFM",
    )

# # ======================================================
# # 5. NEXT BEST PRODUCT (ASSOCIATION RULES) PER CLUSTER
# # ======================================================
# # ============================================
# # ASSOCIATION RULES / NEXT BEST PRODUCT (NBP)
# # ============================================
# from itertools import combinations
# import numpy as np
# import pandas as pd
# import streamlit as st

# # ---------- Util: deteksi kolom ----------
# def _detect_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
#     for cand in candidates:
#         for col in df.columns:
#             if cand.lower() in col.lower():
#                 return col
#     return None

# def _detect_txn_id(df: pd.DataFrame) -> str | None:
#     """
#     Cari kolom ID transaksi. Jika tidak ada, buat proxy dari (CustomerKey + tanggal[date]).
#     """
#     candidates = [
#         "order_id", "no_faktur", "invoice", "id_transaksi",
#         "no_nota", "sales_order", "trx_id", "id_order"
#     ]
#     col = _detect_col(df, candidates)
#     if col:
#         return col

#     # Proxy jika tak ada kolom id
#     if "CustomerKey" in df.columns and "tanggal" in df.columns:
#         df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
#         df["__txn_proxy__"] = (
#             df["CustomerKey"].astype(str) + "_" + df["tanggal"].dt.strftime("%Y-%m-%d"))# type: ignore[reportAttributeAccessIssue]
#         return "__txn_proxy__"
#     return None

# # ---------- Siapkan basket per transaksi ----------
# def _prepare_baskets(
#     df_trx: pd.DataFrame,
#     cluster_assign_df: pd.DataFrame,
#     cust_col_candidates: list[str],
# ) -> pd.DataFrame | None:
#     """
#     Gabungkan transaksi dengan mapping Cluster, dan bentuk basket per transaksi.
#     Output kolom: [Cluster, TxnID, Item]
#     """
#     # cari kolom customer di tabel cluster
#     cluster_cust_col = None
#     for c in cust_col_candidates:
#         if c in cluster_assign_df.columns:
#             cluster_cust_col = c
#             break
#     if cluster_cust_col is None:
#         cluster_cust_col = cluster_assign_df.columns[0]

#     # cari kolom customer di data transaksi
#     base_cust_col = None
#     for c in ["customer_id", "nama_customer"]:
#         if c in df_trx.columns:
#             base_cust_col = c
#             break
#     if base_cust_col is None:
#         base_cust_col = df_trx.columns[0]

#     # cari kolom item
#     item_col = _detect_col(
#         df_trx, ["nama_barang", "nama_produk", "product", "item", "barang"]
#     )
#     if item_col is None:
#         st.info("NBP: Kolom nama barang/produk tidak ditemukan.")
#         return None

#     # normalisasi nama item sederhana (kurangi variasi ejaan)
#     df_trx = df_trx.copy()
#     df_trx[item_col] = (
#         df_trx[item_col].astype(str).str.strip().str.upper().str.replace(r"\s+", " ", regex=True)
#     )

#     # normalisasi tanggal jika ada
#     if "tanggal" in df_trx.columns:
#         df_trx["tanggal"] = pd.to_datetime(df_trx["tanggal"], errors="coerce")

#     # mapping customer->cluster
#     mapping_df = (
#         cluster_assign_df[[cluster_cust_col, "Cluster"]]
#         .drop_duplicates()
#         .rename(columns={cluster_cust_col: "CustomerKey"})
#     )

#     trx = df_trx.rename(columns={base_cust_col: "CustomerKey"}).merge(
#         mapping_df, on="CustomerKey", how="left"
#     )
#     trx = trx.dropna(subset=["Cluster"])
#     if trx.empty:
#         st.info("NBP: Tidak ada transaksi yang terpetakan ke cluster.")
#         return None

#     # deteksi kolom id transaksi (atau buat proxy)
#     txn_col = _detect_txn_id(trx)
#     if txn_col is None:
#         st.info("NBP: Tidak menemukan ID transaksi dan gagal membuat proxy.")
#         return None

#     trx["Cluster"] = trx["Cluster"].astype(int)
#     trx = trx[["Cluster", "CustomerKey", txn_col, item_col]].dropna()
#     trx = trx.rename(columns={txn_col: "TxnID", item_col: "Item"})

#     # pastikan satu item unik per transaksi (hindari duplikasi)
#     trx = trx.drop_duplicates(subset=["Cluster", "TxnID", "Item"])
#     return trx

# # ---------- Hitung asosiasi A->B ----------
# def _assoc_rules_for_cluster(
#     baskets: pd.DataFrame,
#     min_pair_count: int = 3,
# ) -> pd.DataFrame:
#     """
#     baskets: DataFrame [TxnID, Item] satu cluster.
#     Return kolom: A, B, pair_count, support, conf, lift
#     """
#     grouped = baskets.groupby("TxnID")["Item"].apply(list)
#     n_baskets = len(grouped)
#     if n_baskets == 0:
#         return pd.DataFrame(columns=["A", "B", "pair_count", "support", "conf", "lift"])

#     item_counts = {}
#     pair_counts = {}
#     for items in grouped:
#         uniq = sorted(set(items))
#         for a in uniq:
#             item_counts[a] = item_counts.get(a, 0) + 1
#         for a, b in combinations(uniq, 2):
#             pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
#             pair_counts[(b, a)] = pair_counts.get((b, a), 0) + 1  # arah

#     rows = []
#     for (a, b), pc in pair_counts.items():
#         if pc < min_pair_count:
#             continue
#         supp_ab = pc / n_baskets
#         conf = pc / item_counts.get(a, 1)
#         supp_b = item_counts.get(b, 0) / n_baskets
#         lift = conf / supp_b if supp_b > 0 else np.nan
#         rows.append(
#             {"A": a, "B": b, "pair_count": pc, "support": supp_ab, "conf": conf, "lift": lift}
#         )
#     rules = pd.DataFrame(rows)
#     if not rules.empty:
#         rules = rules.sort_values(["lift", "conf", "support", "pair_count"], ascending=False)
#     return rules

# # ---------- Over-Index (SKU khas per cluster) ----------
# def _over_index_table(trx_all: pd.DataFrame) -> pd.DataFrame:
#     """
#     share_cluster / share_global berbasis kemunculan item di basket.
#     """
#     global_baskets = trx_all.drop_duplicates(["TxnID", "Item"])
#     g_item = (
#         global_baskets.groupby("Item")["TxnID"].nunique()
#         .rename("basket_item_global")
#         .reset_index()
#     )
#     g_total = max(global_baskets["TxnID"].nunique(), 1)

#     frames = []
#     for c in sorted(trx_all["Cluster"].unique()):
#         sub = trx_all[trx_all["Cluster"] == c].drop_duplicates(["TxnID", "Item"])
#         c_total = max(sub["TxnID"].nunique(), 1)
#         c_item = sub.groupby("Item")["TxnID"].nunique().rename("basket_item_cluster").reset_index()
#         m = c_item.merge(g_item, on="Item", how="left").fillna(0)
#         m["share_cluster"] = m["basket_item_cluster"] / c_total
#         m["share_global"] = m["basket_item_global"] / g_total
#         m["over_index"] = np.where(
#             m["share_global"] > 0, m["share_cluster"] / m["share_global"], np.nan
#         )
#         m["Cluster"] = c
#         frames.append(m[["Cluster", "Item", "share_cluster", "share_global", "over_index"]])
#     return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# # ---------- Komponen utama untuk dipanggil ----------
# def show_next_best_product(
#     cluster_assign_df: pd.DataFrame,
#     cust_col_candidates: list[str],
#     title_suffix: str,
#     min_pair_count: int = 3,
#     top_k_rules: int = 20,
#     top_k_rec_per_anchor: int = 5,
# ) -> None:
#     """
#     Render NBP per cluster ke Streamlit.
#     Sumber transaksi diambil dari st.session_state['df'].
#     """
#     base_df = st.session_state.get("df")
#     if not isinstance(base_df, pd.DataFrame):
#         st.info("NBP: Data transaksi detail (df) tidak tersedia di session_state.")
#         return

#     trx = _prepare_baskets(base_df, cluster_assign_df, cust_col_candidates)
#     if trx is None or trx.empty:
#         return

#     st.subheader(f"🧠 Next Best Product (NBP) – {title_suffix}")

#     # ---------------- Diagnostics ----------------
#     with st.expander("NBP Diagnostics"):
#         diag_rows = []
#         for c in sorted(trx["Cluster"].unique()):
#             sub = trx[trx["Cluster"] == c]
#             g = sub.groupby("TxnID")["Item"].nunique()
#             diag_rows.append(
#                 {
#                     "Cluster": int(c),
#                     "N_Baskets": int(g.shape[0]),
#                     "Avg_Items_per_Basket": float(g.mean() if len(g) else 0.0),
#                     "Med_Items_per_Basket": float(g.median() if len(g) else 0.0),
#                     "N_Unique_Items": int(sub["Item"].nunique()),
#                 }
#             )
#         st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)

#     # ---------------- Over-Index global ----------------
#     oi = _over_index_table(trx)

#     # ---------------- Per Cluster ----------------
#     for c in sorted(trx["Cluster"].unique()):
#         sub = trx[trx["Cluster"] == c][["TxnID", "Item"]]
#         rules = _assoc_rules_for_cluster(sub, min_pair_count=min_pair_count)

#         with st.expander(f"Cluster {c} – Rekomendasi & Rules", expanded=False):
#             if rules.empty:
#                 st.warning(
#                     "Belum ada aturan asosiasi yang memenuhi ambang minimal. Menampilkan fallback."
#                 )
#                 # Fallback 1: SKU Over-Index
#                 oi_sub = (
#                     oi[oi["Cluster"] == c]
#                     .sort_values("over_index", ascending=False)
#                     .head(15)
#                     .copy()
#                 )
#                 if not oi_sub.empty:
#                     oi_sub["share_cluster"] = (oi_sub["share_cluster"] * 100).round(2)
#                     oi_sub["share_global"] = (oi_sub["share_global"] * 100).round(2)
#                     oi_sub["over_index"] = oi_sub["over_index"].round(3)
#                     oi_sub = oi_sub.rename(
#                         columns={
#                             "Item": "SKU",
#                             "share_cluster": "Share Cluster (%)",
#                             "share_global": "Share Global (%)",
#                             "over_index": "Over-Index",
#                         }
#                     )
#                     st.markdown("**Fallback – SKU Khas (Over-Index Tertinggi)**")
#                     st.dataframe(oi_sub, use_container_width=True)
#                 else:
#                     st.info("Fallback juga kosong (mungkin data terlalu sedikit).")
#                 continue

#             # ---- Top Rules ----
#             st.markdown("**Aturan Asosiasi Terbaik (Top Rules)**")
#             rules_disp = rules.head(top_k_rules).copy()
#             rules_disp["support"] = (rules_disp["support"] * 100).round(2)
#             rules_disp["conf"] = (rules_disp["conf"] * 100).round(2)
#             rules_disp["lift"] = rules_disp["lift"].round(3)
#             rules_disp = rules_disp.rename(
#                 columns={
#                     "A": "If buy (A)",
#                     "B": "Then (B)",
#                     "pair_count": "Pair Cnt",
#                     "conf": "Confidence (%)",
#                     "support": "Support (%)",
#                     "lift": "Lift",
#                 }
#             )
#             st.dataframe(rules_disp, use_container_width=True)

#             # ---- NBP per Anchor ----
#             st.markdown("**Next Best Product per Anchor (A)**")
#             rules["score"] = rules["lift"] * rules["support"]  # bisa disesuaikan (× margin/stok)
#             nbp = (
#                 rules.sort_values(["A", "score", "conf", "lift"], ascending=False)
#                 .groupby("A")
#                 .head(top_k_rec_per_anchor)
#                 .reset_index(drop=True)
#             )
#             nbp_view = nbp[["A", "B", "pair_count", "support", "conf", "lift", "score"]].copy()
#             for col in ["support", "conf"]:
#                 nbp_view[col] = (nbp_view[col] * 100).round(2)
#             nbp_view["lift"] = nbp_view["lift"].round(3)
#             nbp_view["score"] = nbp_view["score"].round(4)
#             nbp_view = nbp_view.rename(
#                 columns={
#                     "A": "Anchor (A)",
#                     "B": "NBP (B)",
#                     "pair_count": "Pair Cnt",
#                     "support": "Support (%)",
#                     "conf": "Confidence (%)",
#                     "lift": "Lift",
#                     "score": "Score",
#                 }
#             )
#             st.dataframe(nbp_view, use_container_width=True)

#             # ---- SKU Over-Index (khas cluster) ----
#             st.markdown("**SKU Over-Index (Khas Cluster)**")
#             oi_sub = oi[oi["Cluster"] == c].copy()
#             oi_sub = oi_sub.sort_values("over_index", ascending=False).head(20)
#             oi_sub["share_cluster"] = (oi_sub["share_cluster"] * 100).round(2)
#             oi_sub["share_global"] = (oi_sub["share_global"] * 100).round(2)
#             oi_sub["over_index"] = oi_sub["over_index"].round(3)
#             oi_sub = oi_sub.rename(
#                 columns={
#                     "Item": "SKU",
#                     "share_cluster": "Share Cluster (%)",
#                     "share_global": "Share Global (%)",
#                     "over_index": "Over-Index",
#                 }
#             )
#             st.dataframe(oi_sub, use_container_width=True)
