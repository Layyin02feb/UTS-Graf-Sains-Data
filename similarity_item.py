
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from sklearn.preprocessing import normalize

# ==================================================
# 1. MEMUAT VSM ITEM DAN INDEKS
# ==================================================
print("Memuat VSM User...")
X_user = load_npz("vsm_user.npz").tocsr()

# Baris = item, kolom = user
X_item = X_user.T.tocsr()
X_item = normalize(X_item, norm="l2", axis=1)

indeks = pd.read_csv("indeks_item.csv")

print("Kolom indeks item:", indeks.columns.tolist())
print("Jumlah item:", X_item.shape[0])

# Cari kolom yang menyimpan ID item asli
kolom_id = next(
    (c for c in indeks.columns
     if c.lower() in ["itemid", "item_id", "item"]),
    None
)

if kolom_id is None:
    raise ValueError(
        "Kolom ID item tidak ditemukan. "
        "Periksa kolom pada indeks_item.csv."
    )

# Tentukan kolom nomor baris matriks
kolom_index = next(
    (c for c in indeks.columns
     if c.lower() in ["index", "item_index", "matrix_index"]),
    None
)

if kolom_index is not None:
    indeks[kolom_index] = pd.to_numeric(
        indeks[kolom_index], errors="coerce"
    )
    indeks = indeks.dropna(subset=[kolom_index])
    indeks[kolom_index] = indeks[kolom_index].astype(int)
else:
    # Jika file indeks mengikuti urutan baris matriks
    indeks = indeks.reset_index(drop=True)
    kolom_index = None

# ==================================================
# 2. PILIH ITEM ACUAN
# ==================================================
# Contoh menggunakan item 5411 yang cukup populer.
# Ganti dengan ID item lain jika diperlukan.
ITEM_ACUAN = 5411
TOP_K = 5

baris = indeks[indeks[kolom_id].astype(str) == str(ITEM_ACUAN)]

if baris.empty:
    raise ValueError(
        f"Item {ITEM_ACUAN} tidak ditemukan di indeks_item.csv"
    )

if kolom_index:
    idx = int(baris.iloc[0][kolom_index])
else:
    idx = int(baris.index[0])

if idx < 0 or idx >= X_item.shape[0]:
    raise ValueError("Indeks item berada di luar dimensi matriks.")

print(f"\nItem acuan: {ITEM_ACUAN}")

# ==================================================
# 3. HITUNG COSINE SIMILARITY
# ==================================================
# Hanya menghitung kemiripan terhadap satu item acuan.
skor = X_item @ X_item[idx].T
skor = skor.toarray().ravel()

# Item tidak boleh menjadi rekomendasi bagi dirinya sendiri.
skor[idx] = -1

# Ambil kandidat dengan skor tertinggi
jumlah = min(TOP_K, len(skor) - 1)
kandidat = np.argpartition(skor, -jumlah)[-jumlah:]
kandidat = kandidat[np.argsort(skor[kandidat])[::-1]]

# ==================================================
# 4. SIMPAN HASIL
# ==================================================
# Petakan indeks matriks kembali ke ID item asli
if kolom_index:
    peta_id = dict(zip(indeks[kolom_index], indeks[kolom_id]))
else:
    peta_id = dict(enumerate(indeks[kolom_id]))

hasil = pd.DataFrame({
    "item_acuan": ITEM_ACUAN,
    "item_mirip": [peta_id.get(int(i), None) for i in kandidat],
    "cosine_similarity": skor[kandidat]
})

hasil = hasil[hasil["item_mirip"].notna()]
hasil.to_csv("similarity_item_top5.csv", index=False)

print("\n=== TOP-5 ITEM PALING MIRIP ===")
print(hasil.to_string(index=False))
print("\nHasil disimpan ke similarity_item_top5.csv")
