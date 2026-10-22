
import time
import pandas as pd
import numpy as np

from scipy.sparse import load_npz
from sklearn.preprocessing import normalize

# ==========================================
# 1. MEMUAT VSM
# ==========================================
X_user = load_npz("vsm_user.npz").tocsr()
X_item = X_user.T.tocsr()

print("=== DIMENSI VSM ===")
print("VSM User:", X_user.shape)
print("VSM Item:", X_item.shape)

# ==========================================
# 2. MENENTUKAN SAMPEL YANG SEBANDING
# ==========================================
SEED = 42
JUMLAH_SAMPEL = min(1000, X_user.shape[0], X_item.shape[0])

rng = np.random.default_rng(SEED)

indeks_user = np.sort(
    rng.choice(X_user.shape[0], JUMLAH_SAMPEL, replace=False)
)
indeks_item = np.sort(
    rng.choice(X_item.shape[0], JUMLAH_SAMPEL, replace=False)
)

U = X_user[indeks_user, :].tocsr()
I = X_item[indeks_item, :].tocsr()

# ==========================================
# 3. FUNGSI PROYEKSI DAN PENGUKURAN
# ==========================================
def analisis_proyeksi(matriks, nama):
    matriks_norm = normalize(matriks, norm="l2")

    mulai = time.perf_counter()

    similarity = (matriks_norm @ matriks_norm.T).tocsr()

    waktu = time.perf_counter() - mulai

    # Hilangkan self-similarity
    similarity.setdiag(0)
    similarity.eliminate_zeros()

    # Hitung pasangan unik di segitiga atas
    similarity = similarity.maximum(similarity.T)
    similarity.setdiag(0)
    similarity.eliminate_zeros()

    pasangan_unik = similarity.nnz // 2

    return {
        "proyeksi": nama,
        "jumlah_simpul_sampel": matriks.shape[0],
        "jumlah_pasangan_similarity_positif": int(pasangan_unik),
        "waktu_komputasi_detik": round(waktu, 4)
    }

# ==========================================
# 4. MEMBANDINGKAN USER DAN ITEM
# ==========================================
hasil = pd.DataFrame([
    analisis_proyeksi(U, "User-User"),
    analisis_proyeksi(I, "Item-Item")
])

print("\n=== PERBANDINGAN PROYEKSI ===")
print(hasil.to_string(index=False))

hasil.to_csv("perbandingan_proyeksi_sampel.csv", index=False)

print("\nHasil disimpan ke perbandingan_proyeksi_sampel.csv")
