
# ============================================================
# REKOMENDASI ITEM RETAILROCKET MENGGUNAKAN KNN
# Evaluasi: Hit@5 dan NDCG@5
# Metode: Item-based Collaborative Filtering + Cosine Similarity
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# 1. PENGATURAN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FILE_TRAIN = DATA_DIR / "training.csv"
FILE_TEST = DATA_DIR / "testing.csv"
FILE_VSM = BASE_DIR / "vsm_user.npz"
FILE_VISITOR = BASE_DIR / "indeks_visitor.csv"
FILE_ITEM = BASE_DIR / "indeks_item.csv"

FILE_EVALUASI = BASE_DIR / "evaluasi_knn.csv"
FILE_REKOMENDASI = BASE_DIR / "rekomendasi_knn.csv"

K = 5
JUMLAH_REKOMENDASI = 5
JUMLAH_SAMPEL = 500
RANDOM_STATE = 42

# Bobot interaksi
BOBOT_EVENT = {
    "view": 1,
    "addtocart": 2,
    "transaction": 3,
}

# ============================================================
# 2. MEMBACA DATA
# ============================================================

for file in [
    FILE_TRAIN,
    FILE_TEST,
    FILE_VSM,
    FILE_VISITOR,
    FILE_ITEM,
]:
    if not file.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {file}")

train = pd.read_csv(FILE_TRAIN)
test = pd.read_csv(FILE_TEST)

indeks_visitor = pd.read_csv(FILE_VISITOR)
indeks_item = pd.read_csv(FILE_ITEM)

print("Baris training:", len(train))
print("Baris testing :", len(test))
print("Kolom indeks visitor:", indeks_visitor.columns.tolist())
print("Kolom indeks item:", indeks_item.columns.tolist())

# Pastikan kolom wajib tersedia
kolom_train = {"visitorid", "itemid", "event"}
kolom_test = {"visitorid", "itemid", "event"}

if not kolom_train.issubset(train.columns):
    raise ValueError(
        f"Kolom training tidak lengkap. Dibutuhkan: {kolom_train}"
    )

if not kolom_test.issubset(test.columns):
    raise ValueError(
        f"Kolom testing tidak lengkap. Dibutuhkan: {kolom_test}"
    )

indeks_visitor.columns = indeks_visitor.columns.str.lower()
indeks_item.columns = indeks_item.columns.str.lower()

if not {"indeks", "visitorid"}.issubset(indeks_visitor.columns):
    raise ValueError(
        "indeks_visitor.csv harus memiliki kolom indeks dan visitorId."
    )

if not {"indeks", "itemid"}.issubset(indeks_item.columns):
    raise ValueError(
        "indeks_item.csv harus memiliki kolom indeks dan itemId."
    )

# Samakan tipe ID supaya pemetaan konsisten
for df in [train, test]:
    df["visitorid"] = pd.to_numeric(
        df["visitorid"], errors="coerce"
    )
    df["itemid"] = pd.to_numeric(
        df["itemid"], errors="coerce"
    )
    df["event"] = df["event"].astype(str).str.lower()

    df.dropna(
        subset=["visitorid", "itemid", "event"],
        inplace=True
    )

    df["visitorid"] = df["visitorid"].astype("int64")
    df["itemid"] = df["itemid"].astype("int64")

indeks_visitor["visitorid"] = indeks_visitor[
    "visitorid"
].astype("int64")

indeks_item["itemid"] = indeks_item["itemid"].astype("int64")


indeks_visitor = indeks_visitor.sort_values("indeks")
indeks_item = indeks_item.sort_values("indeks")

# Pemetaan ID asli ke indeks matriks
visitor_to_idx = dict(
    zip(indeks_visitor["visitorid"], indeks_visitor["indeks"].astype(int))
)

item_to_idx = dict(
    zip(indeks_item["itemid"], indeks_item["indeks"].astype(int))
)

# Pemetaan indeks matriks ke ID asli
visitor_ids = indeks_visitor["visitorid"].to_numpy()
item_ids = indeks_item["itemid"].to_numpy()

# ============================================================
# 3. MEMUAT MATRIKS VSM USER
# ============================================================

X_user = load_npz(FILE_VSM).tocsr()

print("\nBentuk matriks VSM User:", X_user.shape)

if X_user.shape[0] != len(visitor_ids):
    raise ValueError(
        "Jumlah baris VSM tidak cocok dengan indeks visitor. "
        "Periksa urutan dan isi indeks_visitor.csv."
    )

if X_user.shape[1] != len(item_ids):
    raise ValueError(
        "Jumlah kolom VSM tidak cocok dengan indeks item. "
        "Periksa urutan dan isi indeks_item.csv."
    )

# Transpose: baris = item, kolom = pengguna
# Normalisasi diperlukan untuk menghitung cosine similarity.
X_item = normalize(
    X_user.T.tocsr(),
    norm="l2",
    axis=1
).tocsr()

print("Bentuk matriks item:", X_item.shape)
print("Matriks VSM siap digunakan.")

# ============================================================
# 4. MEMBENTUK RIWAYAT INTERAKSI TRAINING
# ============================================================

train["bobot"] = train["event"].map(BOBOT_EVENT)
train = train.dropna(subset=["bobot"])
train["bobot"] = train["bobot"].astype(float)

riwayat_train = (
    train.groupby(
        ["visitorid", "itemid"],
        sort=False,
        as_index=False
    )["bobot"]
    .sum()
)

# Riwayat disimpan sebagai:
# visitorid -> [(itemid, bobot), ...]
riwayat_bobot = {}

for visitor_id, item_id, bobot in riwayat_train[
    ["visitorid", "itemid", "bobot"]
].itertuples(index=False, name=None):
    if visitor_id not in riwayat_bobot:
        riwayat_bobot[visitor_id] = []

    riwayat_bobot[visitor_id].append(
        (int(item_id), float(bobot))
    )

print("Riwayat pengguna berhasil dibentuk.")
print("Jumlah pengguna dengan riwayat:", len(riwayat_bobot))

# ============================================================
# 5. MENENTUKAN ITEM TETANGGA TERDEKAT
# ============================================================

# Cache menyimpan tetangga yang sudah dihitung.
# Dengan demikian, item yang sama tidak perlu dihitung ulang.
cache_tetangga = {}

def tetangga_item(idx_item, k=5):
    """
    Mengembalikan maksimal k item terdekat berdasarkan
    cosine similarity dari matriks item-user sparse.
    """

    key = (int(idx_item), int(k))

    if key in cache_tetangga:
        return cache_tetangga[key]

    # Hanya menghitung kemiripan item terhadap satu item acuan.
    # Tidak membuat matriks kemiripan seluruh item.
    kemiripan = (
        X_item @ X_item[int(idx_item)].T
    ).tocoo()

    indeks = kemiripan.row
    skor = kemiripan.data

    # Hapus item acuan dari hasil tetangga
    valid = (
        (indeks != idx_item)
        & np.isfinite(skor)
        & (skor > 0)
    )

    indeks = indeks[valid]
    skor = skor[valid]

    if len(skor) == 0:
        cache_tetangga[key] = []
        return []

    # Ambil k kemiripan tertinggi
    if len(skor) > k:
        pilih = np.argpartition(skor, -k)[-k:]
        indeks = indeks[pilih]
        skor = skor[pilih]

    urutan = np.argsort(skor)[::-1]

    hasil = [
        (int(indeks[i]), float(skor[i]))
        for i in urutan
    ]

    cache_tetangga[key] = hasil
    return hasil


# ============================================================
# 6. FUNGSI REKOMENDASI
# ============================================================

def rekomendasikan(
    visitor_id,
    k=5,
    jumlah_rekomendasi=5
):
    """
    Skor kandidat dihitung dari:
    bobot interaksi item historis x cosine similarity.

    Item yang sudah pernah diinteraksikan pengguna tidak
    dimasukkan kembali sebagai rekomendasi.
    """

    riwayat = riwayat_bobot.get(int(visitor_id), [])

    if not riwayat:
        return []

    item_dilihat = {
        item_id for item_id, _ in riwayat
    }

    skor_kandidat = {}

    for item_id, bobot in riwayat:
        idx = item_to_idx.get(int(item_id))

        if idx is None:
            continue

        tetangga = tetangga_item(idx, k=k)

        for idx_mirip, similarity in tetangga:
            item_kandidat = int(item_ids[idx_mirip])

            if item_kandidat in item_dilihat:
                continue

            skor = float(bobot) * float(similarity)

            skor_kandidat[item_kandidat] = (
                skor_kandidat.get(item_kandidat, 0.0)
                + skor
            )

    hasil = sorted(
        skor_kandidat.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return hasil[:jumlah_rekomendasi]


# ============================================================
# 7. MEMBENTUK TARGET EVALUASI DARI DATA TESTING
# ============================================================

# Evaluasi berfokus pada item yang dibeli (transaction)
# selama periode testing.
transaksi_test = test[
    test["event"] == "transaction"
].copy()

target_per_visitor = (
    transaksi_test.groupby("visitorid")["itemid"]
    .agg(lambda x: set(x.astype(int)))
    .to_dict()
)

# Hanya pengguna yang memiliki riwayat training dan target
# transaksi testing serta tercatat pada matriks VSM.
visitor_eligible = [
    vid
    for vid in target_per_visitor
    if vid in riwayat_bobot and vid in visitor_to_idx
]

print("\nJumlah transaksi testing:", len(transaksi_test))
print("Pengguna dengan target evaluasi:", len(visitor_eligible))

if not visitor_eligible:
    raise ValueError(
        "Tidak ada pengguna yang cocok antara riwayat training "
        "dan target transaksi testing."
    )

# Sampel acak agar waktu komputasi lebih terkendali
rng = np.random.default_rng(RANDOM_STATE)

if len(visitor_eligible) > JUMLAH_SAMPEL:
    visitor_sampel = rng.choice(
        visitor_eligible,
        size=JUMLAH_SAMPEL,
        replace=False
    ).tolist()
else:
    visitor_sampel = visitor_eligible

print("Jumlah pengguna yang dievaluasi:", len(visitor_sampel))

# ============================================================
# 8. METRIK HIT@5 DAN NDCG@5
# ============================================================

def hit_at_k(rekomendasi, target, k=5):
    """
    Bernilai 1 jika setidaknya satu item target muncul
    dalam k rekomendasi pertama, selain itu 0.
    """

    top_k = rekomendasi[:k]

    return float(
        any(item in target for item in top_k)
    )


def ndcg_at_k(rekomendasi, target, k=5):
    """
    NDCG dengan relevansi biner:
    item dalam target transaksi = relevan.
    """

    top_k = rekomendasi[:k]

    dcg = 0.0

    for posisi, item_id in enumerate(top_k, start=1):
        relevansi = 1.0 if item_id in target else 0.0
        dcg += relevansi / np.log2(posisi + 1)

    # Nilai ideal mempertimbangkan jumlah target transaksi
    jumlah_relevan_ideal = min(len(target), k)

    if jumlah_relevan_ideal == 0:
        return 0.0

    idcg = sum(
        1.0 / np.log2(posisi + 1)
        for posisi in range(1, jumlah_relevan_ideal + 1)
    )

    return float(dcg / idcg) if idcg > 0 else 0.0


# ============================================================
# 9. GENERATE REKOMENDASI DAN EVALUASI
# ============================================================

hasil_evaluasi = []
hasil_rekomendasi = []

for nomor, visitor_id in enumerate(visitor_sampel, start=1):
    target = target_per_visitor[int(visitor_id)]

    hasil = rekomendasikan(
        visitor_id,
        k=K,
        jumlah_rekomendasi=JUMLAH_REKOMENDASI
    )

    daftar_item = [item_id for item_id, skor in hasil]

    hit = hit_at_k(
        daftar_item,
        target,
        k=JUMLAH_REKOMENDASI
    )

    ndcg = ndcg_at_k(
        daftar_item,
        target,
        k=JUMLAH_REKOMENDASI
    )

    hasil_evaluasi.append({
        "visitorid": int(visitor_id),
        "jumlah_item_target": len(target),
        "jumlah_rekomendasi": len(daftar_item),
        "Hit@5": hit,
        "NDCG@5": ndcg,
    })

    for peringkat, (item_id, skor) in enumerate(hasil, start=1):
        hasil_rekomendasi.append({
            "visitorid": int(visitor_id),
            "peringkat": peringkat,
            "itemid_rekomendasi": int(item_id),
            "skor_rekomendasi": float(skor),
            "item_target_testing": int(item_id in target),
        })

    if nomor % 25 == 0 or nomor == len(visitor_sampel):
        print(
            f"Proses: {nomor}/{len(visitor_sampel)} pengguna | "
            f"Cache tetangga: {len(cache_tetangga)} item"
        )

# ============================================================
# 10. MENYIMPAN HASIL
# ============================================================

df_evaluasi = pd.DataFrame(hasil_evaluasi)
df_rekomendasi = pd.DataFrame(hasil_rekomendasi)

df_evaluasi.to_csv(
    FILE_EVALUASI,
    index=False
)

df_rekomendasi.to_csv(
    FILE_REKOMENDASI,
    index=False
)

# ============================================================
# 11. RINGKASAN HASIL
# ============================================================

print("\n" + "=" * 55)
print("HASIL EVALUASI REKOMENDASI ITEM")
print("=" * 55)

print("Metode             : Item-based KNN")
print("Cosine neighbors k  :", K)
print("Jumlah rekomendasi  :", JUMLAH_REKOMENDASI)
print("Jumlah pengguna     :", len(df_evaluasi))

print(
    "Rata-rata Hit@5    :",
    round(df_evaluasi["Hit@5"].mean(), 4)
)

print(
    "Rata-rata NDCG@5   :",
    round(df_evaluasi["NDCG@5"].mean(), 4)
)

print("\nFile tersimpan:")
print(FILE_EVALUASI)
print(FILE_REKOMENDASI)

print("\nContoh evaluasi:")
print(df_evaluasi.head())

print("\nContoh rekomendasi:")
print(df_rekomendasi.head())

print("\nProgram selesai.")
