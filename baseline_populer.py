
from pathlib import Path
import numpy as np
import pandas as pd

# =====================================================
# 1. PENGATURAN FILE
# =====================================================

FOLDER = Path(__file__).resolve().parent

TRAIN_FILE = FOLDER / "data" / "training.csv"
TEST_FILE = FOLDER / "data" / "testing.csv"
EVALUASI_KNN_FILE = FOLDER / "evaluasi_knn.csv"

OUTPUT_BASELINE = FOLDER / "evaluasi_baseline_populer.csv"
OUTPUT_BANDING = FOLDER / "perbandingan_knn_baseline.csv"

K = 5

BOBOT_EVENT = {
    "view": 1,
    "addtocart": 2,
    "transaction": 3,
}

# =====================================================
# 2. MEMBACA DATA
# =====================================================

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)
evaluasi_knn = pd.read_csv(EVALUASI_KNN_FILE)

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

print("Training:", len(train))
print("Testing :", len(test))

# =====================================================
# 3. MENGHITUNG POPULARITAS ITEM DARI TRAINING
# =====================================================

train["bobot"] = train["event"].map(BOBOT_EVENT)
train = train.dropna(subset=["bobot"])

popularitas = (
    train.groupby("itemid")["bobot"]
    .sum()
    .sort_values(ascending=False)
)

daftar_item_populer = popularitas.index.tolist()

print("Jumlah item populer:", len(daftar_item_populer))
print("\n10 item terpopuler:")
print(popularitas.head(10))

# =====================================================
# 4. RIWAYAT ITEM SETIAP PENGGUNA DARI TRAINING
# =====================================================

item_per_visitor = (
    train.groupby("visitorid")["itemid"]
    .agg(set)
    .to_dict()
)

# Target evaluasi: item yang dibeli pada testing
transaksi_test = test[
    test["event"] == "transaction"
]

target_per_visitor = (
    transaksi_test.groupby("visitorid")["itemid"]
    .agg(set)
    .to_dict()
)

# Gunakan pengguna yang sama dengan evaluasi KNN
visitor_sampel = evaluasi_knn["visitorid"].astype(
    "int64"
).tolist()

# =====================================================
# 5. FUNGSI EVALUASI
# =====================================================

def hit_at_k(rekomendasi, target, k=5):
    return float(
        any(item in target for item in rekomendasi[:k])
    )


def ndcg_at_k(rekomendasi, target, k=5):
    dcg = 0.0

    for posisi, item in enumerate(rekomendasi[:k], start=1):
        if item in target:
            dcg += 1.0 / np.log2(posisi + 1)

    jumlah_relevan = min(len(target), k)

    if jumlah_relevan == 0:
        return 0.0

    idcg = sum(
        1.0 / np.log2(posisi + 1)
        for posisi in range(1, jumlah_relevan + 1)
    )

    return dcg / idcg if idcg > 0 else 0.0


# =====================================================
# 6. MEMBUAT BASELINE DAN MENGEVALUASI
# =====================================================

hasil = []
hasil_rekomendasi = []

for nomor, visitor_id in enumerate(visitor_sampel, start=1):
    target = target_per_visitor.get(visitor_id, set())
    riwayat = item_per_visitor.get(visitor_id, set())

    # Jangan rekomendasikan item yang sudah pernah
    # diinteraksikan pengguna pada periode training.
    rekomendasi = []

    for item_id in daftar_item_populer:
        if item_id not in riwayat:
            rekomendasi.append(int(item_id))

        if len(rekomendasi) == K:
            break

    hit = hit_at_k(rekomendasi, target, K)
    ndcg = ndcg_at_k(rekomendasi, target, K)

    hasil.append({
        "visitorid": visitor_id,
        "jumlah_item_target": len(target),
        "jumlah_rekomendasi": len(rekomendasi),
        "Hit@5": hit,
        "NDCG@5": ndcg,
    })

    for peringkat, item_id in enumerate(rekomendasi, start=1):
        hasil_rekomendasi.append({
            "visitorid": visitor_id,
            "peringkat": peringkat,
            "itemid_rekomendasi": item_id,
            "item_target_testing": int(item_id in target),
        })

    if nomor % 50 == 0 or nomor == len(visitor_sampel):
        print(f"Evaluasi: {nomor}/{len(visitor_sampel)}")

# =====================================================
# 7. MENYIMPAN HASIL
# =====================================================

df_baseline = pd.DataFrame(hasil)
df_baseline.to_csv(OUTPUT_BASELINE, index=False)

pd.DataFrame(hasil_rekomendasi).to_csv(
    FOLDER / "rekomendasi_baseline_populer.csv",
    index=False
)

# =====================================================
# 8. MEMBANDINGKAN DENGAN KNN
# =====================================================

ringkasan = pd.DataFrame([
    {
        "Metode": "Item-based KNN",
        "Hit@5": evaluasi_knn["Hit@5"].mean(),
        "NDCG@5": evaluasi_knn["NDCG@5"].mean(),
    },
    {
        "Metode": "Item Populer",
        "Hit@5": df_baseline["Hit@5"].mean(),
        "NDCG@5": df_baseline["NDCG@5"].mean(),
    },
])

ringkasan["Hit@5"] = ringkasan["Hit@5"].round(4)
ringkasan["NDCG@5"] = ringkasan["NDCG@5"].round(4)

ringkasan.to_csv(OUTPUT_BANDING, index=False)

print("\n=== PERBANDINGAN METODE ===")
print(ringkasan.to_string(index=False))

print("\nFile hasil:")
print(OUTPUT_BASELINE)
print(OUTPUT_BANDING)

print("\nProgram selesai.")
