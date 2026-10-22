
from pathlib import Path
import numpy as np
import pandas as pd

FOLDER = Path(__file__).resolve().parent

file_eval = FOLDER / "evaluasi_knn.csv"
file_rekom = FOLDER / "rekomendasi_knn.csv"
file_test = FOLDER / "data" / "testing.csv"

# Membaca hasil yang sudah tersedia
evaluasi = pd.read_csv(file_eval)
rekomendasi = pd.read_csv(file_rekom)
testing = pd.read_csv(file_test)

# Target relevan: item yang dibeli pengguna pada periode testing
testing["event"] = testing["event"].astype(str).str.lower()
testing["visitorid"] = pd.to_numeric(
    testing["visitorid"], errors="coerce"
)
testing["itemid"] = pd.to_numeric(
    testing["itemid"], errors="coerce"
)

transaksi = testing[testing["event"] == "transaction"].dropna(
    subset=["visitorid", "itemid"]
)

target = (
    transaksi.groupby("visitorid")["itemid"]
    .agg(lambda x: set(x.astype(int)))
    .to_dict()
)

# Pilih satu pengguna secara acak dengan seed tetap
pengguna = evaluasi["visitorid"].astype(int).tolist()
rng = np.random.default_rng(42)
user_id = int(rng.choice(pengguna))

# Ambil maksimal 5 rekomendasi untuk pengguna tersebut
rec_user = (
    rekomendasi[rekomendasi["visitorid"] == user_id]
    .sort_values("peringkat")
    .head(5)
    .copy()
)

item_target = target.get(user_id, set())
daftar_item = rec_user["itemid_rekomendasi"].astype(int).tolist()

# Hit@5 untuk satu pengguna
hit = float(any(item in item_target for item in daftar_item))

# NDCG@5 untuk satu pengguna
dcg = sum(
    (1.0 / np.log2(posisi + 1))
    for posisi, item in enumerate(daftar_item, start=1)
    if item in item_target
)

jumlah_relevan_ideal = min(len(item_target), 5)

idcg = sum(
    1.0 / np.log2(posisi + 1)
    for posisi in range(1, jumlah_relevan_ideal + 1)
)

ndcg = dcg / idcg if idcg > 0 else 0.0

print("=" * 50)
print("HASIL REKOMENDASI UNTUK SATU PENGGUNA")
print("=" * 50)
print("Random seed :", 42)
print("User ID     :", user_id)
print("Target transaksi testing:", sorted(item_target))

print("\nDaftar rekomendasi:")
print(
    rec_user[
        ["peringkat", "itemid_rekomendasi", "skor_rekomendasi"]
    ].to_string(index=False)
)

print("\nHit@5  :", round(hit, 4))
print("NDCG@5 :", round(ndcg, 4))
print("Jumlah rekomendasi:", len(daftar_item))
print("Jumlah item target:", len(item_target))
