
import pandas as pd
from pathlib import Path

folder = Path(__file__).resolve().parent

evaluasi = pd.read_csv(folder / "evaluasi_knn.csv")
rekomendasi = pd.read_csv(folder / "rekomendasi_knn.csv")

print("=== PEMERIKSAAN EVALUASI KNN ===")
print("Jumlah pengguna:", evaluasi["visitorid"].nunique())
print("Baris evaluasi:", len(evaluasi))
print("Pengguna dengan rekomendasi kurang dari 5:",
      (evaluasi["jumlah_rekomendasi"] < 5).sum())
print("Rekomendasi duplikat per pengguna:",
      rekomendasi.duplicated(
          subset=["visitorid", "itemid_rekomendasi"]
      ).sum())

jumlah_cocok = rekomendasi.groupby("visitorid")[
    "item_target_testing"
].max()

print("Pengguna dengan minimal satu item cocok:",
      int(jumlah_cocok.sum()))

print("Hit@5 dari file:", round(evaluasi["Hit@5"].mean(), 4))
print("NDCG@5 dari file:", round(evaluasi["NDCG@5"].mean(), 4))

print("\nDistribusi Hit@5:")
print(evaluasi["Hit@5"].value_counts().sort_index())

print("\nPemeriksaan selesai.")
