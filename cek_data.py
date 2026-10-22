import pandas as pd

df = pd.read_csv("data/events.csv")

print("Nama kolom:")
print(df.columns.tolist())

print("\nLima baris pertama:")
print(df.head())

print("\nJenis interaksi:")
print(df["event"].value_counts())


# Konversi Unix epoch ke tanggal
df["tanggal"] = pd.to_datetime(
    df["timestamp"],
    unit="ms",
    utc=True
)

print("Tanggal paling awal:", df["tanggal"].min())
print("Tanggal paling akhir:", df["tanggal"].max())

print("\nJumlah data per tahun:")
print(df["tanggal"].dt.year.value_counts().sort_index())