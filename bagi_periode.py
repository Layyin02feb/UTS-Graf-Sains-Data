
import pandas as pd
from pathlib import Path

# Lokasi file
file_input = Path("data/events.csv")
folder_output = Path("data")

# Periksa apakah file tersedia
if not file_input.exists():
    raise FileNotFoundError(
        f"File tidak ditemukan: {file_input}. "
        "Pastikan terminal berada di folder proyek."
    )

# Baca data
df = pd.read_csv(file_input)

print("Jumlah data awal:", len(df))

# Ubah timestamp Unix milidetik menjadi tanggal UTC
df["tanggal"] = pd.to_datetime(
    df["timestamp"],
    unit="ms",
    utc=True
)

# Tampilkan rentang tanggal dataset
print("Tanggal paling awal:", df["tanggal"].min())
print("Tanggal paling akhir:", df["tanggal"].max())

# Periode training: 4 Mei sampai 23 Juli 2015
training = df[
    (df["tanggal"] >= "2015-05-04") &
    (df["tanggal"] < "2015-07-24")
].copy()

# Periode testing: 24 Juli sampai 13 Agustus 2015
testing = df[
    (df["tanggal"] >= "2015-07-24") &
    (df["tanggal"] < "2015-08-14")
].copy()

# Hapus kolom tanggal tambahan karena tidak diperlukan
# Kolom timestamp asli tetap dipertahankan
training.drop(columns=["tanggal"], inplace=True)
testing.drop(columns=["tanggal"], inplace=True)

# Simpan hasil ke CSV
training.to_csv(
    folder_output / "training.csv",
    index=False
)

testing.to_csv(
    folder_output / "testing.csv",
    index=False
)

# Tampilkan ringkasan
print("\n=== HASIL PEMBAGIAN DATA ===")
print("Jumlah data training:", len(training))
print("Jumlah data testing:", len(testing))
print("File training:", folder_output / "training.csv")
print("File testing:", folder_output / "testing.csv")

print("\nJumlah event training:")
print(training["event"].value_counts())

print("\nJumlah event testing:")
print(testing["event"].value_counts())
