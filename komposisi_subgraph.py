
from neo4j import GraphDatabase
import pandas as pd

# Konfigurasi koneksi Neo4j
URI = "neo4j://127.0.0.1:7687"
USERNAME = "neo4j"
DATABASE = "neo4j"

# Masukkan password database Neo4j milikmu
PASSWORD = input("Masukkan password Neo4j: ")

driver = GraphDatabase.driver(
    URI,
    auth=(USERNAME, PASSWORD)
)


def ambil_komposisi(tx):
    # Query 1: menghitung jumlah setiap jenis node
    query_node = """
    MATCH (n)
    RETURN labels(n) AS jenis_node, count(n) AS jumlah
    ORDER BY jenis_node
    """

    hasil_node = list(tx.run(query_node))

    # Query 2: menghitung jumlah setiap jenis relasi
    query_relasi = """
    MATCH ()-[r]->()
    RETURN type(r) AS jenis_relasi, count(r) AS jumlah
    ORDER BY jenis_relasi
    """

    hasil_relasi = list(tx.run(query_relasi))

    return hasil_node, hasil_relasi


try:
    driver.verify_connectivity()
    print("Berhasil terhubung ke Neo4j.")

    with driver.session(database=DATABASE) as session:
        node, relasi = session.execute_read(ambil_komposisi)

    # Tampilkan hasil dalam tabel
    df_node = pd.DataFrame(
        [dict(baris) for baris in node]
    )

    df_relasi = pd.DataFrame(
        [dict(baris) for baris in relasi]
    )

    print("\n=== KOMPOSISI NODE ===")
    print(df_node.to_string(index=False))

    print("\n=== KOMPOSISI RELASI ===")
    print(df_relasi.to_string(index=False))

    # Simpan hasil untuk laporan
    df_node.to_csv(
        "komposisi_node.csv",
        index=False
    )

    df_relasi.to_csv(
        "komposisi_relasi.csv",
        index=False
    )

    print("\nHasil tersimpan:")
    print("- komposisi_node.csv")
    print("- komposisi_relasi.csv")

finally:
    driver.close()
