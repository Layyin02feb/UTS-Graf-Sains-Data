
import os
import getpass
import pandas as pd
from neo4j import GraphDatabase

# ==============================
# 1. KONEKSI KE NEO4J
# ==============================
URI = "neo4j://127.0.0.1:7687"
USERNAME = "neo4j"
PASSWORD = input("Masukkan password Neo4j: ")

driver = GraphDatabase.driver(
    URI,
    auth=(USERNAME, PASSWORD)
)

# ==============================
# 2. BACA HASIL SIMILARITY
# ==============================
FILE = "similarity_item_top5.csv"

if not os.path.exists(FILE):
    raise FileNotFoundError(f"File {FILE} tidak ditemukan.")

df = pd.read_csv(FILE)

# Pastikan ID item menjadi integer
df["item_acuan"] = df["item_acuan"].astype(int)
df["item_mirip"] = df["item_mirip"].astype(int)
df["cosine_similarity"] = df["cosine_similarity"].astype(float)

data = df.to_dict("records")

# ==============================
# 3. SIMPAN RELASI KE NEO4J
# ==============================
query = """
UNWIND $data AS row
MATCH (a:Item {itemId: row.item_acuan})
MATCH (b:Item {itemId: row.item_mirip})
MERGE (a)-[r:SIMILAR_TO]->(b)
SET r.similarity = row.cosine_similarity
RETURN count(r) AS jumlah_relasi
"""

try:
    with driver.session(database="neo4j") as session:
        # Pastikan semua item ditemukan sebelum menyimpan
        cek = session.run("""
            UNWIND $data AS row
            OPTIONAL MATCH (a:Item {itemId: row.item_acuan})
            OPTIONAL MATCH (b:Item {itemId: row.item_mirip})
            RETURN
                count(*) AS jumlah_baris,
                sum(CASE WHEN a IS NULL THEN 1 ELSE 0 END)
                    AS item_acuan_tidak_ditemukan,
                sum(CASE WHEN b IS NULL THEN 1 ELSE 0 END)
                    AS item_mirip_tidak_ditemukan
        """, data=data).single()

        print("Pemeriksaan item:", dict(cek))

        if (
            cek["item_acuan_tidak_ditemukan"] > 0
            or cek["item_mirip_tidak_ditemukan"] > 0
        ):
            raise ValueError(
                "Ada item yang tidak ditemukan di Neo4j. "
                "Periksa properti ID pada node Item."
            )

        hasil = session.run(query, data=data).single()
        print("Jumlah relasi diproses:", hasil["jumlah_relasi"])

    print("Relasi similarity berhasil disimpan.")

finally:
    driver.close()
