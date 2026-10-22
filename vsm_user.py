
from neo4j import GraphDatabase
from getpass import getpass
from scipy.sparse import coo_matrix, save_npz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# =========================================
# 1. KONEKSI NEO4J
# =========================================
URI = "neo4j://127.0.0.1:7687"
USERNAME = "neo4j"
DATABASE = "neo4j"
PASSWORD = input("Masukkan password Neo4j: ")

driver = GraphDatabase.driver(
    URI,
    auth=(USERNAME, PASSWORD)
)

try:
    driver.verify_connectivity()
    print("Berhasil terhubung ke Neo4j.")

    with driver.session(database=DATABASE) as session:

        # =========================================
        # 2. AMBIL ID USER DAN ITEM
        # =========================================
        print("\nMengambil daftar Visitor...")

        visitor_ids = [
            record["id"]
            for record in session.run("""
                MATCH (v:Visitor)
                RETURN v.visitorId AS id
                ORDER BY id
            """)
        ]

        print("Jumlah Visitor:", len(visitor_ids))

        print("\nMengambil daftar Item...")

        item_ids = [
            record["id"]
            for record in session.run("""
                MATCH (i:Item)
                RETURN i.itemId AS id
                ORDER BY id
            """)
        ]

        print("Jumlah Item:", len(item_ids))

        # Memetakan ID asli ke indeks matriks
        visitor_index = {
            vid: idx for idx, vid in enumerate(visitor_ids)
        }

        item_index = {
            iid: idx for idx, iid in enumerate(item_ids)
        }

        # =========================================
        # 3. AMBIL INTERAKSI DAN BENTUK VSM
        # =========================================
        print("\nMengambil interaksi dari Neo4j...")

        query = """
        MATCH (v:Visitor)-[r]->(i:Item)
        RETURN
            v.visitorId AS visitor_id,
            i.itemId AS item_id,
            type(r) AS jenis
        """

        baris = []
        kolom = []
        nilai = []

        bobot = {
            "VIEWED": 1.0,
            "ADDED_TO_CART": 2.0,
            "BOUGHT": 3.0
        }

        
        result = session.run(query, fetch_size=10000)

        for record in result:
            vid = record["visitor_id"]
            iid = record["item_id"]
            jenis = record["jenis"]

            baris.append(visitor_index[vid])
            kolom.append(item_index[iid])
            nilai.append(bobot[jenis])


        # COO memungkinkan penyimpanan efisien.
        # Pasangan Visitor-Item yang berulang akan dijumlahkan.
        X = coo_matrix(
            (
                np.array(nilai, dtype=np.float32),
                (
                    np.array(baris, dtype=np.int32),
                    np.array(kolom, dtype=np.int32)
                )
            ),
            shape=(len(visitor_ids), len(item_ids)),
            dtype=np.float32
        ).tocsr()

        X.sum_duplicates()

        print("\n=== HASIL VSM USER ===")
        print("Dimensi matriks:", X.ndim)
        print("Shape matriks:", X.shape)
        print("Jumlah nilai bukan nol:", X.nnz)
        print("Kepadatan matriks:",
              f"{X.nnz / (X.shape[0] * X.shape[1]):.8%}")

        # Simpan matriks dan indeks ID
        save_npz("vsm_user.npz", X)

        pd.Series(visitor_ids, name="visitorId").to_csv(
            "indeks_visitor.csv", index_label="indeks"
        )

        pd.Series(item_ids, name="itemId").to_csv(
            "indeks_item.csv", index_label="indeks"
        )

        print("\nFile berhasil disimpan:")
        print("- vsm_user.npz")
        print("- indeks_visitor.csv")
        print("- indeks_item.csv")

        # =========================================
        # 4. VISUALISASI MATRIKS SAMPEL
        # =========================================
        # Ambil maksimal 30 Visitor dan 30 Item pertama
        sampel = X[:30, :30].toarray()

        plt.figure(figsize=(10, 6))
        plt.imshow(sampel, aspect="auto", interpolation="nearest")
        plt.xlabel("Indeks Item (30 pertama)")
        plt.ylabel("Indeks Visitor (30 pertama)")
        plt.title("Matriks VSM User–Item (Sampel 30 × 30)")
        plt.colorbar(label="Bobot interaksi")
        plt.tight_layout()
        plt.savefig("vsm_user_sampel.png", dpi=150)
        plt.show()

        # =========================================
        # 5. VISUALISASI NETWORKX SAMPEL
        # =========================================
        # Graf kecil untuk visualisasi, bukan seluruh graf.
        G = nx.Graph()

        jumlah_user = min(10, len(visitor_ids))
        user_sampel = set(visitor_ids[:jumlah_user])

        query_graf = """
        MATCH (v:Visitor)-[r]->(i:Item)
        WHERE v.visitorId IN $ids
        RETURN
            v.visitorId AS visitor_id,
            i.itemId AS item_id,
            type(r) AS jenis
        LIMIT 100
        """

        hasil = session.run(
            query_graf,
            ids=list(user_sampel)
        )

        for record in hasil:
            u = "V_" + str(record["visitor_id"])
            i = "I_" + str(record["item_id"])

            G.add_node(u, tipe="Visitor")
            G.add_node(i, tipe="Item")
            G.add_edge(u, i, jenis=record["jenis"])

        posisi = nx.spring_layout(G, seed=42)

        warna = [
            "skyblue" if G.nodes[n]["tipe"] == "Visitor"
            else "orange"
            for n in G.nodes
        ]

        plt.figure(figsize=(12, 8))
        nx.draw(
            G,
            posisi,
            with_labels=False,
            node_color=warna,
            node_size=100,
            edge_color="gray",
            width=0.6
        )

        plt.title("Graf Bipartit Visitor–Item (Sampel)")
        plt.savefig("networkx_user_sampel.png", dpi=150)
        plt.show()

        print("\nVisualisasi disimpan:")
        print("- vsm_user_sampel.png")
        print("- networkx_user_sampel.png")

finally:
    driver.close()
