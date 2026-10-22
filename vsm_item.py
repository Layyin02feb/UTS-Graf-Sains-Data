
from scipy.sparse import load_npz, save_npz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# =========================================
# 1. MEMUAT VSM USER
# =========================================
print("Memuat matriks VSM User...")

X_user = load_npz("vsm_user.npz")

visitor_df = pd.read_csv("indeks_visitor.csv")
item_df = pd.read_csv("indeks_item.csv")

print("Shape VSM User:", X_user.shape)

# =========================================
# 2. MEMBENTUK VSM ITEM
# =========================================
# Transpose: baris menjadi Item, kolom menjadi Visitor
X_item = X_user.T.tocsr()

print("\n=== HASIL VSM ITEM ===")
print("Dimensi matriks:", X_item.ndim)
print("Shape matriks:", X_item.shape)
print("Jumlah nilai bukan nol:", X_item.nnz)

total_elemen = X_item.shape[0] * X_item.shape[1]
kepadatan = X_item.nnz / total_elemen

print(f"Kepadatan matriks: {kepadatan:.8%}")

# =========================================
# 3. STATISTIK INTERAKSI SETIAP ITEM
# =========================================
jumlah_user_per_item = np.diff(X_item.indptr)

statistik_item = pd.DataFrame({
    "itemId": item_df["itemId"].to_numpy(),
    "jumlah_user_berinteraksi": jumlah_user_per_item,
    "total_bobot_interaksi": np.asarray(
        X_item.sum(axis=1)
    ).ravel()
})

statistik_item = statistik_item.sort_values(
    "jumlah_user_berinteraksi",
    ascending=False
)

print("\n=== 10 ITEM DENGAN USER TERBANYAK ===")
print(statistik_item.head(10).to_string(index=False))

statistik_item.to_csv(
    "statistik_vsm_item.csv",
    index=False
)

# =========================================
# 4. SIMPAN VSM ITEM
# =========================================
save_npz("vsm_item.npz", X_item)

print("\nFile tersimpan:")
print("- vsm_item.npz")
print("- statistik_vsm_item.csv")

# =========================================
# 5. VISUALISASI MATRIKS SAMPEL
# =========================================
sampel = X_item[:30, :30].toarray()

plt.figure(figsize=(10, 6))
plt.imshow(
    sampel,
    aspect="auto",
    interpolation="nearest"
)
plt.xlabel("Indeks Visitor (30 pertama)")
plt.ylabel("Indeks Item (30 pertama)")
plt.title("Matriks VSM Item–User (Sampel 30 x 30)")
plt.colorbar(label="Bobot interaksi")
plt.tight_layout()
plt.savefig("vsm_item_sampel.png", dpi=150)
plt.show()

# =========================================
# 6. VISUALISASI NETWORKX
#    Gunakan item dengan jumlah user terbanyak
# =========================================
# Ambil 10 item dengan jumlah user terbanyak
item_teratas = statistik_item.head(10)["itemId"].tolist()

# Buat pemetaan ID Item ke indeks matriks
item_index = {
    item_id: idx
    for idx, item_id in enumerate(item_df["itemId"])
}

visitor_ids = visitor_df["visitorId"].to_numpy()

G = nx.Graph()

for item_id in item_teratas:
    idx_item = item_index[item_id]
    awal = X_item.indptr[idx_item]
    akhir = X_item.indptr[idx_item + 1]

    indeks_user = X_item.indices[awal:akhir]
    bobot_user = X_item.data[awal:akhir]

    node_item = "I_" + str(item_id)
    G.add_node(node_item, tipe="Item")

    # Batasi jumlah relasi agar visualisasi terbaca
    for idx_user, bobot in list(
        zip(indeks_user, bobot_user)
    )[:20]:
        user_id = visitor_ids[idx_user]
        node_user = "V_" + str(user_id)

        G.add_node(node_user, tipe="Visitor")
        G.add_edge(
            node_item,
            node_user,
            weight=float(bobot)
        )

posisi = nx.spring_layout(G, seed=42)

warna = [
    "orange" if G.nodes[n]["tipe"] == "Item"
    else "skyblue"
    for n in G.nodes
]

plt.figure(figsize=(14, 9))
nx.draw(
    G,
    posisi,
    with_labels=False,
    node_color=warna,
    node_size=100,
    edge_color="gray",
    width=0.6
)

plt.title(
    "Graf Bipartit Item–Visitor "
    "(10 Item Teratas, Maksimal 20 User per Item)"
)
plt.savefig("networkx_item_sampel.png", dpi=150)
plt.show()

print("\nVisualisasi tersimpan:")
print("- vsm_item_sampel.png")
print("- networkx_item_sampel.png")
