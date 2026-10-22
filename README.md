# UTS-Graf-Sains-Data

## Deskripsi
Proyek ini menggunakan dataset RetailRocket untuk membangun graf interaksi pengguna dan barang di Neo4j, menerapkan Vector Space Model (VSM), menghitung kemiripan antarbarang menggunakan cosine similarity, dan membuat rekomendasi menggunakan item-based KNN.

## Dataset
Dataset: RetailRocket E-commerce Dataset.

Data interaksi terdiri dari tiga jenis aktivitas:
- `view`: melihat barang.
- `addtocart`: menambahkan barang ke keranjang.
- `transaction`: transaksi pembelian.

Dataset mentah tidak disertakan dalam repository apabila ukurannya terlalu besar. Sesuaikan dengan file yang benar-benar diunggah.

## Pembagian Data
| Data | Periode | Jumlah records |
|---|---|---:|
| Training | 4 Mei–23 Juli 2015 | 1.689.124 |
| Testing | 24 Juli–13 Agustus 2015 | 431.332 |

## Teknologi
- Python
- pandas dan SciPy
- scikit-learn
- NetworkX
- Neo4j dan Cypher

## Tahapan Analisis
1. Memfilter data berdasarkan timestamp.
2. Membuat graf interaksi Visitor dan Item di Neo4j.
3. Menghitung komposisi simpul dan relasi.
4. Membuat VSM User dan VSM Item.
5. Membandingkan proyeksi User–User dan Item–Item.
6. Menghitung cosine similarity antarbarang.
7. Membuat rekomendasi menggunakan item-based KNN.
8. Mengevaluasi rekomendasi dengan Hit@5 dan NDCG@5.

## File Program Utama
- `bagi_periode.py`: memisahkan data training dan testing.
- `import_retailrocket.py`: memasukkan data training ke Neo4j.
- `komposisi_subgraph.py`: menghitung komposisi graf.
- `vsm_user.py`: membuat VSM User.
- `vsm_item.py`: membuat VSM Item.
- `bandingkan_proyeksi.py`: membandingkan proyeksi User–User dan Item–Item.
- `similarity_item.py`: menghitung similarity antarbarang.
- `simpan_similarity_neo4j.py`: menyimpan relasi similarity ke Neo4j.
- `rekomendasi_knn.py`: menghasilkan rekomendasi KNN.
- `pilih_user_acak.py`: memilih user secara reproducible dan mengevaluasi rekomendasi.

## Hasil Utama
- Jumlah simpul Visitor: 857.874.
- Jumlah simpul Item: 190.822.
- Jumlah relasi interaksi: 1.689.124.
- Proyeksi sampel User–User: 34 pasangan similarity positif.
- Proyeksi sampel Item–Item: 342 pasangan similarity positif.
- Evaluasi KNN: Hit@5 = 0,0356 dan NDCG@5 = 0,0188.
- Evaluasi baseline popularitas: Hit@5 = 0,0178 dan NDCG@5 = 0,0067.

## Menjalankan Program
1. Instal Python dan pustaka yang dibutuhkan.
2. Siapkan dataset RetailRocket pada folder `data/`.
3. Jalankan `bagi_periode.py` untuk membuat data training dan testing.
4. Jalankan Neo4j dan sesuaikan konfigurasi koneksi secara lokal.
5. Jalankan skrip analisis sesuai urutan tahapan.

Catatan: Jangan menjalankan ulang `import_retailrocket.py` pada database yang sudah terisi tanpa membersihkan atau memeriksa data terlebih dahulu karena skrip menggunakan CREATE.

## Keamanan
Jangan mengunggah password Neo4j, token, atau kredensial ke repository.

## Bukti Visualisasi
Screenshot graf sebelum dan sesudah similarity serta visualisasi VSM dapat dilihat pada folder screenshot jika sudah disertakan.
