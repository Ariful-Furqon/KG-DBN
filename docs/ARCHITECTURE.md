# Arsitektur KG-DBN

Dokumen ini menjelaskan desain pipeline KG-DBN, kontrak antarmodul, dan status tiap bagian. Tujuannya supaya pekerjaan bisa dibagi ke beberapa orang atau agen tanpa saling bentrok. Setiap paket kerja hanya menyentuh modulnya sendiri dan berkomunikasi lewat kontrak di bawah.

## 1. Masalah

Diagnosis hama/penyakit padi dari gejala yang diamati petani atau penyuluh.

- **Masukan:** himpunan gejala dari satu kasus (individu kelas `Gejala` di ontologi).
- **Keluaran:** satu label, yaitu individu `PenyakitPadi` atau `HamaPadi`.
- **Hipotesis penelitian:** embedding dari knowledge graph (relasi gejala–penyakit–hama–patogen–pengendalian) yang dipakai sebagai fitur Deep Belief Network meningkatkan akurasi dibanding fitur gejala mentah (multi-hot) dan dibanding jaringan tanpa pretraining RBM.

## 2. Alur data

```
Ontologi (RDF/XML)                         data/kasus.csv
        │                                        │
        ▼                                        │
[ontology.py] load_ontology ──► KnowledgeGraph ──┤
        │                          │             ▼
        │                          │    [cases.py] load_cases ──► cases (DataFrame)
        │                          ▼                                 │
        │               [embedding.py] node2vec                      │
        │                          │                                 │
        │                          ▼                                 ▼
        │                 {entity_id: vector} ──► [cases.py] multi_hot / kg_features
        │                                                            │
        │                                                            ▼
        │                                                 X (fitur) , y (label)
        │                                                            │
        │                                                            ▼
        │                                  [dbn.py] DBN: RBM pretraining ─► fine-tuning
        │                                                            │
        ▼                                                            ▼
                              [experiment.py] run: split, baseline, ablation, metrik
                                                             │
                                                             ▼
                                                  results/results.csv
```

## 3. Struktur repo

```
KG-DBN/
├── Ontologi/OntologiHamaPenyakitPadi_v1.1.rdf   # sumber pengetahuan (read-only, versi baru = file baru)
├── kgdbn/
│   ├── ontology.py     # RDF → KnowledgeGraph (+ validasi skema)
│   ├── embedding.py    # node2vec (random walk + skip-gram, PyTorch)
│   ├── cases.py        # loader CSV kasus + encoding fitur
│   ├── literature.py   # lembar ekstraksi literatur → kasus.csv
│   ├── dbn.py          # RBM + DBN
│   └── experiment.py   # orkestrasi eksperimen & CLI
├── tests/test_pipeline.py
├── docs/ARCHITECTURE.md
├── docs/DATA_PROTOCOL.md  # protokol pengumpulan data kasus (P2)
├── templates/          # template CSV ekstraksi & sumber (di-commit)
├── data/               # data kasus (di-.gitignore, tidak masuk repo)
├── results/            # keluaran eksperimen
└── KG-DBN.ipynb        # notebook eksplorasi, hanya memanggil kgdbn
```

Aturan: logika hanya ditulis di `kgdbn/`. Notebook cukup memanggil package, jangan menyalin kode ke notebook.

## 4. Modul dan kontraknya

### 4.1 `ontology.py`: ontologi → graf

```python
load_ontology(path) -> KnowledgeGraph
KnowledgeGraph.entities: dict[str, Entity]        # Entity(id, type, label, abstract)
KnowledgeGraph.triples:  list[(subj_id, relasi, obj_id)]   # hanya triple valid
KnowledgeGraph.dropped:  list[(subj_id, relasi, obj_id)]   # melanggar SCHEMA / self-loop
KnowledgeGraph.of_type(*types) -> list[str]
KnowledgeGraph.symptom_profiles(target_types) -> dict[target_id, list[gejala_id]]
```

- `id` adalah local name IRI (mis. `Blas`, `Lesi_pada_Daun`). Semua modul lain memakai `id` ini sebagai kunci.
- Validasi memakai tabel `SCHEMA` di kode, bukan `rdfs:domain`, karena domain di v1.1 salah (semuanya `Padi`).
- Di v1.1 ada 7 triple yang dibuang (lihat `kg.dropped`).

### 4.2 `embedding.py`: graf → vektor

```python
node2vec(kg, dim=32, num_walks=20, walk_length=20, window=5, p=1, q=1,
         exclude_relations=(), seed=42) -> dict[entity_id, np.ndarray(dim)]
```

- Graf diperlakukan tak berarah dan tak berbobot, dengan semua tipe relasi digabung.
- `exclude_relations` dipakai untuk ablation, misalnya menyembunyikan `memilikiGejala`.
- Implementasi sendiri di PyTorch (tanpa gensim, karena belum ada wheel untuk Python 3.14).
- Cara embedding bisa diganti (TransE, RotatE, R-GCN, embedding teks `abstract`) **asalkan keluarannya tetap `dict[entity_id, vector]`**.

### 4.3 `cases.py`: kasus → fitur

Format `data/kasus.csv`:

```
gejala,label
Lesi_pada_Daun;Lesi_pada_batang,Blas
Daun menguning;Layu,WerengBatangCoklat
```

```python
load_cases(path, kg, target_types) -> DataFrame[symptoms: list[gejala_id], label: target_id]
multi_hot(cases, vocabulary)       -> np.ndarray (n, |Gejala|), {0,1}
kg_features(cases, embeddings)     -> np.ndarray (n, dim), rata-rata embedding gejala
```

- Nama boleh berupa `id` atau `rdfs:label`. Pencocokan tidak peka huruf besar/kecil, spasi, dan underscore.
- Nama yang tidak dikenal langsung membuat `ValueError` yang menyebutkan barisnya. Tidak ada nama yang dibuang diam-diam.
- **Tidak ada data sintetis.** Semua kasus berasal dari data nyata.

### 4.4 `dbn.py`: model

```python
RBM(n_visible, n_hidden, visible="bernoulli"|"gaussian")
DBN(hidden_layers=(128, 64), visible=..., k=1, pretrain=True, ...)
    .fit(X, y, X_val=None, y_val=None) -> self
    .predict(X) -> labels
    .predict_proba(X) -> (n, n_classes)
    .history_ = {"pretrain": [[recon_err/epoch] per RBM], "train_loss": [...], "val_loss": [...]}
```

- **Pretraining:** RBM ditumpuk satu per satu (greedy, layer-wise) dan dilatih dengan CD-k. Gradien diturunkan dari selisih *free energy* `F(v0) - F(vk)`.
- **Visible layer:** `bernoulli` untuk masukan di [0,1] (multi-hot), `gaussian` untuk masukan yang sudah distandarkan (embedding KG). Layer di atasnya selalu Bernoulli.
- **Fine-tuning:** bobot RBM menjadi inisialisasi MLP sigmoid, ditambah layer softmax, dilatih dengan cross-entropy + Adam dan early stopping pada val loss.
- `pretrain=False` memberi arsitektur yang sama dengan bobot acak. Ini baseline "MLP tanpa pretraining" untuk membuktikan kontribusi RBM.
- API-nya meniru scikit-learn (`fit`/`predict`), jadi bisa dipakai di mana pun model sklearn dipakai.

### 4.5 `experiment.py`: orkestrasi

```bash
python -m kgdbn.experiment --cases data/kasus.csv [--targets PenyakitPadi] [--dim 32] [--hidden 128 64]
```

- Split train/test stratified (20% uji).
- Matriks eksperimen: fitur {multi-hot, KG, multi-hot + KG} × model {KG-DBN, MLP tanpa pretraining, Random Forest, Bernoulli NB (khusus multi-hot)}.
- Metrik: accuracy dan F1-macro. Ada juga **batas atas akurasi**, yaitu akurasi maksimum yang bisa dicapai kalau hanya melihat himpunan gejala.
- Keluaran: `results/results.csv`.

### 4.6 `neo4j_io.py`: Neo4j (opsional)

```python
get_driver(uri=None, user=None, password=None, env_path=".env") -> neo4j.Driver
export_graph(kg, driver, replace=False) -> None
gds_embeddings(driver, method="fastRP"|"node2vec", dim=32, seed=42,
               relation_types=None) -> dict[entity_id, np.ndarray(dim)]
```

- Kredensial dibaca dari `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` (environment atau `.env`), tidak pernah ditulis di kode.
- Setiap entitas menjadi node berlabel `:KGEntity:<tipe>` dengan properti `id`, `label`, `abstract`. Ada constraint unik pada `KGEntity.id`. Setiap triple menjadi relasi bertipe nama relasinya.
- `export_graph` memakai `MERGE`, jadi aman dijalankan berulang, tapi **tidak pernah menghapus**. Pakai `replace=True` saat mengekspor versi ontologi baru supaya entitas dan relasi yang sudah dihapus tidak tertinggal. Opsi ini menghapus semua node `:KGEntity`; node lain di database tidak disentuh.
- `gds_embeddings` hanya memproyeksikan node `:KGEntity` sebagai graf tak berarah (sama dengan `node2vec` di §4.2). `relation_types=None` berarti semua relasi. Keluarannya memenuhi kontrak §4.2, jadi bisa langsung dipakai `kg_features`.
- Paket `neo4j` opsional dan hanya di-import saat fungsi dipanggil.

### 4.7 `literature.py`: lembar ekstraksi → kasus

```python
resolve_extraction(raw, kg)          -> DataFrame (+ a1, a2, final: list[gejala_id] | None, label_id)
symptom_kappa(resolved, vocabulary)  -> (kappa | None, n_kasus)   # A1 vs A2, matriks kasus × Gejala
select_cases(resolved, kg)           -> (DataFrame kasus dipakai, Counter alasan eksklusi)
export_cases(kept, path)             # format §4.3 + kolom jejak (id_kasus, id_sumber, ...)
prisma_counts(sources)               -> {tahap: jumlah sumber}
```

- Kolom dan aturan seleksi dijelaskan di [DATA_PROTOCOL.md](DATA_PROTOCOL.md). Nilai tidak valid langsung `ValueError` dengan nomor baris.
- `kasus.csv` hasil ekspor punya `id_sumber`, supaya evaluasi (P5) bisa memakai split per sumber.

## 5. Keputusan desain

| Keputusan | Alasan |
|---|---|
| Ontologi dibaca langsung dengan rdflib, Neo4j opsional | Pipeline bisa jalan dan dites tanpa server. Neo4j berguna untuk visualisasi dan GDS, tapi bukan syarat. |
| Id entitas = local name IRI | Stabil, unik, dan sama dengan yang terlihat di Protégé. |
| Tanpa data kasus sintetis | Kasus yang dibangkitkan dari KG yang sama dengan sumber embedding membuat evaluasi melingkar. |
| Fitur KG = rata-rata embedding gejala | Sederhana dan tidak tergantung jumlah gejala. Alternatif (attention, sum, concat per bagian tanaman) bisa ditambahkan sebagai fungsi baru di `cases.py`. |
| DBN berbentuk class ala sklearn | Bisa dibandingkan langsung dengan baseline sklearn di loop eksperimen yang sama. |

## 6. Status

| Modul | Status |
|---|---|
| `ontology.py` | Selesai. Test lulus (jumlah entitas, triple yang dibuang, profil gejala). |
| `embedding.py` | Selesai. Test hanya memeriksa bentuk keluaran, belum kualitas embedding. |
| `cases.py` | Selesai. Test lulus (resolusi nama, error untuk nama tak dikenal). |
| `dbn.py` | Selesai. `pretrain_lr=None` (default) memilih 0.1 untuk RBM Bernoulli dan 0.01 untuk Gaussian, sesuai literatur. `history_["pseudo_likelihood"]` berisi satu entri per RBM (sejajar dengan `history_["pretrain"]`), hanya dihitung untuk layer pertama Bernoulli, `None` untuk lainnya. Test data separable lulus untuk seed 0–4. **Catatan:** kegagalan lama (akurasi 0.70) *bukan* disebabkan `pretrain_lr`. Dengan `pretrain_lr=0.01` akurasinya juga 1.0 di seed 0–9, jadi kegagalan itu kemungkinan kebetulan dari angka acak (pretraining hanya 5 epoch). Apakah default baru memang lebih baik belum terbukti dan perlu diuji dengan data kasus nyata. |
| `neo4j_io.py` | Selesai (P6). Kontrak di §4.6. Test memakai mock; test live di-skip kalau `NEO4J_URI` tidak diset. Belum pernah dijalankan terhadap server Neo4j + GDS sungguhan. |
| `experiment.py` | Selesai, belum pernah dijalankan dengan data kasus nyata. |
| `literature.py` | Selesai (D0 di [DATA_PROTOCOL.md](DATA_PROTOCOL.md#2-roadmap)). Test lulus (validasi, aturan seleksi, kappa, round-trip ke `load_cases`, PRISMA). |
| Data kasus nyata | **Belum ada.** Ini penghambat utama. Protokol dan roadmap D0–D7 ada di [DATA_PROTOCOL.md](DATA_PROTOCOL.md). |

## 7. Paket kerja

Setiap paket punya batas file yang jelas supaya bisa dikerjakan paralel.

| # | Paket | File yang disentuh | Bergantung pada |
|---|---|---|---|
| P1 | **Ontologi v1.2**: perbaiki domain property, 7 triple yang salah, tambah gejala untuk `Tungro`, `JelagaPalsu`, `WalangSangit`, `UlatTandukHijau`, `Meloidogyne_spp.`, dan bedakan profil yang identik (`Thrips`/`TungauMalaiPadi`, `BelalangSawah`/`WerengPunggungPutih`) | `Ontologi/*_v1.2.rdf` (file baru) | — (butuh validasi pakar) |
| P2 | **Data kasus nyata**: literatur + lapangan, rinciannya di [DATA_PROTOCOL.md](DATA_PROTOCOL.md) | `data/ekstraksi_kasus.csv`, `data/sumber.csv` → `data/kasus.csv` | P1 untuk penamaan gejala |
| P3 | **Tuning RBM/DBN**: perbaiki test Bernoulli yang gagal, atur `pretrain_lr` terpisah per tipe visible, tambah monitoring (recon error, pseudo-likelihood) | `kgdbn/dbn.py`, `tests/` | — |
| P4 | **Embedding alternatif**: TransE/RotatE (PyKEEN) atau embedding teks `abstract` (IndoBERT), dengan kontrak keluaran §4.2 | `kgdbn/embedding.py` (fungsi baru) | — |
| P5 | **Evaluasi**: k-fold CV, beberapa seed, confusion matrix, uji signifikansi, ablation `exclude_relations` | `kgdbn/experiment.py` | P2 untuk hasil nyata |
| P6 | **Integrasi Neo4j (opsional)**: ekspor `KnowledgeGraph` ke Neo4j, embedding lewat GDS FastRP/node2vec, kredensial dari `.env` | `kgdbn/neo4j_io.py` (baru) | — |

Aturan kolaborasi:

- Jangan ubah signature di §4 tanpa memperbarui dokumen ini dan semua pemakainya.
- Setiap paket menambah atau memperbarui test di `tests/`. `pytest` harus lulus sebelum di-merge.
- Data kasus dan hasil eksperimen tidak di-commit (`data/` dan `*.csv` ada di `.gitignore`).
