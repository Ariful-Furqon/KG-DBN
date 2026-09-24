# KG-DBN

Proyek penelitian **Deep Knowledge Graph** untuk diagnosis hama dan penyakit tanaman padi. Proyek ini menggabungkan *Knowledge Graph* (KG) berbasis ontologi dengan *Deep Belief Network* (DBN).

Idenya: pengetahuan pakar tentang penyakit, hama, gejala, patogen, dan pengendaliannya disusun sebagai ontologi, lalu dibaca sebagai KG (Neo4j opsional). Embedding node dari KG tersebut menjadi fitur masukan DBN untuk memprediksi penyakit atau hama dari gejala yang diamati.

> **Status:** pipeline sudah tersusun sebagai package `kgdbn`, tapi belum ada data kasus nyata dan DBN masih perlu tuning. Lihat [status](docs/ARCHITECTURE.md#6-status).

## Struktur repo

```
KG-DBN/
├── Ontologi/OntologiHamaPenyakitPadi_v1.1.rdf   # Ontologi hama & penyakit padi (OWL/RDF-XML)
├── kgdbn/                                       # Package pipeline
│   ├── ontology.py     # RDF → knowledge graph
│   ├── embedding.py    # node2vec
│   ├── cases.py        # loader data kasus + fitur
│   ├── dbn.py          # RBM + Deep Belief Network
│   └── experiment.py   # eksperimen & CLI
├── tests/              # pytest
├── docs/ARCHITECTURE.md
├── KG-DBN.ipynb        # notebook eksplorasi
└── requirements.txt
```

Desain lengkap, kontrak antarmodul, status, dan paket kerja ada di [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Ontologi

Ontologi disalin dari repo [Sistem-Pakar-Deep-Knowledge-Graph](https://github.com/Ariful-Furqon/Sistem-Pakar-Deep-Knowledge-Graph/tree/main/Ontologi) (v1.1).

| Kelas | Individu |
|---|---|
| `PenyakitPadi` | 19 |
| `HamaPadi` | 27 |
| `Gejala` | 41 |
| `PatogenPadi` | 15 |
| `Fungisida` / `Bakterisida` / `Pestisida` / `Biologis` | 19 / 5 / 13 / 5 |

Relasi utama:

- `memilikiGejala`: Penyakit/Hama → Gejala
- `terkenaPatogen`: Penyakit → Patogen
- `menyebabkanPenyakit`: Hama → Penyakit
- `diberikanFungisida`, `diberikanBakterisida`, `diberikanPestisida`, `diberikanAgenBiologi`: Penyakit/Hama → pengendalian

Sebagian besar individu juga punya anotasi `rdfs:abstract` berisi deskripsi teks.

**Masalah yang diketahui di v1.1** (akan diperbaiki di v1.2):

- Domain object property ditulis `Padi`, padahal subjeknya `PenyakitPadi`/`HamaPadi`. Reasoner akan salah menyimpulkan tipe.
- Ada 7 triple dengan tipe subjek/objek yang salah, misalnya `Lesi_pada_bulir memilikiGejala Lesi_pada_bulir`.
- `Tungro`, `JelagaPalsu`, `WalangSangit`, `UlatTandukHijau`, dan `Meloidogyne_spp.` belum punya gejala.
- Beberapa entitas punya profil gejala yang identik atau merupakan subset entitas lain (mis. `Thrips` = `TungauMalaiPadi`), sehingga tidak bisa dibedakan dari gejala saja.

## Instalasi

Prasyarat: Python 3.10+. Neo4j tidak wajib, karena ontologi dibaca langsung dengan rdflib.

```bash
pip install -r requirements.txt
```

### Dukungan Neo4j (Opsional)

Jika ingin mengekspor graf ke Neo4j atau memakai GDS (FastRP / node2vec):

```bash
pip install neo4j
```

Konfigurasikan kredensial melalui variabel lingkungan atau file `.env`:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password_anda
```

## Penggunaan

Siapkan data kasus nyata di `data/kasus.csv`:

```
gejala,label
Lesi_pada_Daun;Lesi_pada_batang,Blas
Daun menguning;Layu,WerengBatangCoklat
```

`gejala` berisi daftar individu `Gejala` yang dipisah `;`. `label` berisi individu `PenyakitPadi` atau `HamaPadi`. Keduanya boleh ditulis sebagai id ontologi atau `rdfs:label`.

```bash
python -m kgdbn.experiment --cases data/kasus.csv
python -m pytest tests
```

## Roadmap

Lihat paket kerja P1–P6 di [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#7-paket-kerja).

## Lisensi

[MIT](LICENSE)
