# Protokol Data Kasus

Dokumen ini menjabarkan paket kerja **P2 (data kasus nyata)** di [ARCHITECTURE.md](ARCHITECTURE.md#7-paket-kerja): dari mana kasus diambil, bagaimana diekstraksi, dan bagaimana diubah menjadi `data/kasus.csv`.

## 1. Desain dua lapis

| Set | Sumber | Target | Fungsi di paper |
|---|---|---|---|
| **Literatur** | Kasus uji di skripsi/jurnal sistem pakar, laporan penyakit (*Disease Notes*, *New Disease Reports*), laporan survei OPT | 200–400 kasus | Set utama (CV) |
| **Lapangan** | Penyuluh / BPP / pakar HPT di Jember, form gejala dari ontologi | 30–50 kasus | Validasi eksternal |

Sumber yang hanya berisi **deskripsi** penyakit (IRRI Rice Knowledge Bank, CABI, buku BBPadi) bukan kasus. Sumber ini dipakai untuk ontologi (P1), bukan untuk data uji.

## 2. Roadmap

| # | Langkah | Keluaran | Status |
|---|---|---|---|
| D0 | Template ekstraksi, konverter, dan panduan | `templates/*.csv`, `kgdbn/literature.py`, dokumen ini | Selesai |
| D1 | **Pilot**: ekstraksi 5–10 skripsi/jurnal, lalu revisi template dan kriteria | `data/ekstraksi_kasus.csv` (pilot), catatan revisi | Berikutnya |
| D2 | Protokol pencarian: query, basis data, kriteria inklusi/eksklusi (§4) | `data/sumber.csv` terisi sampai tahap penyaringan | |
| D3 | Ekstraksi penuh oleh anotator A1 | `ekstraksi_kasus.csv` dengan `gejala_a1` | |
| D4 | Pemetaan ulang terpisah oleh A2 (ahli HPT), hitung kappa, adjudikasi | `gejala_a2`, `gejala_final`, laporan kappa | |
| D5 | Seleksi dan ekspor: buang profil aturan, duplikat, sumber ontologi | `data/kasus.csv` + laporan seleksi | |
| D6 | Set lapangan (paralel dengan D2–D5) | `data/kasus_lapangan.csv` | |
| D7 | Dataset card dan rilis Zenodo (CC BY 4.0) | DOI dataset | |

Setelah ontologi v1.2 selesai (P1), cukup ulangi D4–D5: teks gejala asli sudah tersimpan di `gejala_asli`, jadi paper tidak perlu dibaca ulang.

## 3. Aturan utama

1. **Tidak sirkular.** Sumber yang dipakai membangun ontologi tidak boleh menjadi sumber kasus (`dipakai_ontologi = ya` dibuang). Kasus yang gejalanya identik dengan profil ontologi label-nya ditandai `sama_profil_ontologi` dan dilaporkan.
2. **Bukan profil aturan.** Baris aturan IF–THEN atau daftar gejala per penyakit dari tabel basis pengetahuan dicatat sebagai `profil_aturan` dan dibuang. Kasus dengan gejala yang dipilih acak oleh penulis dicatat `uji_acak` dan juga dibuang.
3. **Jejak asal.** Setiap kasus punya `id_sumber`, halaman/tabel, dan teks asli. Evaluasi memakai split berbasis sumber (GroupKFold pada `id_sumber`) supaya kasus dari sumber yang sama tidak muncul di train dan test sekaligus.
4. **Dua anotator.** A1 dan A2 memetakan `gejala_asli` ke individu `Gejala` secara terpisah, tanpa melihat hasil satu sama lain. Kappa dilaporkan di paper.
5. **Tidak ada data sintetis** (sesuai [ARCHITECTURE.md §4.3](ARCHITECTURE.md#43-casespy-kasus--fitur)).

## 4. Protokol pencarian (D2)

Basis data: Google Scholar, Garuda (garuda.kemdikbud.go.id), repositori kampus (skripsi), Crossref/Scopus untuk laporan penyakit.

Query awal (disempurnakan setelah pilot):

- `"sistem pakar" (hama OR penyakit) padi (pengujian OR "kasus uji" OR konsultasi)`
- `"sistem pakar" padi ("certainty factor" OR "dempster shafer" OR "forward chaining" OR "case based reasoning" OR bayes)`
- `"expert system" rice (pest OR disease) diagnosis "test case"`
- `"first report" rice (Oryza sativa) (Indonesia OR Asia)` di *Plant Disease* / *New Disease Reports*

Kriteria inklusi:

- Memuat kasus per tanaman/lokasi dengan daftar gejala yang teramati **dan** diagnosis.
- Diagnosis dikonfirmasi pakar, laboratorium, atau penulis ahli (dicatat di `konfirmasi_label`).
- Target ada di ontologi, atau dicatat sebagai kandidat penambahan ontologi.

Kriteria eksklusi: hanya aturan/basis pengetahuan tanpa kasus, hanya citra, duplikat publikasi (skripsi dan artikel jurnal dari penulis yang sama: ambil satu), teks tidak dapat diakses.

Setiap sumber dicatat di `data/sumber.csv` beserta tahap terakhir yang dicapainya, sehingga angka diagram alur PRISMA bisa dihitung langsung.

## 5. Panduan pengisian

Salin template ke `data/` (folder ini di-`.gitignore`):

```bash
mkdir -p data
cp templates/sumber.csv templates/ekstraksi_kasus.csv data/
```

### `sumber.csv` (satu baris per publikasi yang ditemukan)

| Kolom | Isi |
|---|---|
| `id_sumber` | `S001`, `S002`, ... |
| `sitasi` | Penulis (tahun), judul singkat |
| `jenis_sumber` | `skripsi`, `jurnal`, `laporan_penyakit`, `survei`, `log_petani`, `lainnya` |
| `url_doi` | DOI atau URL |
| `basis_data` | Tempat ditemukan (Scholar, Garuda, ...) |
| `query` | Query yang menemukannya |
| `tahap` | `ditemukan`, `disaring`, `layak`, `dimasukkan` (tahap terakhir yang lolos) |
| `alasan_eksklusi` | Wajib jika `tahap` bukan `dimasukkan` |

### `ekstraksi_kasus.csv` (satu baris per kasus)

| Kolom | Isi |
|---|---|
| `id_kasus` | `S001-01`, `S001-02`, ... |
| `id_sumber` | Sama dengan di `sumber.csv` |
| `halaman` | Halaman/tabel tempat kasus berada, mis. `hlm 45, Tabel 4.3` |
| `jenis_kasus` | `lapangan`, `uji_pakar`, `laporan_penyakit` (dipakai); `profil_aturan`, `uji_acak` (dibuang) |
| `konfirmasi_label` | `laboratorium`, `pakar`, `penulis`, `tidak_ada` |
| `dipakai_ontologi` | `ya` jika sumber ini juga dipakai membangun ontologi, selain itu `tidak` |
| `lokasi`, `tahun_observasi` | Jika disebut |
| `gejala_asli` | Teks gejala **persis** seperti di sumber, dipisah `;` |
| `label_asli` | Diagnosis persis seperti di sumber |
| `gejala_a1` | Pemetaan A1 ke individu `Gejala` (id atau `rdfs:label`), dipisah `;` |
| `gejala_a2` | Pemetaan A2, diisi terpisah. Kosong = hanya satu anotator |
| `gejala_final` | Hasil adjudikasi, hanya diisi jika A1 ≠ A2 |
| `gejala_tak_terpetakan` | Gejala asli yang tidak punya padanan di ontologi, dipisah `;` (bahan P1) |
| `label` | Individu `PenyakitPadi`/`HamaPadi` |
| `catatan` | Bebas |

Contoh baris (disingkat):

```
S001-01,S001,"hlm 45, Tabel 4.3",uji_pakar,pakar,tidak,Jember,2021,"bercak belah ketupat pada daun;leher malai busuk;bau apek",Blas,Lesi_pada_Daun;Lesi pada Malai,,,bau apek,Blas,
```

## 6. Konversi

```bash
python -m kgdbn.literature --ekstraksi data/ekstraksi_kasus.csv --sumber data/sumber.csv --out data/kasus.csv
```

Konverter memeriksa semua nama terhadap ontologi (nama yang tidak dikenal langsung error dengan nomor barisnya), menghitung kappa A1/A2, membuang kasus sesuai §3, lalu menulis `kasus.csv` dengan kolom `gejala,label` ditambah kolom jejak (`id_kasus`, `id_sumber`, `jenis_kasus`, `konfirmasi_label`, `sama_profil_ontologi`). File ini langsung bisa dipakai `load_cases`.

Laporan yang dicetak: jumlah kasus per alasan eksklusi, kappa, kasus yang perlu adjudikasi, distribusi label, gejala tak terpetakan yang paling sering, dan hitungan PRISMA dari `sumber.csv`.
