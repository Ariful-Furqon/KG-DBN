# Protokol Data Kasus

Dokumen ini menjabarkan paket kerja **P2 (data kasus nyata)** di [ARCHITECTURE.md](ARCHITECTURE.md#7-paket-kerja): dari mana kasus diambil, bagaimana diekstraksi, dan bagaimana diubah menjadi `data/kasus.csv`.

## 1. Desain

Tidak ada dataset publik berupa kasus gejala → diagnosis padi (sudah diperiksa: ESforRPD2, Kisan Call Centre, survei IRRI, data klinik tanaman, Mendeley/Kaggle). Pilot D1 juga menunjukkan bahwa paper sistem pakar Indonesia hampir tidak pernah mempublikasikan kasus lapangan, tapi **hampir semuanya memuat tabel aturan atau profil penyakit** yang disusun pakar masing-masing. Karena itu set utamanya berupa vinyet, seperti evaluasi *symptom checker* medis (Semigran et al., *BMJ* 2015).

| Set | `jenis_kasus` | Sumber | Target | Fungsi di paper |
|---|---|---|---|---|
| **Vinyet pakar** | `vinyet_pakar` | Tabel aturan / basis pengetahuan dari paper sistem pakar yang **independen dari ontologi** | 200–400 vinyet dari ≥30 sumber | Set utama: GroupKFold per sumber |
| **Kasus literatur** | `lapangan`, `uji_pakar`, `laporan_penyakit` | Kasus nyata yang kebetulan dipublikasikan | Seadanya | Dilaporkan terpisah |
| **Lapangan** | `lapangan` | Penyuluh / BPP / pakar HPT di Jember, form gejala dari ontologi | 30–50 kasus | Validasi eksternal |

Vinyet adalah profil yang ditulis pakar, **bukan** observasi lapangan. Paper harus menyebutnya begitu, dan hasilnya dilaporkan per `jenis_kasus`. Pertanyaan yang dijawab set vinyet: apakah model yang dibangun dari ontologi ini bisa mendiagnosis profil yang disusun pakar lain?

Sumber yang hanya berisi **deskripsi naratif** penyakit (IRRI Rice Knowledge Bank, CABI, buku BBPadi) dipakai untuk ontologi (P1), bukan data uji.

## 2. Roadmap

| # | Langkah | Keluaran | Status |
|---|---|---|---|
| D0 | Template ekstraksi, konverter, dan panduan | `templates/*.csv`, `kgdbn/literature.py`, dokumen ini | Selesai |
| D1 | Pilot 12 sumber | `data/PILOT_REPORT.md` | Selesai. Hasil: 0–1 kasus nyata dari 12 sumber, sedangkan sebagian besar sumber punya tabel aturan → beralih ke vinyet |
| D2 | Pencarian dan ekstraksi vinyet (§4), termasuk meninjau ulang sumber pilot yang dieksklusi karena "hanya aturan" | `data/sumber.csv`, `data/ekstraksi_kasus.csv` dengan `gejala_a1` | Berikutnya |
| D3 | Daftar sumber yang dipakai membangun ontologi v1.1, lalu isi `dipakai_ontologi` | `dipakai_ontologi` terisi benar | Butuh pemilik ontologi |
| D4 | Pemetaan ulang terpisah oleh A2 (ahli HPT), hitung kappa, adjudikasi | `gejala_a2`, `gejala_final`, laporan kappa | |
| D5 | Seleksi dan ekspor | `data/kasus.csv` + laporan seleksi | |
| D6 | Set lapangan (paralel) | `data/kasus_lapangan.csv` | |
| D7 | Dataset card dan rilis Zenodo (CC BY 4.0) | DOI dataset | |

Setelah ontologi v1.2 selesai (P1), cukup ulangi D4–D5: teks gejala asli sudah tersimpan di `gejala_asli`, jadi paper tidak perlu dibaca ulang.

## 3. Aturan utama

1. **Tidak sirkular.** Sumber yang dipakai membangun ontologi dibuang (`dipakai_ontologi = ya`). Vinyet atau kasus yang identik dengan profil ontologi label-nya ditandai `sama_profil_ontologi` dan dilaporkan; jumlah yang tinggi menandakan sumbernya sama dengan sumber ontologi.
2. **Independensi antar sumber.** Skripsi sering menyalin tabel aturan dari skripsi lain. Vinyet yang identik (gejala dan label sama) di lebih dari satu sumber ditandai `duplikat_lintas_sumber`. Jumlahnya dilaporkan, dan analisis sensitivitas dilakukan dengan membuangnya.
3. **Bukan uji acak.** Gejala yang dipilih penguji untuk mencoba sistem, atau contoh ilustrasi perhitungan, dicatat `uji_acak` dan dibuang. Label yang berasal dari keluaran sistem (bukan pakar) juga dibuang.
4. **Jejak asal.** Setiap baris punya `id_sumber`, halaman/tabel, dan teks asli. Evaluasi memakai GroupKFold pada `id_sumber`.
5. **Dua anotator.** A1 dan A2 memetakan `gejala_asli` ke individu `Gejala` secara terpisah, tanpa melihat hasil satu sama lain. Kappa dilaporkan di paper.
6. **Verbatim.** Setiap item `gejala_asli` harus ada persis di teks sumbernya (`berkas` di `sumber.csv`). Konverter membuang baris yang tidak lolos (perbandingan mengabaikan huruf besar, spasi, dan tanda baca). Aturan ini ditambahkan setelah ekstraksi D2 oleh agen LLM ternyata hanya 28% verbatim.
7. **Label harus tertulis di sumber.** Jika kode target di tabel aturan tidak cocok dengan daftar target di paper yang sama, aturan itu tidak dipakai. Label tidak boleh disimpulkan dari isi gejala.
8. **Tidak ada data yang dibangkitkan sendiri.** Vinyet disalin dari tulisan pakar lain, tidak pernah dibuat dari ontologi (sesuai [ARCHITECTURE.md §4.3](ARCHITECTURE.md#43-casespy-kasus--fitur)).

## 4. Protokol pencarian (D2)

Basis data: Google Scholar, Garuda (garuda.kemdikbud.go.id), repositori kampus (skripsi), Crossref/Scopus untuk laporan penyakit.

Query:

- `"sistem pakar" (hama OR penyakit) padi ("basis pengetahuan" OR "tabel aturan" OR "rule")`
- `"sistem pakar" padi ("certainty factor" OR "dempster shafer" OR "forward chaining" OR "case based reasoning" OR bayes OR fuzzy)`
- `"expert system" rice (pest OR disease) diagnosis "knowledge base"`
- `"first report" rice (Oryza sativa) (Indonesia OR Asia)` di *Plant Disease* / *New Disease Reports* (kasus nyata)

**Semua** hasil yang disaring lewat judul/abstrak dicatat di `sumber.csv` (tahap `ditemukan`), bukan hanya yang dibuka.

Kriteria inklusi:

- Memuat tabel yang memetakan setiap hama/penyakit ke **daftar gejala tertulis** (bukan hanya kode G01..Gnn tanpa kamus), atau kasus nyata dengan daftar gejala dan diagnosis.
- Penyusun pengetahuan disebut (pakar, penyuluh, atau rujukan pustaka); dicatat di `konfirmasi_label`.
- Target ada di ontologi, atau dicatat sebagai kandidat penambahan ontologi.

Kriteria eksklusi: hanya citra; kode gejala tanpa kamus; duplikat publikasi (skripsi dan artikel dari penulis yang sama: ambil satu); teks tidak dapat diakses.

Setiap sumber dicatat di `data/sumber.csv` beserta tahap terakhir yang dicapainya, sehingga angka diagram alur PRISMA bisa dihitung langsung. Sumber dianggap `dimasukkan` hanya jika menyumbang minimal satu baris yang lolos seleksi.

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
| `berkas` | Path teks sumber (hasil `pdftotext`), relatif terhadap folder `sumber.csv`. Wajib untuk sumber yang diekstraksi |

### `ekstraksi_kasus.csv` (satu baris per kasus atau vinyet)

Untuk vinyet: satu baris per aturan. Jika satu penyakit punya beberapa aturan alternatif (OR), setiap aturan menjadi baris tersendiri. Kode gejala (G01, ...) diganti dengan teks gejala dari kamus di paper yang sama.

| Kolom | Isi |
|---|---|
| `id_kasus` | `S001-01`, `S001-02`, ... |
| `id_sumber` | Sama dengan di `sumber.csv` |
| `halaman` | Halaman/tabel tempat kasus berada, mis. `hlm 45, Tabel 4.3` |
| `jenis_kasus` | `vinyet_pakar` (satu baris = satu profil hama/penyakit di tabel aturan), `lapangan`, `uji_pakar`, `laporan_penyakit` (dipakai); `uji_acak` (dibuang) |
| `konfirmasi_label` | `laboratorium`, `pakar`, `penulis`, `tidak_ada`. Untuk vinyet: siapa penyusun aturannya |
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

Konverter memeriksa semua nama terhadap ontologi (nama yang tidak dikenal langsung error dengan nomor barisnya), memeriksa bahwa `gejala_asli` verbatim ada di `berkas` sumbernya, menghitung kappa A1/A2, membuang kasus sesuai §3, lalu menulis `kasus.csv` dengan kolom `gejala,label` ditambah kolom jejak (`id_kasus`, `id_sumber`, `jenis_kasus`, `konfirmasi_label`, `sama_profil_ontologi`, `duplikat_lintas_sumber`). File ini langsung bisa dipakai `load_cases`.

Laporan yang dicetak: jumlah kasus per alasan eksklusi, jumlah per `jenis_kasus`, duplikat lintas sumber, kappa, kasus yang perlu adjudikasi, distribusi label, gejala tak terpetakan yang paling sering, dan hitungan PRISMA dari `sumber.csv`.
