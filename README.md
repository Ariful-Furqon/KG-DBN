# KG-DBN

**KG-DBN** is an architecture for symptom-based diagnosis of rice pests and diseases that combines an ontology-grounded **knowledge graph** (KG) with a **deep belief network** (DBN).

Expert knowledge about rice diseases, pests, symptoms, pathogens and control agents is encoded as an OWL ontology and read as a knowledge graph (Neo4j is optional). Node embeddings learned from the graph are used as input features of a DBN that predicts the pest or disease from a set of observed symptoms.

> **Status:** the pipeline is implemented as the `kgdbn` package, and the analysis of the knowledge layer (which needs no case data) is available in `kgdbn.analysis`. Case data (expert vignettes) are being collected and validated; the diagnostic evaluation has not been reported yet.

## Repository layout

```
KG-DBN/
├── Ontologi/OntologiHamaPenyakitPadi_v1.1.rdf   # rice pest & disease ontology (OWL, RDF/XML)
├── kgdbn/                                       # pipeline package
│   ├── ontology.py     # RDF → knowledge graph, with schema validation
│   ├── embedding.py    # node2vec (biased random walks + skip-gram, PyTorch)
│   ├── cases.py        # case loader and feature encodings
│   ├── literature.py   # literature extraction sheet → kasus.csv
│   ├── dbn.py          # RBM + deep belief network
│   ├── analysis.py     # ontology and embedding analysis (no case data)
│   ├── neo4j_io.py     # optional Neo4j export and GDS embeddings
│   └── experiment.py   # experiments and CLI
├── tests/              # pytest
├── templates/          # CSV templates for extracting cases from the literature
├── KG-DBN.ipynb        # exploration notebook
└── requirements.txt
```

Pipeline:

```
ontology (RDF) → ontology.py → KnowledgeGraph → embedding.py (node2vec) → {entity_id: vector}
kasus.csv → cases.py → multi-hot m  +  mean symptom embedding k → dbn.py (RBM pretraining → fine-tuning)
                                                                → experiment.py (baselines, metrics)
```

## Ontology

The ontology is taken from [Sistem-Pakar-Deep-Knowledge-Graph](https://github.com/Ariful-Furqon/Sistem-Pakar-Deep-Knowledge-Graph/tree/main/Ontologi) (v1.1). Its class, property and individual names are in Indonesian.

| Class | Meaning | Individuals |
|---|---|---|
| `PenyakitPadi` | rice disease | 19 |
| `HamaPadi` | rice pest | 27 |
| `Gejala` | symptom | 41 |
| `PatogenPadi` | pathogen | 15 |
| `Fungisida` / `Bakterisida` / `Pestisida` / `Biologis` | fungicide / bactericide / pesticide / biological agent | 19 / 5 / 13 / 5 |

Main relations:

- `memilikiGejala` (has symptom): disease/pest → symptom
- `terkenaPatogen` (caused by pathogen): disease → pathogen
- `menyebabkanPenyakit` (causes disease): pest → disease
- `diberikanFungisida`, `diberikanBakterisida`, `diberikanPestisida`, `diberikanAgenBiologi` (treated with …): disease/pest → control agent

Most individuals also carry an `rdfs:abstract` annotation with a text description.

**Known issues in v1.1** (to be fixed in v1.2):

- The domain of every object property is declared as `Padi` (rice plant), although the subjects are `PenyakitPadi`/`HamaPadi`, so a reasoner would infer wrong types. `ontology.py` therefore validates assertions against its own schema table.
- 7 assertions have subjects or objects of the wrong type, e.g. `Lesi_pada_bulir memilikiGejala Lesi_pada_bulir`; they are dropped when the graph is built.
- `Tungro`, `JelagaPalsu`, `WalangSangit`, `UlatTandukHijau` and `Meloidogyne_spp.` have no symptoms.
- Some targets have identical symptom profiles or profiles that are strict subsets of another (e.g. `Thrips` = `TungauMalaiPadi`), so they cannot be distinguished from symptoms alone.

## Installation

Requires Python 3.10+. Neo4j is not required; the ontology is read directly with rdflib.

```bash
pip install -r requirements.txt
```

### Optional Neo4j support

To export the graph to Neo4j or compute embeddings with Graph Data Science (FastRP / node2vec):

```bash
pip install neo4j
```

Credentials are read from environment variables or a `.env` file:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

## Usage

### Reproducing the knowledge-layer analysis

The ontology analysis and the embedding analysis need no case data. This command reproduces every number in Sections 3 and 5 of the preprint (10 node2vec seeds plus the hyperparameter sensitivity runs):

```bash
python -m kgdbn.analysis --seeds 10 --sensitivity --out results/analysis
```

### Case data

Cases are stored in `data/kasus.csv`:

```
gejala,label
Lesi_pada_Daun;Lesi_pada_batang,Blas
Daun menguning;Layu,WerengBatangCoklat
```

`gejala` (symptoms) is a `;`-separated list of `Gejala` individuals, and `label` is a `PenyakitPadi` or `HamaPadi` individual. Both may be written as the ontology id or its `rdfs:label`; unknown names raise an error instead of being dropped.

Cases from the literature (expert vignettes taken from the rule tables of published expert systems) are first extracted with the templates in `templates/` (`sumber.csv` lists the publications, `ekstraksi_kasus.csv` the cases) and then converted. The converter checks every name against the ontology, verifies that the original symptom text appears verbatim in the source, drops circular or non-random cases, and reports inter-annotator kappa and PRISMA counts.

```bash
python -m kgdbn.literature --ekstraksi data/ekstraksi_kasus.csv --sumber data/sumber.csv --out data/kasus.csv
python -m kgdbn.experiment --cases data/kasus.csv
```

### Tests

```bash
python -m pytest tests
```

## Roadmap

- Ontology v1.2: fix the property domains and invalid assertions, add symptoms to the targets that have none, and separate the identical profiles.
- Diagnostic evaluation on expert vignettes: source-grouped cross-validation, multiple seeds, McNemar tests, and the symptom-only graph ablation.
- Alternative embeddings (TransE/RotatE, text embeddings of `rdfs:abstract`).

## License

[MIT](LICENSE)
