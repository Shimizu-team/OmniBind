# Data

## Overview

OmniBind uses [BindingDB](https://www.bindingdb.org/) as the primary training data source, with protein 3Di structural alphabet sequences obtained via [Foldseek](https://github.com/steineggerlab/foldseek).

**Note**: Raw data files are not included in this repository due to their size. Follow the instructions below to download and preprocess the data.

## Data Download

### 1. BindingDB

Download the full BindingDB dataset from:
- https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?all_download=yes

Select "BindingDB_All.tsv" (tab-separated format).

**Versions used in this study:**
- **Label reversal experiments**: BindingDB January 2023 release
- **Time-split experiments**: BindingDB January 2023 (training) and November 2024 (evaluation) releases

### 2. Protein 3D Structures (for Foldseek)

Obtain PDB structures for target proteins. Sources:
- [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/)
- [RCSB PDB](https://www.rcsb.org/)

## Preprocessing Pipeline

Run the following scripts **in order**. Each step depends on the output of the previous step.

### Step 1: Process BindingDB TSV

Filter, clean, and convert BindingDB data to pickle format. Produces per-label pickle files and one combined file.

```bash
python data/preprocessing/process_bindingdb.py \
    --input_path /path/to/BindingDB_All.tsv \
    --output_dir ./data/processed
```

**Output:**
- `data/processed/BindingDB_all_affinities.pkl` — All data with Ki/Kd/IC50/EC50 columns (pK values)
- `data/processed/BindingDB_Ki.pkl`, `BindingDB_Kd.pkl`, `BindingDB_IC50.pkl`, `BindingDB_EC50.pkl`

### Step 2: Create sequence ID mappings

Extract unique protein sequences, assign integer IDs, and generate a FASTA file for Foldseek.

```bash
python data/preprocessing/make_sequence_id.py \
    --data_dir ./data/processed
```

**Output:**
- `data/processed/id_2_seq.pkl` — `{int_id: aa_sequence}`
- `data/processed/seq_2_id.pkl` — `{aa_sequence: int_id}`
- `data/processed/bdb_sequences.fasta` — FASTA for Foldseek input

### Step 3: Generate 3Di sequences with Foldseek

Use [Foldseek](https://github.com/steineggerlab/foldseek) to obtain 3Di structural alphabet sequences from protein structures.

```bash
# Create Foldseek structure database from PDB files
foldseek createdb /path/to/pdb_structures/ structureDB

# Extract 3Di sequences
foldseek lndb structureDB_h structureDB_ss_h
foldseek convert2fasta structureDB_ss structureDB_ss.fasta
```

Then create `id_2_ss.pkl` by mapping each protein ID to its 3Di sequence. This mapping must use the same integer IDs from Step 2 (`id_2_seq.pkl`).

```python
# Example script to create id_2_ss.pkl
import pickle

# Parse Foldseek output FASTA
id_2_ss = {}
with open('structureDB_ss.fasta') as f:
    current_id = None
    for line in f:
        if line.startswith('>'):
            current_id = int(line.strip().lstrip('>'))
        else:
            id_2_ss[current_id] = line.strip()

with open('./data/processed/id_2_ss.pkl', 'wb') as f:
    pickle.dump(id_2_ss, f)
```

**Output:**
- `data/processed/id_2_ss.pkl` — `{int_id: 3di_sequence}`

### Step 4: Add protein IDs to pickle files

Add `protein_id` column to all BindingDB pickle files for linking to sequence/3Di dictionaries.

```bash
python data/preprocessing/add_ids_to_bdb.py \
    --data_dir ./data/processed
```

### Step 5: Split data into train/validation/test sets

Ligand-based splitting ensures no compound leakage between splits.

```bash
python data/preprocessing/split_data.py \
    --data_dir ./data/processed \
    --seeds 42 123 369 777 2024
```

**Output** (per seed):
- `data/processed/seed42/BindingDB_all_affinities_train.pkl`
- `data/processed/seed42/BindingDB_all_affinities_valid.pkl`
- `data/processed/seed42/BindingDB_all_affinities_test.pkl`

### Step 6: Convert to numpy arrays

Featurize compounds, AA sequences, and 3Di sequences. Produces flat npy files with separate files for each affinity label (ki, kd, ic50, ec50). Missing labels are filled with -1. Run for each split (train, valid, test) and each seed.

```bash
for seed in 42 123 369 777 2024; do
    for suffix in train valid test; do
        python data/preprocessing/convert_to_npy.py \
            --data_dir ./data/processed/seed${seed} \
            --source_dir ./data/processed \
            --output_suffix ${suffix}
    done
done
```

**Output** (per seed):
```
data/processed/seed42/
├── compounds_train.npy    # Atom feature arrays
├── adjancies_train.npy    # Adjacency matrices
├── aas_train.npy          # TAPE-tokenized AA sequences
├── sas_train.npy          # 3Di alphabet index arrays
├── ki_train.npy           # pKi values (-1 for missing)
├── kd_train.npy           # pKd values (-1 for missing)
├── ic50_train.npy         # pIC50 values (-1 for missing)
├── ec50_train.npy         # pEC50 values (-1 for missing)
├── compounds_valid.npy
├── ...
└── ec50_test.npy
```

**Important**: Each label npy file has shape `(N, 1)` where `-1` indicates a missing value for that sample. The model uses masked multi-task learning, computing loss only on valid (non -1) entries.

## Sample Data

A small sample dataset is provided in `data/sample/sample_cpi.csv` for testing and demonstration purposes.

## Full Directory Structure After Preprocessing

```
data/
├── processed/
│   ├── BindingDB_all_affinities.pkl
│   ├── BindingDB_Ki.pkl
│   ├── BindingDB_Kd.pkl
│   ├── BindingDB_IC50.pkl
│   ├── BindingDB_EC50.pkl
│   ├── id_2_seq.pkl
│   ├── seq_2_id.pkl
│   ├── id_2_ss.pkl
│   ├── bdb_sequences.fasta
│   └── seed42/
│       ├── BindingDB_all_affinities_train.pkl
│       ├── BindingDB_all_affinities_valid.pkl
│       ├── BindingDB_all_affinities_test.pkl
│       ├── compounds_train.npy
│       ├── adjancies_train.npy
│       ├── aas_train.npy
│       ├── sas_train.npy
│       ├── ki_train.npy
│       ├── kd_train.npy
│       ├── ic50_train.npy
│       ├── ec50_train.npy
│       ├── compounds_valid.npy
│       ├── adjancies_valid.npy
│       ├── aas_valid.npy
│       ├── sas_valid.npy
│       ├── ki_valid.npy
│       ├── kd_valid.npy
│       ├── ic50_valid.npy
│       ├── ec50_valid.npy
│       ├── compounds_test.npy
│       ├── adjancies_test.npy
│       ├── aas_test.npy
│       ├── sas_test.npy
│       ├── ki_test.npy
│       ├── kd_test.npy
│       ├── ic50_test.npy
│       └── ec50_test.npy
└── sample/
    └── sample_cpi.csv
```
