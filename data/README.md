# Data

## Overview

OmniBind uses [BindingDB](https://www.bindingdb.org/) as the primary training data source. Protein 3Di structural alphabet sequences are predicted directly from amino acid sequences with the [ProstT5](https://huggingface.co/Rostlab/ProstT5) protein language model, accessed through [Foldseek](https://github.com/steineggerlab/foldseek), so no experimental or predicted 3D structures are required.

**Note**: Raw data files are not included in this repository due to their size. Follow the instructions below to download and preprocess the data.

## Data Download

### 1. BindingDB

Download the full BindingDB dataset (`BindingDB_All_YYYYmm_tsv`, tab-separated format) from:
- https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?all_download=yes

**Versions used in this study:**
- **Model development and label reversal experiments**: BindingDB May 2023 release
- **Temporal validation**: BindingDB May 2023 (training) and November 2024 (evaluation) releases

The BindingDB download page provides only the current release; BindingDB additionally maintains a quarterly archive of its full dataset at the UC San Diego Library Digital Collections (https://doi.org/10.6075/J0HD7VVF).

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

### Step 3: Predict 3Di sequences with ProstT5

3Di sequences are predicted from the amino acid sequences in `bdb_sequences.fasta`
(Step 2) using the ProstT5 protein language model through Foldseek. No 3D
structures are required.

The 3Di sequences used in this study were generated with Foldseek 9.427df8a
(bioconda build `pl5321hb365157_0`).

```bash
# One-time: download the ProstT5 model weights
foldseek databases ProstT5 weights tmp

# Predict 3Di sequences from the amino acid FASTA
# (append --gpu 1 to run inference on a GPU instead of the CPU)
foldseek createdb ./data/processed/bdb_sequences.fasta seqDB --prostt5-model weights

# Export the predicted 3Di sequences as FASTA
foldseek lndb seqDB_h seqDB_ss_h
foldseek convert2fasta seqDB_ss seqDB_ss.fasta
```

The FASTA headers are the integer protein IDs assigned in Step 2, so the output
can be converted into the `id_2_ss.pkl` mapping directly:

```python
import pickle

id_2_ss = {}
with open('seqDB_ss.fasta') as f:
    current_id = None
    for line in f:
        if line.startswith('>'):
            current_id = int(line.strip().lstrip('>'))
        else:
            id_2_ss[current_id] = line.strip()

with open('./data/processed/id_2_ss.pkl', 'wb') as f:
    pickle.dump(id_2_ss, f)
```

For a small number of proteins, the [Foldseek web server](https://search.foldseek.com)
can be used instead of a local installation.

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

This step also applies the length filters used in the paper: samples whose SMILES is
longer than 500 characters (`--smiles_max_len`) or whose protein sequence is longer
than 1,500 residues (`--seq_max_len`) are dropped. Those are the defaults, so the npy
files hold fewer samples than the pickle files of Step 5; the script prints how many
were removed.

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

## Temporal Validation Set

The temporal validation set is built from two BindingDB releases: the release
used for training (May 2023) and a later one (November 2024). An interaction
record of the later release is kept when its compound-protein pair is not
present in the earlier release, so the set consists of interaction pairs that
were newly deposited after the training data was collected. Compounds and
proteins themselves may already occur in the training data; only the pair is
new.

### Step T1: Extract newly deposited interaction pairs

```bash
python data/preprocessing/make_temporal_set.py \
    --train_tsv /path/to/BindingDB_All_2023m5.tsv \
    --test_tsv /path/to/BindingDB_All_2024m11.tsv \
    --output_dir ./data/processed/temporal
```

**Output** (same file names as Step 1, in a separate directory):
- `data/processed/temporal/BindingDB_all_affinities.pkl`
- `data/processed/temporal/BindingDB_Ki.pkl`, `BindingDB_Kd.pkl`, `BindingDB_IC50.pkl`, `BindingDB_EC50.pkl`

### Step T2: Sequence IDs, 3Di sequences and protein IDs

Run Steps 2 to 4 on the temporal directory. The commands are identical, with
`--data_dir ./data/processed/temporal`; the 3Di sequences are predicted from
`data/processed/temporal/bdb_sequences.fasta` exactly as in Step 3.

```bash
python data/preprocessing/make_sequence_id.py --data_dir ./data/processed/temporal
# ... predict 3Di as in Step 3 and write id_2_ss.pkl into the same directory ...
python data/preprocessing/add_ids_to_bdb.py --data_dir ./data/processed/temporal
```

### Step T3: Convert to numpy arrays

`convert_to_npy.py` expects the split naming used in Step 5, so the combined
pickle is copied once under the `test` suffix:

```bash
cp ./data/processed/temporal/BindingDB_all_affinities.pkl \
   ./data/processed/temporal/BindingDB_all_affinities_test.pkl

python data/preprocessing/convert_to_npy.py \
    --data_dir ./data/processed/temporal \
    --source_dir ./data/processed/temporal \
    --output_suffix test
```

The resulting arrays can be evaluated with `scripts/test.py` by pointing
`dataset.data_dir` at `./data/processed/temporal`.

## Sample Data

`data/sample/` holds small datasets for the demo in the main README and for the
notebooks in `notebooks/`. They are not used for training or evaluation.

| File | Contents |
|------|----------|
| `sample_proteins.csv` | 102 human proteins: UniProt accession, gene name, amino acid sequence and the matching 3Di sequence. Includes the clozapine and clomipramine targets, ABL1, MAP2K1, PDE5A, KLKB1 and SIRT3, plus unrelated proteins as background. The last two entries, `ABL1_KD` and `ABL1_KD_T315I`, are the 293-residue ABL1 kinase domain used in the attention analysis and its T315I resistance mutant. |
| `sample_compounds.csv` | 40 compounds (ZINC ID and SMILES) drawn from the FDA-approved library used in the paper. |
| `sample_cpi.csv` | A few compound-protein pairs with affinity labels, illustrating the column layout of the processed data. |

Sequences are taken from the UniProt reviewed human proteome; the 3Di sequences were
predicted with ProstT5 exactly as in Step 3 above. The 3Di sequences of the two kinase
domain entries were predicted from their own amino acid sequences in the same way. Every protein satisfies the model's
requirement that the amino acid and 3Di sequences have the same length.

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
    ├── sample_proteins.csv
    ├── sample_compounds.csv
    └── sample_cpi.csv
```
