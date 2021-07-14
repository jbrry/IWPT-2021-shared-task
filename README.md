# IWPT-2021-shared-task


### Installation
Create environment:

```
conda create -n multitask_parser python=3.7
conda activate multitask_parser
```

Install software as of 11/05/2021:

```pip install -r requirements.txt```

Or install AllenNLP:

```
pip install allennlp
pip install allennlp-models
```

###  Download the Shared Task data

```
./scripts/download_iwpt_data.sh
```

###  Run a model
For running a single-task model:
```
./scripts/train_eud_parser.sh fr_sequoia enhanced transformer bert-base-multilingual-cased enhanced_kg_parser
```

For using the multitask components:
```
train_eud_multitask_parser.sh fr_sequoia enhanced transformer bert-base-multilingual-cased enhanced_kg_parser enhanced_dependencies
```

Launch a batch job:
```
sbatch --gres=gpu:rtx6000:1 ./scripts/train_eud_parser.sh en_ewt:bg_btb:pl_pdb enhanced transformer bert-base-multilingual-cased enhanced_kg_parser transformer
```

###  Preprocess test files
To pre-process the blind test data, we use Trankit.

```
conda create -n trankit python=3.7
conda activate trankit
pip install trankit==1.0.1
```

To train a trankit model, see `scripts/trankit.py`

### Citation
If you use this code for your research, please cite these works as:  
```
@misc{barry2021dcuepfl,
    title={The DCU-EPFL Enhanced Dependency Parser at the IWPT 2021 Shared Task},
    author={James Barry and Alireza Mohammadshahi and Joachim Wagner and Jennifer Foster and James Henderson},
    year={2021},
    eprint={2107.01982},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
