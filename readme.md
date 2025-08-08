# CRABS: A syntactic-semantic pincer strategy for bounding LLM interpretation of Python notebooks

This repository contains the code and data for the paper:

```bibtex
@inproceedings{li2025crabs,
  title={CRABS: A syntactic-semantic pincer strategy for bounding LLM interpretation of Python notebooks},
  author={Li, Meng and McPhillips, Timothy M and Wang, Dingmin and Tsai, Shin-Rong and Lud{\"a}scher, Bertram},
  booktitle={Conference on Language Modeling (COLM)},
  year={2025}
}
```

## Structure of the folder
```
├── data
│   ├── ProspectiveCorrectGivenSpecification.json: human-annotated ground truth
│   ├── builtin_funcs_and_vars.txt
│   ├── crabs_meta.csv
│   ├── inputs: this folder contains 50 notebooks of the CRABS dataset
│   └── outputs
│       ├── informationflowgraph: this folder stores results per notebook for information flow graphs
│       ├── cellexecutiondependencygraph: this folder stores results per notebook for cell execution dependency graphs
│       ├── records: this folder records the prompts together with the LLM's output
│       ├── informationflowgraph.csv: this file merges results from the informationflowgraph folder by computing the average and standard deviation, together with "em" for exact matching
│       └── cellexecutiondependencygraph.csv: this file merges results from the cellexecutiondependencygraph folder by computing the average and standard deviation, together with "em" for exact matching
├── parsers
│   ├── prompts: this folder contains prompts for LLM-based approaches
│   │   ├── baseline.txt
│   │   ├── crabs_subtask1.txt
│   │   ├── crabs_subtask2.txt
│   │   ├── a1.txt
│   │   └── a2.txt
│   ├── baseline.py: baseline approach using an LLM to query the entire notebook
│   ├── crabs.py: our main approach
│   ├── a1.py: ablation study 1 (w/o syntactic parsing)
│   ├── a2.py: ablation study 2 (w/o cell-by-cell prompting)
│   └── yw_generator.py: syntactic processor
├── key
├── environment.yml: for environment setup
├── 1_model_and_eval.py: extract information flow graphs and cell execution dependency graphs for notebooks under the data/inputs folder
├── 2_crabs_llm_acc.py: compute the accuracy for an LLM to resolve ambiguities left by analyzing the syntactic structure of these notebooks
├── 3_metrics_across_notebooks.py: compute the average and standard deviation, together with exact matching across all notebooks by taking files under the informationflowgraph and the cellexecutiondependencygraph folders as inputs
└── readme.md
```

Note for the `builtin_funcs_and_vars.txt` file: built-in functions and variables are recognized through the curated list of [Python version 3.13.2](https://docs.python.org/3/library/functions.html) (retrieved on Mar. 05, 2025)

## Setup
### Installation
1. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

2. Create a Python virtual environment
```
conda env create -f environment.yml -n crabs
conda activate crabs
```

3. Data preparation

- Download the [Meta Kaggle Code](https://www.kaggle.com/datasets/kaggle/meta-kaggle-code/data) dataset[^1][^2], and retrieve notebooks from it based on the `kernelVersionId` column of the `data/crabs_meta.csv` file. Note that the `data/crabs_meta.csv` file contains the meta data of the CRABS dataset, which is created by selecting 50 highly up-voted Python notebooks in [Kaggle](https://www.kaggle.com/code?language=Python) and linking them with `kernelVersionId` by using the [Meta Kaggle](https://www.kaggle.com/datasets/kaggle/meta-kaggle) dataset[^2][^3]. These kernel version IDs can be further used to identify notebooks in the Meta Kaggle Code dataset since they are named as `{kernelVersionId}.ipynb`.

- Move these notebooks under the `data/inputs` folder and rename them as `{crabsId}-{authorName}:{urlSlug}.ipynb` where `crabsId`, `authorName`, and `urlSlug` are columns in the `data/crabs_meta.csv` file.

### Configuration
1. The key for OpenAI: put your OpenAI key in the `key` file

2. Go to the `1_model_and_eval.py` script and update the *Parameters* section as needed

### Execution
After you have installed and configured CRABS, run the following commands in sequence: 
```
python 1_model_and_eval.py
python 2_crabs_llm_acc.py
python 3_metrics_across_notebooks.py
```

### Output Column Mapping and Terminology

The table below maps column names from the output CSV files to the corresponding terms used in the paper for consistency and clarity.

| **CSV Column**      | **Paper Terminology**  |
|---------------------|------------------------|
| `MinIOParser`       | *lower estimate*       |
| `MaxIOParser`       | *upper estimate*       |
| `A1`                | Ablation study S1      |
| `A2`                | Ablation study S2      |


[^1]: Jim Plotts, and Megan Risdal. (2023). Meta Kaggle Code [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DS/3240808
[^2]: Our experiments are based on the dataset Retrieved on Mar. 27, 2025.
[^3]: Megan Risdal, and Timo Bozsolik. (2022). Meta Kaggle [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DS/9
