# Do Large Language Models Think Like the Brain? Sentence-Level Evidences from Layer-Wise Embeddings and fMRI (AAAI 2026)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Conference](https://img.shields.io/badge/Conference-AAAI%202026-green.svg)](https://aaai.org/)

> **Abstract**: This study explores the alignment between the internal representations of Large Language Models (LLMs) and neural activity in human brain language processing regions. Through systematic analysis of 14 different large language models combined with fMRI neuroimaging data, we found that intermediate layer representations of LLMs show significant correlations with brain language region activity, and different model architectures and fine-tuning strategies exhibit unique brain alignment characteristics.

**Research Highlights**:
- 🧠 First systematic study of alignment between LLM layer-wise representations and human brain language regions during sentence processing
- 📊 Covers 14 mainstream large language models (Chinese/English, Base/Instruct versions)
- 🔬 Based on high-quality fMRI data from 34 participants (1577 sentences)
- 🎯 12 Federenko language localizer brain regions + whole-brain analysis of 90 AAL regions
- 💻 Complete open-source code and detailed documentation with high reproducibility

## 🎯 Project Overview

This study aims to answer the following core questions:

1. **Can large language models capture semantic representations similar to the human brain?**
2. **What is the correlation between LLM representations of different languages (Chinese, English) and corresponding brain region activity?**
3. **Which layers of LLMs are closest to the human brain's language processing mechanisms?**

We explore these questions through two studies:

- Performance evaluation of multiple LLMs on semantic similarity tasks
- Alignment analysis between LLM representations and fMRI neuroimaging data

## 🔬 Key Findings

- ✅ Intermediate layer representations of large language models show significant correlations with human brain language region activity
- ✅ Different model architectures exhibit different brain alignment characteristics
- ✅ Instruct fine-tuned models differ from Base models in brain alignment

## 📚 Research Content

### Semantic Similarity Analysis

Evaluating 14 different large language models on semantic similarity tasks, including:

- **Chinese Models**: Baichuan-7B, Qwen2.5-7B, GLM-4-9B, etc.
- **English Models**: LLaMA-3.1-8B, Mistral-7B, Gemma-2-9B, etc.
- **Multilingual Models**: BERT-base-uncased, DeepSeek, etc.

**Main Tasks**: 
- Calculate embedding similarity between reference and candidate sentences
- Compare semantic understanding capabilities of different models
- Analyze the impact of CLS token vs average pooling strategies

### fMRI Neuroimaging Alignment Study

Analyzing alignment between LLM representations and human brain language region activity using fMRI data:

- **Participant Data**: 34 Chinese participants
- **Brain ROIs**: 12 language-related brain regions (based on Federenko language localizer system)
  - Left hemisphere: IFGorb, IFG, MFG, AntTemp, PostTemp, AngG
  - Right hemisphere: IFGorb, IFG, MFG, AntTemp, PostTemp, AngG
- **Experimental Materials**: "The Little Prince" audiobook
- **Analysis Method**: Ridge regression + 5-fold cross-validation



## ⚙️ Environment Setup


### Installation Steps

1. **Clone Repository**

```bash
git clone https://github.com/Lucasuuu02/LLM4Brain-Release.git
cd LLM4Brain-Release
```

2. **Create Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Install AFNI (for fMRI data preprocessing)**

Please refer to [AFNI Official Installation Guide](https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/background_install/main_toc.html)

## 🚀 Quick Start

### Semantic Similarity Analysis

**Important: Configure paths before running!**

Before running, you need to modify the path configuration in `Semantic_Similarity_Eval/code/sim.py`:

```python
# Modify the following configuration in the main function of sim.py
root_path = './path/to/your/data'  # Change to your data root directory path
model_path = './path/to/your/models'  # Change to your model root directory path
```

Then run:

```bash
cd Semantic_Similarity_Eval/code
python sim.py
```

**Configuration Instructions**:
- Change `root_path` in the code to your data path
- Change `model_path` in the code to your model path
- Select models to evaluate in `model_list`
- Results will be saved in the project's `results` directory

### fMRI Alignment Analysis

**Data Preparation**: Download fMRI raw dataset first: https://openneuro.org/datasets/ds003643  
Place the downloaded data in the `LLM_fMRI_Alignment/data` directory

#### Step 1: GLM Analysis

```bash
cd LLM_fMRI_Alignment/code/GLM
python glm.py
```

This will perform first-level GLM analysis on each run of each participant to extract beta values for each sentence.

#### Step 2: Calculate LLM-fMRI Correlation

**Important: Configure paths before running!**

Before running, you need to modify the path configuration in `LLM_fMRI_Alignment/code/llm_r.py`:

```python
# Modify embedding_list in llm_r.py
embedding_list = [
    './embeddings/embedding_CN_gemma-2-9b-it/',  # Change to your embedding file path
    # Add more model paths...
]
```

Similarly, if using `Rcv.py`, you also need to modify the `embedding_list` configuration.

Then run:

```bash
cd LLM_fMRI_Alignment/code
python llm_r.py
```

This will calculate the correlation coefficients between LLM layer representations and fMRI data.



## 📊 Data Description

### Data Organization Structure

**This repository provides**: Text corpus (sentence data)

```
Project Root/
├── data/
│   └── process_sentences_data/    # Sentence data
│       ├── lppCN_all_sentense.csv  # Chinese sentences
│       ├── lppEN_all_sentense.csv  # English sentences
│       └── ...
├── Semantic_Similarity_Eval/      # Semantic similarity analysis code
├── LLM_fMRI_Alignment/            # fMRI alignment analysis code
└── ...
```

### Data Acquisition

Due to file size limitations, **fMRI raw data needs to be downloaded separately**:

**fMRI Raw Data**: https://openneuro.org/datasets/ds003643  
After downloading, please place in `LLM_fMRI_Alignment/data/` directory

### Data Format Description

- **fMRI Data**: NIfTI format (.nii.gz)
- **LLM embeddings**: NumPy arrays (.npy)
  - Shape: `(n_sentences, n_layers, hidden_size)`
- **ROI masks**: NIfTI format (.nii.gz)

## 📁 Project Structure

```
LLM4Brain-Release/
├── data/                                    # Text data
│   └── process_sentences_data/              # Processed sentence data
│       ├── lppCN_all_sentense.csv           # Chinese sentences (1577 sentences)
│       ├── lppCN_tree.csv                   # Chinese syntax trees
│       ├── lppEN_all_sentense.csv           # English sentences (1577 sentences)
│       └── lppEN_sentence.csv               # English sentence list
│
├── Semantic_Similarity_Eval/                # Semantic similarity evaluation module
│   ├── code/
│   │   └── sim.py                           # Semantic similarity calculation main script
│   └── data/
│       └── data_sentences.csv               # Sentence pair data
│
├── LLM_fMRI_Alignment/                      # LLM-fMRI alignment analysis module
│   ├── code/
│   │   ├── GLM/                             # GLM first-level analysis
│   │   │   ├── CN_onsets/                   # Chinese experiment onset files (9 runs)
│   │   │   ├── EN_onsets/                   # English experiment onset files (9 runs)
│   │   │   ├── masks_new/                   # Brain region ROI masks
│   │   │   │   ├── masks/                   # 90 AAL brain region masks (.nii.gz)
│   │   │   │   └── masklist.txt             # Mask list
│   │   │   ├── glm.py                       # GLM analysis main script (parallel processing)
│   │   │   ├── roi.py                       # ROI beta value extraction script
│   │   │   ├── grey_mask.nii.gz             # Gray matter mask
│   │   │   ├── glm.ipynb                    # GLM interactive analysis notebook
│   │   │   ├── GLM_LSS.ipynb                # LSS single-trial analysis notebook
│   │   │   └── ROI_analysis.ipynb           # ROI extraction notebook
│   │   ├── llm_r.py                         # LLM-fMRI correlation calculation (main script)
│   │   └── Rcv.py                           # Ridge regression cross-validation script
│   ├── data/                                # fMRI raw data (needs to be downloaded separately)
│   │   └── README                           # Data description
│   └── results/                             # Analysis results output directory
│
├── visualizations/                          # Results visualization
│   └── res_vis.ipynb                        # Visualization notebook
│
├── requirements.txt                         # Python dependencies
├── LICENSE                                  # MIT open-source license
└── README.md                                # Project documentation
```

## 💡 Core Module Description

### Module 1: Semantic Similarity Evaluation (`Semantic_Similarity_Eval/`)

**Function**: Evaluate the performance of multiple large language models on semantic similarity tasks

**Main Script**: `sim.py`

**Function Description**:
- Load pre-trained large language models (supports 14 Chinese/English models)
- Extract sentence-level embedding representations (CLS token or average pooling)
- Calculate cosine similarity between reference and candidate sentences
- Cross-model layer analysis of semantic similarity

**Input Data**:
- Chinese/English sentence embeddings (.npy format)
- Sentence pair data (CSV format)

**Output Results**:
- JSON format model performance scores
- Performance of each model on semantic similarity tasks

### Module 2: GLM First-Level Analysis (`LLM_fMRI_Alignment/code/GLM/`)

**Function**: Perform first-level GLM (General Linear Model) analysis on fMRI data

**Main Scripts**:

1. **`glm.py`** - GLM analysis main script
   - Perform first-level GLM analysis on each run of each participant
   - Extract beta values (effect sizes) for each sentence
   - Support parallel processing of multiple participants and runs
   - Output: effect_size.nii.gz file for each sentence

2. **`roi.py`** - ROI beta value extraction
   - Extract average beta values for ROIs from GLM results
   - Support 90 AAL brain region masks
   - Output: CSV format beta values for each run

3. **Jupyter Notebooks**:
   - `glm.ipynb`: Interactive GLM analysis with parallel processing
   - `GLM_LSS.ipynb`: Least Squares Separate (LSS) analysis for single-trial estimation
   - `ROI_analysis.ipynb`: ROI beta value extraction and analysis

**Input Data**:
- fMRI preprocessed data (NIfTI format)
- Onset files (Excel format, annotating start time and duration of each sentence)
- ROI mask files (90 AAL brain regions)

**Output Results**:
- 1stGLM directory: Contains GLM results for each sentence of each participant
- CSV files: ROI beta values for each run

### Module 3: LLM-fMRI Alignment Analysis (`LLM_fMRI_Alignment/code/`)

**Function**: Calculate correlation between LLM representations and fMRI brain activity

**Main Scripts**:

1. **`llm_r.py`** - LLM-fMRI correlation calculation main script
   - Load LLM layer-wise embedding representations
   - Load fMRI beta values
   - Use Ridge regression for prediction
   - Calculate correlation coefficients between each layer embedding and brain region activity
   - Support multi-model parallel processing
   - 5-fold cross-validation ensures result reliability

2. **`Rcv.py`** - Ridge regression cross-validation script (GPU accelerated version)
   - Use closed-form solution to accelerate Ridge regression
   - Grid search for best regularization parameter alpha
   - 10-fold cross-validation
   - Support GPU acceleration (requires CuPy)

**Input Data**:
- LLM embedding: (1577 sentences, n_layers, hidden_size)
- fMRI beta values: (34 participants, 1577 sentences, n_ROIs)

**Output Results**:
- Correlation coefficients for each layer and each ROI
- P-values for each layer and each ROI
- Best regularization parameter alpha
- Visualization charts (correlation curves)

### Module 4: Results Visualization (`visualizations/`)

**Function**: Interactive visualization of analysis results

**Main File**: `res_vis.ipynb`

**Visualization Content**:
- Correlation heatmaps between LLM layers and brain regions
- Cross-model comparative analysis
- Statistical significance test results
- Layer-specific analysis

## 💡 Detailed Usage

### 1. Extract LLM Embeddings

```python
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Load model
model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Extract embeddings
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs, output_hidden_states=True)
    # Return hidden states from all layers
    return [h.detach().numpy() for h in outputs.hidden_states]
```

### 2. Calculate Brain Alignment Correlation

```python
import numpy as np
from sklearn.linear_model import RidgeCV

# Load data (please replace with your actual file paths)
embeddings = np.load("your_embeddings_file.npy")  # (n_samples, n_layers, hidden_size)
fmri_data = np.load("your_fmri_beta_file.npy")     # (n_samples, n_voxels)

# Ridge regression
for layer in range(n_layers):
    X = embeddings[:, layer, :]
    model = RidgeCV(alphas=np.logspace(-3, 1, 10))
    model.fit(X, fmri_data)
    correlation = model.score(X, fmri_data)
    print(f"Layer {layer}: r = {correlation:.3f}")
```

### 3. ROI Analysis and Visualization

```bash
cd LLM_fMRI_Alignment/code/GLM
jupyter notebook ROI_analysis.ipynb
```

Or use visualization module:

```bash
cd visualizations
jupyter notebook res_vis.ipynb
```

## 🧠 Brain ROI Description

This study uses two sets of ROI definitions:

### 1. Federenko Language Localizer System (12 language-related brain regions)

**Left Hemisphere**:
- **LH_IFGorb**: Left Inferior Frontal Gyrus, Orbital part
- **LH_IFG**: Left Inferior Frontal Gyrus
- **LH_MFG**: Left Middle Frontal Gyrus
- **LH_AntTemp**: Left Anterior Temporal
- **LH_PostTemp**: Left Posterior Temporal
- **LH_AngG**: Left Angular Gyrus

**Right Hemisphere**:
- **RH_IFGorb**: Right Inferior Frontal Gyrus, Orbital part
- **RH_IFG**: Right Inferior Frontal Gyrus
- **RH_MFG**: Right Middle Frontal Gyrus
- **RH_AntTemp**: Right Anterior Temporal
- **RH_PostTemp**: Right Posterior Temporal
- **RH_AngG**: Right Angular Gyrus

### 2. AAL (Automated Anatomical Labeling) Template (90 brain regions)

The project provides complete masks for 90 AAL brain regions, including:
- Frontal lobes
- Temporal lobes
- Parietal lobes
- Occipital lobes
- Subcortical structures



## 📊 Experimental Data Description

### Experimental Design

**Task**: "The Little Prince" audiobook listening comprehension task
- **Chinese Version**: 34 participants, 9 runs each
- **English Version**: To be added

**fMRI Parameters**:
- TR (Repetition Time): 2 seconds
- Each run contains multiple sentence stimuli
- Number of sentences: 1577 (corresponding Chinese and English)

**Onset File Format**:
Each Excel file contains the following columns:
- `onset`: Sentence start time (seconds)
- `duration`: Sentence duration (seconds)
- `trial_type`: Trial type
- Other event-related information

### Supported Models

**Chinese Models**:
- Baichuan-7B / Baichuan2-7B-Chat
- Qwen2.5-7B / Qwen2.5-7B-Instruct
- GLM-4-9B / GLM-4-9B-Chat

**English Models**:
- LLaMA-3.1-8B / LLaMA-3.1-8B-Instruct
- Mistral-7B / Mistral-7B-Instruct
- Gemma-2-9B / Gemma-2-9B-it
- OPT-6.7B

**Multilingual Models**:
- BERT-base-uncased
- DeepSeek-R1-Distill-Qwen-7B

## ⚙️ Path Configuration

### Important Notice

Before running the code, please modify the path configurations in the following files:

#### 1. Semantic Similarity Analysis Configuration

Edit `Semantic_Similarity_Eval/code/sim.py`:

```python
# Lines 90-93
root_path = './path/to/your/data'      # Change to your data root directory
model_path = './path/to/your/models'   # Change to your model root directory
```

#### 2. LLM-fMRI Alignment Analysis Configuration

Edit `LLM_fMRI_Alignment/code/llm_r.py`:

```python
# Lines 12-28
embedding_list = [
    './embeddings/embedding_CN_gemma-2-9b-it/',  # Change to actual path
    # Add more models...
]
```

Edit `LLM_fMRI_Alignment/code/Rcv.py` (if using GPU version):

```python
# Lines 4-20
embedding_list = [
    './embeddings/embedding_CN_gemma-2-9b-it_scheme2/',  # Change to actual path
    # Add more models...
]
```


## 📖 Citation

If you use the code or data from this project, please cite:

```bibtex
@article{lei2025large,
  title={Do Large Language Models Think Like the Brain? Sentence-Level Evidence from fMRI and Hierarchical Embeddings},
  author={Lei, Yu and Ge, Xingyang and Zhang, Yi and Yang, Yiming and Ma, Bolei},
  journal={arXiv preprint arXiv:2505.22563},
  year={2025}
}
```


## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- The copyright of "The Little Prince" text used in this study belongs to the original author

---

**Note**: 
- This project is for academic research purposes only
- The use of fMRI data must comply with relevant ethical guidelines
- Large model files and fMRI data are not included in the repository due to size limitations, please download separately via the links above

