# English-French Machine Translation with Encoder-Decoder LSTM

**TDTU - Machine Learning Midterm Project**
**Assignment: English to French Translation using LSTM**

## Project Overview

This project implements a **from-scratch Encoder-Decoder LSTM model** for English to French machine translation using PyTorch. The model uses a **fixed context vector** approach without attention mechanism, following the classic sequence-to-sequence architecture.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Evaluation](#evaluation)
- [Results](#results)
- [Error Analysis](#error-analysis)
- [Improvements](#improvements)

## Requirements

- Python 3.8+
- PyTorch 1.13+
- SpaCy 3.5+
- NLTK
- Matplotlib
- NumPy

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Download SpaCy Models

```bash
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

### Alternative: Install Manually

```bash
pip install spacy torch torchtext nltk matplotlib requests
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

## Dataset

**Multi30K English-French Dataset**

- **Training set**: 29,000 sentence pairs
- **Validation set**: 1,000 sentence pairs
- **Test set**: 1,000 sentence pairs
- **Source**: [Multi30K GitHub Repository](https://github.com/multi30k/dataset)

The dataset will be automatically downloaded when you run the notebook.

## Project Structure

```
midterm/
├── en_fr_translation.ipynb    # Main implementation notebook
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/                       # Dataset directory (auto-created)
│   ├── train.en
│   ├── train.fr
│   ├── val.en
│   ├── val.fr
│   ├── test.en
│   └── test.fr
├── best_model.pth             # Best model checkpoint (after training)
└── loss_chart.png             # Training/validation loss chart (after training)
```

## How to Run

### Option 1: Run in Jupyter Notebook

1. Open the notebook:
   ```bash
   jupyter notebook en_fr_translation.ipynb
   ```

2. Run all cells sequentially (Runtime → Run all)

3. The notebook will:
   - Download and prepare the dataset
   - Build vocabularies
   - Train the model
   - Save the best model checkpoint
   - Generate loss charts
   - Evaluate BLEU score
   - Perform error analysis

### Option 2: Run on Google Colab

1. Upload `en_fr_translation.ipynb` to Google Colab
2. Run all cells
3. Download the outputs:
   - `best_model.pth`
   - `loss_chart.png`

## Model Architecture

### Encoder
- **Embedding**: Converts English tokens to dense vectors (dim=256)
- **LSTM**: 2-layer LSTM with hidden size 512
- **Dropout**: 0.5 for regularization
- **Output**: Final hidden state (context vector) + all hidden states

### Decoder
- **Embedding**: Converts French tokens to dense vectors (dim=256)
- **LSTM**: 2-layer LSTM with hidden size 512
- **Linear Layer**: Projects hidden states to vocabulary size
- **Dropout**: 0.5 for regularization
- **Teacher Forcing**: 50% probability during training

### Seq2Seq Model
```
Input (English) → Encoder → Context Vector → Decoder → Output (French)
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 256-512 |
| Hidden Size | 512 |
| Number of LSTM Layers | 2 |
| Dropout | 0.5 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss (ignore padding) |
| Teacher Forcing Ratio | 0.5 |
| Epochs | 10-15 |
| Early Stopping Patience | 3 epochs |
| Gradient Clipping | 1.0 |

## Evaluation

### BLEU Score

The model is evaluated using the **BLEU (Bilingual Evaluation Understudy)** metric on the test set:

```python
from nltk.translate.bleu_score import corpus_bleu
bleu_score = corpus_bleu(references, hypotheses)
```

### Translation Function

```python
def translate(sentence: str) -> str:
    """
    Translate English sentence to French

    Process:
    1. Tokenize English sentence
    2. Convert to tensor
    3. Pass through encoder
    4. Greedy decoding with decoder
    5. Stop when <eos> or max_length=50
    6. Detokenize and return French sentence
    """
```

## Results

### Expected Performance

- **BLEU Score**: 15-25 (on Multi30K test set)
- **Training Time**: 15-30 minutes (CPU) / 5-10 minutes (GPU)
- **Model Size**: ~15-20M parameters

### Outputs

1. **best_model.pth**: Trained model checkpoint
2. **loss_chart.png**: Training and validation loss over epochs
3. **BLEU Score**: Printed in notebook
4. **Translation Examples**: 5 examples with error analysis

## Error Analysis

### Common Error Types

#### 1. Rare Words (Out-of-Vocabulary)
- **Problem**: Words not in the 10,000 word vocabulary are replaced with `<unk>`
- **Example**: Proper nouns, technical terms
- **Impact**: Loss of semantic information

#### 2. Long Sentences
- **Problem**: Fixed context vector loses information for long sequences
- **Example**: Sentences > 20 words may lose details
- **Impact**: Missing or incorrect words in translation

#### 3. Grammar and Word Order
- **Problem**: Model may generate grammatically incorrect French
- **Example**: Incorrect verb conjugation, gender agreement
- **Impact**: Unnatural or incorrect translations

#### 4. Missing or Extra Words
- **Problem**: Decoder may skip words or add unnecessary ones
- **Impact**: Incomplete or verbose translations

## Improvements

### Suggested Enhancements (Optional Extensions)

1. **Attention Mechanism** (+5-10 BLEU points)
   - Add Bahdanau or Luong attention
   - Allows decoder to focus on relevant encoder states

2. **Beam Search** (+2-5 BLEU points)
   - Replace greedy decoding with beam search (beam size 3-5)
   - Explores multiple translation paths

3. **Subword Tokenization (BPE)** (+3-7 BLEU points)
   - Use Byte Pair Encoding instead of word-level tokens
   - Handles rare words and morphology better

4. **Larger Dataset** (+10-15 BLEU points)
   - Train on WMT 2014 (~36M pairs)
   - More diverse examples

5. **Model Architecture** (+3-5 BLEU points)
   - Increase LSTM layers (3-4)
   - Increase hidden size (1024)
   - Add bidirectional encoder

## Project Deliverables

### Required Submissions

1. **Source Code**: `en_fr_translation.ipynb` (fully commented and executable)
2. **Model Checkpoint**: `best_model.pth`
3. **PDF Report** (5-10 pages):
   - Architectural diagram
   - Training/validation loss chart
   - BLEU score on test set
   - 5 translation examples with error analysis
   - Suggestions for improvement
   - Source code in Appendix

### Grading Criteria (10 points)

| Criteria | Points |
|----------|--------|
| Correct Encoder-Decoder LSTM model | 3.0 |
| Data processing, DataLoader, padding/packing | 2.0 |
| Stable training with early stopping, checkpoints | 1.5 |
| translate() function works with new sentences | 1.0 |
| BLEU score + loss chart | 1.0 |
| Analysis of 5 error examples + suggestions | 1.0 |
| Clean, commented, well-structured code | 0.5 |
| Complete report with charts, citations | 0.5 |
| **Plus points (extensions)** | **1.0** |

## Important Notes

- The source code **must be executable from start to finish** on Google Colab or local machine
- **DO NOT use** seq2seq libraries like `torchtext.legacy` or `transformers`
- **DO NOT copy code** (0 points)
- Deadline: **November 22, 2025 (23:59)**
- No late submissions accepted

## References

- Sutskever et al. (2014). [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- PyTorch Documentation: [torch.nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- Multi30K Dataset: [GitHub Repository](https://github.com/multi30k/dataset)

## Author

**TDTU Student**
**Course**: Machine Learning
**Semester**: Fall 2025

## License

This project is for educational purposes only.
