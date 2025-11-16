# Report Template

## English-French Machine Translation with Encoder-Decoder LSTM

---

### Student Information
- **Name**: [Your Name]
- **Student ID**: [Your ID]
- **Course**: Machine Learning
- **Instructor**: [Instructor Name]
- **Date**: November 2025

---

## Table of Contents
1. Introduction
2. Related Work
3. Dataset
4. Model Architecture
5. Implementation Details
6. Training Process
7. Results
8. Error Analysis
9. Suggested Improvements
10. Conclusion
11. References
12. Appendix: Source Code

---

## 1. Introduction (0.5 pages)

### Background
Machine translation is the task of automatically converting text from one language to another. Traditional approaches relied on rule-based systems or statistical methods. With the advent of deep learning, neural machine translation (NMT) has become the dominant paradigm.

### Problem Statement
In this project, we implement a sequence-to-sequence (seq2seq) model using LSTM (Long Short-Term Memory) networks to translate English sentences to French. The model follows the encoder-decoder architecture proposed by Sutskever et al. (2014).

### Objectives
1. Implement an Encoder-Decoder LSTM model from scratch using PyTorch
2. Train the model on the Multi30K English-French dataset
3. Evaluate performance using BLEU score
4. Analyze translation errors and suggest improvements

---

## 2. Related Work (0.5 pages)

### Sequence-to-Sequence Learning
Sutskever et al. (2014) introduced the seq2seq architecture for machine translation, using two LSTMs: one for encoding the source sentence and another for decoding to the target language.

### Key Contributions
- **Fixed Context Vector**: The encoder compresses the entire source sentence into a fixed-size vector
- **LSTM**: Handles long-term dependencies better than traditional RNNs
- **Teacher Forcing**: Uses ground truth during training to stabilize learning

### Limitations
- Fixed context vector creates information bottleneck for long sentences
- No mechanism to focus on relevant parts of the source sentence
- Later addressed by attention mechanisms (Bahdanau et al., 2015)

---

## 3. Dataset (0.5 pages)

### Multi30K Dataset
- **Task**: English to French translation
- **Domain**: Image captions (short, descriptive sentences)
- **Size**:
  - Training: 29,000 sentence pairs
  - Validation: 1,000 sentence pairs
  - Test: 1,000 sentence pairs
- **Characteristics**: Short sentences (10-15 words), suitable for basic CPU/GPU

### Example Pairs
| English | French |
|---------|--------|
| A man in a blue shirt is standing on a ladder. | Un homme en chemise bleue est debout sur une échelle. |
| Two young girls are playing in the park. | Deux jeunes filles jouent dans le parc. |

---

## 4. Model Architecture (1 page)

### Overall Architecture

```
Input (English) → Encoder → Context Vector → Decoder → Output (French)
```

### Encoder

**Formula**: `(h_t, c_t) = LSTM(embed(x_t), (h_{t-1}, c_{t-1}))`

**Components**:
- **Embedding Layer**: Maps English tokens to dense vectors (dim = 256)
- **LSTM**: 2-layer bidirectional LSTM with hidden size 512
- **Output**: Final hidden state (h_n, c_n) serves as context vector

**Parameters**:
```
Embedding: [vocab_size_en x 256]
LSTM: [256 → 512, 2 layers]
Total: ~X million parameters
```

### Decoder

**Formula**: `(h_t, c_t) = LSTM(embed(y_{t-1}), (h_{t-1}, c_{t-1}))`
            `p(y_t) = softmax(Linear(h_t))`

**Components**:
- **Embedding Layer**: Maps French tokens to dense vectors (dim = 256)
- **LSTM**: 2-layer LSTM with hidden size 512
- **Linear Layer**: Projects hidden state to vocabulary size
- **Output**: Probability distribution over French vocabulary

**Parameters**:
```
Embedding: [vocab_size_fr x 256]
LSTM: [256 → 512, 2 layers]
Linear: [512 → vocab_size_fr]
Total: ~X million parameters
```

### Architectural Diagram

[INSERT YOUR DIAGRAM HERE - Draw using tools like draw.io, PowerPoint, or hand-drawn and scanned]

```
    English Sentence: "A man rides a bike"
           ↓
    [Tokenization & Embedding]
           ↓
    ┌─────────────────┐
    │     ENCODER     │
    │   LSTM Layers   │
    │   (2 layers)    │
    └─────────────────┘
           ↓
    [Context Vector (h_n, c_n)]
           ↓
    ┌─────────────────┐
    │     DECODER     │
    │   LSTM Layers   │
    │   (2 layers)    │
    └─────────────────┘
           ↓
    [Linear + Softmax]
           ↓
    French Sentence: "Un homme fait du vélo"
```

### Hyperparameters Table

| Parameter | Value |
|-----------|-------|
| Vocabulary Size (EN) | 10,000 |
| Vocabulary Size (FR) | 10,000 |
| Embedding Dimension | 256 |
| Hidden Size | 512 |
| Number of Layers | 2 |
| Dropout | 0.5 |
| Total Parameters | ~15M |

---

## 5. Implementation Details (1 page)

### Data Preprocessing

**Tokenization**:
- English: SpaCy `en_core_web_sm`
- French: SpaCy `fr_core_news_sm`
- Lowercase all tokens

**Vocabulary Building**:
- Limit: 10,000 most common words per language
- Special tokens: `<pad>`, `<unk>`, `<sos>`, `<eos>`
- OOV words mapped to `<unk>`

**Padding and Packing**:
```python
# Pad sequences to same length in batch
src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)

# Pack for efficient LSTM processing
packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=True)
```

### Teacher Forcing

During training, at each decoder step:
- With probability 0.5: use ground truth token (teacher forcing)
- With probability 0.5: use predicted token (scheduled sampling)

```python
if random.random() < 0.5:
    decoder_input = target[:, t]  # Ground truth
else:
    decoder_input = predicted_token  # Prediction
```

---

## 6. Training Process (1 page)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 64 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | CrossEntropyLoss (ignore padding) |
| Gradient Clipping | 1.0 |
| Epochs | 15 (with early stopping) |
| Early Stopping Patience | 3 epochs |
| Scheduler | ReduceLROnPlateau |

### Training Progress

[INSERT YOUR LOSS CHART HERE - from loss_chart.png]

**Description**: The chart shows training and validation loss over epochs. Training loss decreases steadily, while validation loss plateaus after epoch X, triggering early stopping.

### Training Statistics

- **Total Epochs**: [X epochs]
- **Best Epoch**: [Epoch X]
- **Best Validation Loss**: [X.XXX]
- **Training Time**: [XX minutes on CPU/GPU]
- **Final Model Parameters**: [Saved to best_model.pth]

---

## 7. Results (1.5 pages)

### BLEU Score

**Test Set Performance**:
- **BLEU Score**: [XX.XX]
- **Interpretation**: This score indicates [good/moderate/poor] translation quality. For reference, professional human translation typically scores 50-60 BLEU.

### Translation Examples

#### Example 1
- **Source (EN)**: [Insert English sentence]
- **Reference (FR)**: [Insert reference French]
- **Prediction (FR)**: [Insert model prediction]
- **Sentence BLEU**: [XX.XX]
- **Status**: ✓ Correct / ✗ Incorrect

#### Example 2
- **Source (EN)**: [Insert English sentence]
- **Reference (FR)**: [Insert reference French]
- **Prediction (FR)**: [Insert model prediction]
- **Sentence BLEU**: [XX.XX]
- **Status**: ✓ Correct / ✗ Incorrect

#### Example 3
- **Source (EN)**: [Insert English sentence]
- **Reference (FR)**: [Insert reference French]
- **Prediction (FR)**: [Insert model prediction]
- **Sentence BLEU**: [XX.XX]
- **Status**: ✓ Correct / ✗ Incorrect

#### Example 4
- **Source (EN)**: [Insert English sentence]
- **Reference (FR)**: [Insert reference French]
- **Prediction (FR)**: [Insert model prediction]
- **Sentence BLEU**: [XX.XX]
- **Status**: ✓ Correct / ✗ Incorrect

#### Example 5
- **Source (EN)**: [Insert English sentence]
- **Reference (FR)**: [Insert reference French]
- **Prediction (FR)**: [Insert model prediction]
- **Sentence BLEU**: [XX.XX]
- **Status**: ✓ Correct / ✗ Incorrect

---

## 8. Error Analysis (1.5 pages)

### Common Error Types

#### 1. Out-of-Vocabulary (OOV) Words
- **Example**: [Provide example where <unk> appears]
- **Cause**: Word not in 10,000 word vocabulary
- **Impact**: Loss of semantic information
- **Frequency**: [X%] of test samples

#### 2. Long Sentences
- **Example**: [Provide example of long sentence translation]
- **Cause**: Fixed context vector loses information
- **Impact**: Missing words or incorrect translations at sentence end
- **Frequency**: [X%] of test samples

#### 3. Grammatical Errors
- **Example**: [Provide example with grammar error]
- **Cause**: Model doesn't capture French grammar rules
- **Impact**: Incorrect verb conjugation, gender agreement, or word order
- **Frequency**: [X%] of test samples

#### 4. Missing or Extra Words
- **Example**: [Provide example]
- **Cause**: Decoder skip or repetition
- **Impact**: Incomplete or verbose translations
- **Frequency**: [X%] of test samples

### Detailed Analysis of Examples

[For each of your 5 examples above, identify which error type applies and explain why]

---

## 9. Suggested Improvements (1 page)

### 1. Attention Mechanism
- **Description**: Allow decoder to focus on relevant encoder states
- **Expected Improvement**: +5-10 BLEU points
- **Implementation**: Bahdanau or Luong attention
- **Benefit**: Handles long sentences better

### 2. Beam Search Decoding
- **Description**: Explore multiple translation paths instead of greedy selection
- **Expected Improvement**: +2-5 BLEU points
- **Implementation**: Beam size 3-5
- **Benefit**: Finds better overall translations

### 3. Subword Tokenization (BPE)
- **Description**: Break words into subword units
- **Expected Improvement**: +3-7 BLEU points
- **Implementation**: Byte Pair Encoding
- **Benefit**: Handles OOV words and morphology

### 4. Larger Dataset
- **Description**: Train on WMT 2014 (~36M pairs)
- **Expected Improvement**: +10-15 BLEU points
- **Benefit**: More diverse examples, better generalization

### 5. Model Architecture Improvements
- **Bidirectional Encoder**: Encode from both directions
- **More Layers**: 3-4 LSTM layers
- **Larger Hidden Size**: 1024 instead of 512
- **Expected Improvement**: +3-5 BLEU points

---

## 10. Conclusion (0.5 pages)

### Summary
In this project, we successfully implemented an Encoder-Decoder LSTM model for English-French machine translation from scratch using PyTorch. The model achieved a BLEU score of [XX.XX] on the Multi30K test set.

### Key Findings
1. The model successfully learns to translate short sentences
2. Performance degrades on long sentences due to fixed context vector
3. OOV words are a major source of errors
4. Teacher forcing ratio of 0.5 provides good balance

### Limitations
1. Fixed context vector creates information bottleneck
2. Limited vocabulary (10,000 words) causes OOV issues
3. No attention mechanism to focus on relevant parts
4. Small dataset limits generalization

### Future Work
1. Implement attention mechanism
2. Use beam search for decoding
3. Apply subword tokenization (BPE)
4. Train on larger datasets (WMT 2014)
5. Experiment with Transformer architecture

---

## 11. References

1. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. *Advances in Neural Information Processing Systems*, 27.

2. Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *arXiv preprint arXiv:1409.0473*.

3. Luong, M. T., Pham, H., & Manning, C. D. (2015). Effective approaches to attention-based neural machine translation. *arXiv preprint arXiv:1508.04025*.

4. PyTorch Documentation. (2024). torch.nn.LSTM. Retrieved from https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

5. Multi30K Dataset. (2024). GitHub Repository. Retrieved from https://github.com/multi30k/dataset

6. Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). BLEU: a method for automatic evaluation of machine translation. *Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics*, 311-318.

---

## 12. Appendix: Source Code

[INSERT YOUR COMPLETE SOURCE CODE HERE - Copy from en_fr_translation.ipynb]

### A. Environment Setup
```python
# [Your setup code]
```

### B. Data Preparation
```python
# [Your data loading code]
```

### C. Model Architecture
```python
# [Your Encoder, Decoder, Seq2Seq classes]
```

### D. Training
```python
# [Your training loop code]
```

### E. Evaluation
```python
# [Your evaluation and BLEU calculation code]
```

### F. Translation Function
```python
# [Your translate function]
```

---

**End of Report**
