# Quick Start Guide

## English-French Machine Translation Project

### Fast Setup (5 minutes)

#### Step 1: Install Dependencies

Open your terminal and run:

```bash
cd D:\TDTU\llm\midterm
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

#### Step 2: Run the Notebook

```bash
jupyter notebook en_fr_translation.ipynb
```

Or open in Google Colab:
1. Go to https://colab.research.google.com/
2. Upload `en_fr_translation.ipynb`
3. Run all cells

#### Step 3: Wait for Training

- CPU: ~20-30 minutes
- GPU: ~5-10 minutes

The notebook will automatically:
- Download the Multi30K dataset
- Train the model
- Save `best_model.pth`
- Generate `loss_chart.png`
- Calculate BLEU score
- Show translation examples

### Expected Outputs

After running the notebook, you should have:

1. **best_model.pth** - Your trained model checkpoint
2. **loss_chart.png** - Training/validation loss visualization
3. **data/** folder - Downloaded dataset
4. Console output with:
   - Training progress
   - BLEU score
   - Translation examples
   - Error analysis

### Key Results to Note

When the notebook finishes, make note of:

1. **Final BLEU Score** (should be 15-25)
2. **Best Validation Loss**
3. **Number of epochs trained**
4. **5 Translation Examples** for your report

### Creating Your Report

Use these sections in your PDF report:

#### 1. Introduction (0.5 pages)
- Describe the problem
- Mention the dataset
- State your objectives

#### 2. Model Architecture (1 page)
- Draw the Encoder-Decoder diagram
- List hyperparameters
- Explain the architecture

#### 3. Training Process (1 page)
- Include the loss chart from `loss_chart.png`
- Describe training configuration
- Mention early stopping

#### 4. Results (1.5 pages)
- Report BLEU score
- Show 5 translation examples from the notebook output
- Create a table:

| Example | Source (EN) | Reference (FR) | Prediction (FR) | BLEU |
|---------|-------------|----------------|-----------------|------|
| 1       | ...         | ...            | ...             | ...  |
| 2       | ...         | ...            | ...             | ...  |

#### 5. Error Analysis (1.5 pages)
For each of the 5 examples, identify:
- Error type (OOV, long sentence, grammar, etc.)
- Why it occurred
- How to fix it

#### 6. Improvements (0.5 pages)
Suggest:
- Attention mechanism
- Beam search
- BPE tokenization
- Larger dataset
- Model architecture improvements

#### 7. Conclusion (0.5 pages)
- Summarize your results
- Mention limitations
- Future work

#### 8. Appendix (as needed)
- Include the entire source code from the notebook
- Make sure code is well-formatted

### Troubleshooting

#### Problem: "No module named 'spacy'"
**Solution:**
```bash
pip install spacy
```

#### Problem: "Can't find model 'en_core_web_sm'"
**Solution:**
```bash
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

#### Problem: CUDA out of memory
**Solution:**
- Reduce BATCH_SIZE to 32
- Or run on CPU (slower but works)

#### Problem: Dataset download fails
**Solution:**
- Check internet connection
- Try running the download cell again
- Or download manually from: https://github.com/multi30k/dataset

### Testing Your translate() Function

After training, test with custom sentences:

```python
# Test the translate function
test_sentences = [
    "A man is riding a bike.",
    "The dog is running in the park.",
    "She is reading a book."
]

for sentence in test_sentences:
    translation = translate(sentence, model, en_vocab, fr_vocab, tokenize_en, device)
    print(f"EN: {sentence}")
    print(f"FR: {translation}\n")
```

### Submission Checklist

Before submitting, make sure you have:

- [ ] Executable notebook from start to finish
- [ ] `best_model.pth` file
- [ ] PDF report (5-10 pages)
- [ ] Loss chart in report
- [ ] BLEU score in report
- [ ] 5 translation examples with analysis
- [ ] Source code in appendix
- [ ] All code is well-commented
- [ ] Report has citations
- [ ] Architectural diagram
- [ ] Combined into 1 PDF file

### Time Estimates

- Setup: 5 minutes
- Training: 10-30 minutes (depending on hardware)
- Testing: 5 minutes
- Writing report: 2-3 hours
- **Total: 3-4 hours**

### Tips for Success

1. **Start early** - Don't wait until the deadline
2. **Run on GPU** if possible (Google Colab free tier has GPUs)
3. **Save checkpoints** - The notebook already does this
4. **Document everything** - Take screenshots of outputs
5. **Test translation** on various sentences
6. **Explain errors** thoroughly in your report

### Getting Help

If you encounter issues:

1. Check the error message carefully
2. Read the documentation in README.md
3. Review the notebook comments
4. Search for the error online
5. Ask your instructor or classmates

### Good Luck!

This implementation is complete and ready to run. Just execute the notebook, collect the results, and write your report. You should achieve a BLEU score of 15-25, which is expected for this architecture and dataset.
