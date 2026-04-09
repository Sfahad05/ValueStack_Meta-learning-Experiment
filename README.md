# Enhanced classification of Human Values in Multiple Apps Store's App Reviews (ValueStack Experiments)

This repository contains datasets and notebooks for identifying and classifying **10 Human Values** from app reviews. The project includes classical ML baselines, deep learning models, Transformer models, and a **stacking ensemble (ValueStacking)** that combines multiple base learners into a stronger meta-model.

This README focuses on the notebooks located in:

- `ValueStack_Experiment/`

---

## Directory Structure

### Data Files (expected by notebooks)
- **combineddataset.xls**: Main dataset used by the ValueStack experiments.
  - Expected columns:
    - `Base_Reviews` (text)
    - `category` (string label)

> Note: Some notebooks apply preprocessing (HTML/URL stripping, lemmatization). Transformer notebooks can run directly on raw text, but still require correct label mapping.

---

## ValueStack_Experiment Notebooks

### 1) Transformer Models (Multiclass)

#### BERT (K-Fold CV + Final Test)
- **`ValueStack_Experiment/K-Fold_BERT_Multiclassification.ipynb`**
- Pipeline:
  1. Stratified split: **Train 85% / Test 15%** (test is frozen)
  2. **5-Fold CV** on Train (85%) using fixed “best HPs”
  3. Retrain final model on full oversampled train split
  4. Evaluate once on the frozen test split
- Saves model to:
  - `./bert_best`

#### RoBERTa (K-Fold CV + Final Test)
- **`ValueStack_Experiment/K-Fold_RoBERTa_Multiclassification.ipynb`**
- Same overall pipeline as BERT
- Saves model to:
  - `./roberta_best`
---

### 2) Deep Learning Baseline (Multiclass)

#### GRU (Grid Search + CV + Final Test)
- **`ValueStack_Experiment/K-FoldGRU_Multiclassification.ipynb`**
- Pipeline:
  1. Text cleaning + tokenization
  2. Stratified split: **Train 85% / Test 15%**
  3. Grid search with inner CV to select hyperparameters
  4. **5-Fold CV** performance estimate using the best HPs
  5. Final retraining on oversampled full train + final evaluation on test
- Saves artifacts:
  - `final_gru_10class_cv.h5`
  - `final_gru_tokenizer_cv.pkl`
---

### 3) Classical ML Baseline (Multiclass)

#### TF-IDF + SVM
- **`ValueStack_Experiment/Train_tfidf_svm.ipynb`**
- Pipeline:
  1. Text preprocessing (tokenization + lemmatization)
  2. Stratified split: **Train 85% / Test 15%**
  3. **5-Fold CV** on train (vectorizer fit only on fold-train; oversampling only on fold-train)
  4. Final retraining on oversampled full train + evaluation on test
- Saves artifacts:
  - `tfidf_svm_final.pkl`
  - `tfidf_vectorizer_final.pkl`
---

### 4) Ensemble (Meta-learning / Stacking)

#### ValueStacking Ensemble for Human Values
- **`ValueStack_Experiment/ValueStacking_Ensemble_for_HumanValues.ipynb`**
- Base models used:
  - GRU
  - BERT
  - RoBERTa
  - TF‑IDF + SVM
- Approach:
  - Generates **OOF (out-of-fold)** predicted probabilities for train split (meta-features)
  - Trains meta-learners and evaluates once on held-out test
- Meta-learners included:
  - **XGBoost** (weighted)
  - Logistic Regression (balanced)
- Saves artifacts:
  - `human_values_stacking_xgb_weighted.pkl` (or LR equivalent)
  - `meta_scaler.pkl`
  - `optimal_thresholds_xgb.pkl` (or LR equivalent)
---

### 5) Sampling Utility (Optional)

#### Random sampling algorithm (balanced category sampling)
- **`ValueStack_Experiment/Random sampling algorithm.ipynb`**
- Builds balanced samples per category (e.g., 40 reviews/category), prioritizing longer reviews.
- Intended for preparing more balanced subsets from different app-store datasets.

---

## Label Mapping (10 Classes)

All ValueStack notebooks use the following mapping:

```python
y_dict = {
  'self-direction': 0, 'stimulation': 1, 'hedonism': 2, 'achievement': 3, 'power': 4,
  'security': 5, 'conformity': 6, 'tradition': 7, 'benevolence': 8, 'universalism': 9
}
```

---

## Requirements

Dependencies are listed in:
- **`requirements.txt`**

Notes:
- Some notebooks use **PyTorch + Transformers** (BERT / RoBERTa)
- Some notebooks use **TensorFlow/Keras** (GRU)
- The ensemble notebook uses both, and may force TensorFlow to CPU on macOS for stability.

---

## Running the Experiments (Recommended Order)

1. Put `combineddataset.xls` in your working directory (or update paths inside notebooks).
2. Run base model notebooks to generate saved models:
   - TF‑IDF+SVM → produces `.pkl`
   - GRU → produces `.h5` + tokenizer `.pkl`
   - BERT → produces `./bert_best`
   - RoBERTa → produces `./roberta_best`
3. Run:
   - `ValueStack_Experiment/ValueStacking_Ensemble_for_HumanValues.ipynb`

---

## Troubleshooting Notes

- If you see warnings like *“Some weights were not initialized …”* during Transformer training, this is normal when initializing a classification head.
- Make sure oversampling happens only on training folds (the notebooks already attempt to do this).
- If running on macOS with MPS: PyTorch may use `mps`, while TensorFlow is often set to CPU in these notebooks.

---

## Maintainer

- **@Sfahad05**
