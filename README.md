# Fake Review Detection using BERT

## Overview

This project implements a **Fake Review Detection System** using **BERT (Bidirectional Encoder Representations from Transformers)** to classify product reviews as **Genuine (OR)** or **Fake (CG)**. The model is fine-tuned on a labeled review dataset and leverages contextual embeddings to achieve accurate classification.

In addition to prediction, the system provides **confidence scores** and **explainable insights** by identifying linguistic patterns such as promotional language, excessive punctuation, review length, and specificity.

---

## Features

* Fine-tuned **BERT Base (bert-base-uncased)** model
* Binary classification: Genuine (OR) vs Fake (CG)
* Context-aware review understanding
* Confidence score generation
* Explainable AI with rule-based reasoning
* Model evaluation using Accuracy, Precision, Recall, and F1-Score
* Save and reuse the trained model

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Scikit-learn
* Pandas
* NumPy
* Google Colab

---

## Project Workflow

1. Load the review dataset.
2. Preprocess and combine category, rating, and review text.
3. Encode labels (OR → 0, CG → 1).
4. Split the dataset into training, validation, and test sets.
5. Tokenize text using the BERT tokenizer.
6. Fine-tune the pre-trained BERT model.
7. Evaluate the model using standard classification metrics.
8. Save the trained model.
9. Predict whether new reviews are genuine or fake with confidence and explanations.

---

## Dataset

Each review contains:

* Product Category
* Rating
* Review Text
* Label

  * **OR** – Genuine Review
  * **CG** – Fake Review

---

## Model

* **Architecture:** BERT Base (bert-base-uncased)
* **Framework:** Hugging Face Transformers
* **Task:** Sequence Classification
* **Number of Classes:** 2

---

## Training Configuration

* Epochs: 2
* Learning Rate: 2e-5
* Batch Size (Training): 32
* Batch Size (Evaluation): 64
* Weight Decay: 0.01
* Optimizer: AdamW (used internally by the Hugging Face Trainer)
* Mixed Precision (FP16): Enabled when GPU is available

---

## Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## Explainability

The system provides human-readable explanations by detecting:

* Promotional phrases
* Excessive exclamation marks
* ALL-CAPS words
* Very short reviews
* Lack of specific usage details
* Repeated words

---

## Project Structure

```text
fake-review-detection/
│
├── dataset/
│   └── fake_reviews_dataset.csv
│
├── fake_review_bert_model/
│
├── results/
│
├── notebooks/
│   └── Fake_Review_Detection.ipynb
│
├── README.md
│
└── requirements.txt
```

---

## Installation

```bash
pip install transformers datasets torch scikit-learn pandas numpy
```

---

## Usage

1. Clone the repository.
2. Install the required dependencies.
3. Update the dataset path in the notebook or script.
4. Run the training pipeline.
5. Save the trained model.
6. Use the prediction function to classify new reviews.

Example:

```python
print_prediction(
    "Electronics_5",
    5.0,
    "The battery lasts about 8 hours and performs consistently."
)
```

---

## Future Improvements

* Fine-tune larger Transformer models such as RoBERTa or DeBERTa.
* Deploy as a web application using Flask or Streamlit.
* Add multilingual fake review detection.
* Integrate explainable AI techniques such as SHAP or LIME.
* Optimize for real-time prediction through model compression.

---

## Author

**Shreyashi Ghosh**

B.Tech in Computer Science & Engineering

---

## License
MIT License

This project is intended for educational and research purposes.
