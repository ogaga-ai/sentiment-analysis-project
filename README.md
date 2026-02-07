# Sentiment Analysis Tool

A general-purpose sentiment analysis tool that classifies text reviews as positive or negative, trained on 238K+ reviews from three domains.

## Project Overview

**Problem:** Businesses, researchers, and individuals need to quickly understand sentiment in large volumes of text data.

**Solution:** A sentiment classifier built with TF-IDF and Logistic Regression, achieving 91.4% accuracy on unseen data.

## Current Results

| Metric | Score |
|--------|-------|
| Accuracy | 91.4% |
| Precision (positive) | 92% |
| Recall (positive) | 94% |
| F1 Score (overall) | 91% |

## Target Users

| User                   | Use Case                            |
| ---------------------- | ----------------------------------- |
| Small business owners  | Scan customer feedback quickly      |
| E-commerce sellers     | Monitor product reviews at scale    |
| Product managers       | Track feature reception post-launch |
| Customer support       | Prioritize urgent/negative tickets  |
| Marketing teams        | Measure campaign sentiment          |
| Researchers / Students | Analyze public opinion datasets     |

## Features

- [x] Multi-domain training (Amazon, IMDB, Yelp)
- [x] Text preprocessing pipeline (HTML removal, normalization)
- [x] TF-IDF + Logistic Regression baseline (91.4% accuracy)
- [x] Model evaluation (precision, recall, F1, confusion matrix)
- [x] Custom review prediction
- [ ] Model comparison (Naive Bayes, SVM, Random Forest)
- [ ] Deep learning (DistilBERT fine-tuning)
- [ ] Explainability (SHAP/LIME)
- [ ] Bias analysis
- [ ] Streamlit web app

## Project Structure

```
sentiment-analysis-project/
├── data/
│   ├── raw/               # Original datasets (not tracked)
│   └── processed/         # Visualizations and cleaned data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   └── 03_baseline_model.ipynb
├── models/
│   ├── logistic_regression_baseline.pkl
│   └── tfidf_vectorizer.pkl
├── README.md
├── requirements.txt
└── .gitignore
```

## Datasets

| Dataset        | Domain         | Size Used   | Source                                                                                      |
| -------------- | -------------- | ----------- | ------------------------------------------------------------------------------------------- |
| IMDB Reviews   | Entertainment  | 50,000      | [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |
| Amazon Reviews | E-commerce     | 100,000     | [Kaggle](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)                       |
| Yelp Reviews   | Local business | 88,638      | [Kaggle](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset)                         |
| **Combined**   | **Multi-domain** | **238,638** |                                                                                           |

## Installation

```bash
# Clone the repository
git clone https://github.com/ogaga-ai/sentiment-analysis-project.git
cd sentiment-analysis-project

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
import joblib

# Load the saved model and vectorizer
model = joblib.load('models/logistic_regression_baseline.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

# Predict sentiment on any text
review = "This product is absolutely amazing!"
review_tfidf = tfidf.transform([review])
prediction = model.predict(review_tfidf)

print(prediction)  # ['positive']
```

## Limitations & Ethical Considerations

This model has known limitations:

- **Sarcasm/Irony:** May misclassify sarcastic statements
- **Dialect bias:** Trained primarily on standard English
- **Domain shift:** Performance varies across different review types
- **Neutral text:** Forced into positive/negative — no neutral option
- **Context:** Cannot understand broader context or nuance

### Responsible Use

This tool should NOT be used for:

- Surveillance of individuals without consent
- Automated decision-making without human oversight
- Suppressing legitimate negative feedback

## License

MIT License

## Acknowledgments

- Kaggle for datasets
- scikit-learn for ML tools
