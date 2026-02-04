# Sentiment Analysis Tool

A general-purpose sentiment analysis tool that classifies text (reviews, feedback, comments) as positive neutral or negative.

## Project Overview

**Problem:** Businesses, researchers, and individuals need to quickly understand sentiment in large volumes of text data.

**Solution:** develop a user-friendly tool that analyzes text sentiment with explainability and bias-awareness.

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

- Single text analysis (paste and predict)
- Batch CSV upload for bulk analysis
- Confidence scores with predictions
- Word highlighting (explainability via SHAP/LIME)
- Bias analysis and documentation

## Project Structure

```
sentiment-analysis-project/
├── data/
│   ├── raw/           # Original datasets
│   └── processed/     # Cleaned datasets
├── notebooks/         # Jupyter notebooks for exploration
├── src/               # Source code
├── models/            # Saved model files
├── app/               # Streamlit application
├── README.md
├── requirements.txt
└── .gitignore
```

## Datasets

| Dataset        | Domain         | Size        | Source                                                                                      |
| -------------- | -------------- | ----------- | ------------------------------------------------------------------------------------------- |
| Amazon Reviews | E-commerce     | 4M reviews  | [Kaggle](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)                       |
| Yelp Reviews   | Local business | 5M reviews  | [Kaggle](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset)                         |
| IMDB Reviews   | Entertainment  | 50K reviews | [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd sentiment-analysis-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
# Example usage (after training)
from src.model import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.predict("This product is amazing!")
print(result)  # {'sentiment': 'positive', 'confidence': 0.95}
```

## Limitations & Ethical Considerations

This model has known limitations:

- **Sarcasm/Irony:** May misclassify sarcastic statements
- **Dialect bias:** Trained primarily on standard English
- **Domain shift:** Performance varies across different review types
- **Context:** Cannot understand broader context or nuance

### Responsible Use

This tool should NOT be used for:

- Surveillance of individuals without consent
- Automated decision-making without human oversight
- Suppressing legitimate negative feedback

For detailed bias analysis, see [notebooks/07_bias_analysis.ipynb](notebooks/07_bias_analysis.ipynb).

## License

MIT License

## Acknowledgments

- Hugging Face for transformer models
- Kaggle for datasets
- Research papers cited in documentation
