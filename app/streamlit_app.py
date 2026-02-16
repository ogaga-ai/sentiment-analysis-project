import streamlit as st
import joblib
import os
import re

# --- Page Config ---
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎯",
    layout="centered",
)

# --- Load Model ---
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = joblib.load(os.path.join(base, "models", "logistic_regression_baseline.pkl"))
    tfidf = joblib.load(os.path.join(base, "models", "tfidf_vectorizer.pkl"))
    return model, tfidf

model, tfidf = load_model()

# --- Text Cleaning (same as training pipeline) ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Header ---
st.title("Sentiment Analyzer")
st.markdown("Classify any review as **positive** or **negative** — powered by a model trained on 238K+ reviews from IMDB, Amazon, and Yelp.")

st.divider()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Single Review", "Batch Analysis", "About"])

# --- Tab 1: Single Review ---
with tab1:
    review = st.text_area(
        "Paste a review below:",
        height=120,
        placeholder="e.g. This product exceeded my expectations. Great quality and fast shipping!"
    )

    if st.button("Analyze", type="primary", use_container_width=True):
        if review.strip():
            cleaned = clean_text(review)
            vectorized = tfidf.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            probabilities = model.predict_proba(vectorized)[0]
            confidence = max(probabilities) * 100

            col1, col2 = st.columns(2)
            with col1:
                if prediction == "positive":
                    st.success(f"**Positive**", icon="👍")
                else:
                    st.error(f"**Negative**", icon="👎")
            with col2:
                st.metric("Confidence", f"{confidence:.1f}%")

            # Show probability breakdown
            neg_prob, pos_prob = probabilities
            st.markdown("**Probability breakdown:**")
            col_neg, col_pos = st.columns(2)
            with col_neg:
                st.progress(neg_prob, text=f"Negative: {neg_prob*100:.1f}%")
            with col_pos:
                st.progress(pos_prob, text=f"Positive: {pos_prob*100:.1f}%")
        else:
            st.warning("Please enter a review to analyze.")

# --- Tab 2: Batch Analysis ---
with tab2:
    st.markdown("Paste multiple reviews — **one per line** — to analyze them all at once.")

    batch_input = st.text_area(
        "Reviews (one per line):",
        height=200,
        placeholder="This product is amazing!\nTerrible quality, broke after one day.\nDecent value for the price.",
        key="batch"
    )

    if st.button("Analyze All", type="primary", use_container_width=True, key="batch_btn"):
        reviews = [r.strip() for r in batch_input.strip().split("\n") if r.strip()]
        if reviews:
            cleaned = [clean_text(r) for r in reviews]
            vectorized = tfidf.transform(cleaned)
            predictions = model.predict(vectorized)
            probabilities = model.predict_proba(vectorized)

            # Summary
            pos_count = sum(1 for p in predictions if p == "positive")
            neg_count = len(predictions) - pos_count

            col1, col2, col3 = st.columns(3)
            col1.metric("Total", len(reviews))
            col2.metric("Positive", pos_count)
            col3.metric("Negative", neg_count)

            st.divider()

            # Individual results
            for i, (review, pred, probs) in enumerate(zip(reviews, predictions, probabilities)):
                confidence = max(probs) * 100
                short = review[:80] + "..." if len(review) > 80 else review
                icon = "👍" if pred == "positive" else "👎"
                st.markdown(f"{icon} **{pred.upper()}** ({confidence:.0f}%) — {short}")
        else:
            st.warning("Please enter at least one review.")

# --- Tab 3: About ---
with tab3:
    st.markdown("""
    ### How It Works

    This tool uses a **TF-IDF + Logistic Regression** model trained on **238,638 reviews**
    from three domains:

    | Source | Reviews | Domain |
    |--------|---------|--------|
    | IMDB | 50,000 | Movies |
    | Amazon | 100,000 | Products |
    | Yelp | 88,638 | Businesses |

    ### Performance

    | Metric | Score |
    |--------|-------|
    | Accuracy | **91.4%** |
    | Precision (positive) | 92% |
    | Recall (positive) | 94% |
    | F1 Score (positive) | 93% |

    ### Limitations

    - **Sarcasm/irony** may be misclassified
    - **Neutral reviews** are forced into positive or negative
    - Trained on **English text only**
    - Performance may vary across different review domains

    ### Source Code

    [GitHub Repository](https://github.com/ogaga-ai/sentiment-analysis-project)

    ---
    *Built by Ogaga Okedi*
    """)

# --- Footer ---
st.divider()
st.caption("Trained on 238K+ reviews from IMDB, Amazon, and Yelp | 91.4% accuracy | [GitHub](https://github.com/ogaga-ai/sentiment-analysis-project)")
