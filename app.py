import pandas as pd
import joblib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA_FILE = "twitter_sentiment_dataset.csv"
MODEL_FILE = "sentiment_model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"
METRICS_FILE = "metrics.csv"

# ====== Sentiment Colors ======
SENTIMENT_COLORS = {
    "happy": "white",
    "sad": "blue",
    "fear": "purple",
    "angry": "red",
    "confusion": "orange",
    "supportive": "teal",
    "opposing": "brown",
    "irrelevant": "gray"
}

# ====== Train Model ======
def train_model():
    df = pd.read_csv(DATA_FILE).dropna(subset=['tweet', 'sentiment'])
    X = df['tweet']
    y = df['sentiment']  # Keep string labels

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_vec = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=300)
    model.fit(X_vec, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    return model, vectorizer

# ====== Load or Train ======
def load_model():
    try:
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
    except:
        model, vectorizer = train_model()
    return model, vectorizer

# ====== Update Metrics ======
def update_metrics(correct):
    try:
        metrics = pd.read_csv(METRICS_FILE)
    except:
        metrics = pd.DataFrame(columns=["total", "correct", "accuracy"])
    
    total = metrics["total"].iloc[-1] + 1 if not metrics.empty else 1
    correct_count = metrics["correct"].iloc[-1] + (1 if correct else 0) if not metrics.empty else (1 if correct else 0)
    accuracy = round((correct_count / total) * 100, 2)

    metrics = pd.concat(
        [metrics, pd.DataFrame([[total, correct_count, accuracy]], columns=["total", "correct", "accuracy"])],
        ignore_index=True
    )
    metrics.to_csv(METRICS_FILE, index=False)

# ====== Load Model ======
model, vectorizer = load_model()

# ====== Streamlit UI ======
st.title("Twitter Sentiment Analysis")
st.write("Enter a tweet to predict its sentiment. Feedback will improve the model.")

tweet_input = st.text_area("Tweet text", st.session_state.get("tweet_input", ""))

if st.button("Analyze Sentiment"):
    if tweet_input.strip():
        st.session_state["tweet_input"] = tweet_input
        tweet_vec = vectorizer.transform([tweet_input])
        sentiment_name = str(model.predict(tweet_vec)[0])  # Always string
        st.session_state["prediction"] = sentiment_name
    else:
        st.warning("Please enter a tweet before analyzing.")

# Show prediction & feedback if available
if "prediction" in st.session_state:
    sentiment_name = st.session_state["prediction"]
    color = SENTIMENT_COLORS.get(sentiment_name.lower(), "black")
    st.markdown(
        f"**Predicted Sentiment:** <span style='color:{color}; font-weight:bold;'>{sentiment_name}</span>",
        unsafe_allow_html=True
    )

    # If irrelevant, skip feedback
    if sentiment_name.lower() != "irrelevant":
        feedback = st.radio("Was this prediction correct?", ("Yes", "No"), index=0, key="feedback_choice")

        if feedback == "No":
            correct_label = st.selectbox(
                "Select the correct sentiment:",
                sorted(pd.read_csv(DATA_FILE)['sentiment'].unique()),
                key="correct_label"
            )
        else:
            correct_label = sentiment_name

        topic_input = st.text_input("Topic (optional but recommended)", key="topic_input")

        if st.button("Submit Feedback"):
            df = pd.read_csv(DATA_FILE)
            new_row = pd.DataFrame(
                [[len(df) + 1, topic_input.strip() if topic_input.strip() else "Unknown",
                  st.session_state["tweet_input"], correct_label]],
                columns=['id', 'topic', 'tweet', 'sentiment']
            )
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(DATA_FILE, index=False)

            update_metrics(feedback == "Yes")
            model, vectorizer = train_model()

            st.success("Feedback saved, model retrained, and metrics updated!")

            # 🔹 Clear only relevant session state keys
            for key in ["tweet_input", "prediction", "feedback_choice", "correct_label", "topic_input"]:
                if key in st.session_state:
                    del st.session_state[key]

            # 🔹 Force UI refresh
            try:
                st.rerun()  # For Streamlit >= 1.25
            except AttributeError:
                st.experimental_rerun()  # For older versions
    else:
        st.info("This tweet was classified as irrelevant and will not affect sentiment training.")

# ====== Analytics Dashboard ======
st.markdown("---")
st.header("📊 Model Analytics")

try:
    metrics = pd.read_csv(METRICS_FILE)
    st.subheader("Accuracy Over Time")
    st.line_chart(metrics["accuracy"])
    st.write(f"**Current Accuracy:** {metrics['accuracy'].iloc[-1]}%")
except:
    st.info("No accuracy data yet. Test some tweets first!")

try:
    df = pd.read_csv(DATA_FILE)
    st.subheader("Sentiment Class Distribution")
    st.bar_chart(df['sentiment'].value_counts())
    st.write(f"**Total Tweets in Dataset:** {len(df)}")

    st.subheader("Topic Distribution")
    st.bar_chart(df['topic'].value_counts())
except:
    st.info("Dataset not found yet.")
