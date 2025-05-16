!python -m pip install --upgrade pip

!pip install --upgrade pip

!pip install nltk
!pip install vaderSentiment

import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Make sure to download necessary NLTK data
nltk.download('vader_lexicon')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
texts = [
    "I love this product!",
    "This is the worst experience I've ever had.",
    "Absolutely fantastic service.",
    "I am not happy with this.",
    "Great job!",
    "Terrible, will not buy again.",
    "This made my day.",
    "I hate this so much.",
    "So disappointing.",
    "Very satisfied and happy!",
    "The product arrived yesterday.",
    "It's just okay, nothing special.",
    "I have no strong feelings about this.",
    "Mediocre performance overall.",
    "I used it once and forgot about it.",
    "Packaging was fine.",
    "It's neither good nor bad.",
    "It functions as expected.",
    "Delivery time was average.",
    "This is an acceptable result."
]

labels = [
    "positive", "negative", "positive", "negative",
    "positive", "negative", "positive", "negative",
    "negative", "positive",
    "neutral", "neutral", "neutral", "neutral", "neutral",
    "neutral", "neutral", "neutral", "neutral", "neutral"
]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

model = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=True, stop_words='english')),
    ('clf', LogisticRegression(max_iter=300, class_weight='balanced'))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Model Evaluation:")
print(classification_report(y_test, predictions))

def analyze_sentiment(text):
    prediction = model.predict([text])[0]
    return prediction

if _name_ == "_main_":
    print("\nSentiment Analysis Ready (positive / negative / neutral). Type 'quit' to exit.")
    while True:
        user_input = input("Enter a sentence for sentiment analysis: ")
        if user_input.lower() == 'quit':
            break
        sentiment = analyze_sentiment(user_input)
        print(f"Predicted Sentiment: {sentiment}\n"
