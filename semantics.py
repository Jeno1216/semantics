import streamlit as st
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")

# Define function to preprocess text data
def preprocess(text):
    # Tokenize the text
    doc = nlp(text)
    # Remove stop words, punctuations and convert to lowercase
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    # Return the processed tokens as a string
    return " ".join(tokens)

# Define function to train and test the model
def train_test_model(df):
    # Preprocess the text data
    df['text'] = df['text'].apply(preprocess)
    # Create bag of words representation using CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['text'])
    y = df['label'].values
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train a logistic regression model
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    # Predict on the test set
    y_pred = clf.predict(X_test)
    # Evaluate the model performance
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    # Print the evaluation metrics
    st.write("Accuracy:", acc)
    st.write("Confusion Matrix:", cm)
    st.write("Classification Report:", cr)

# Define function to create the survey and save the result to a CSV file
def create_survey():
    # Create a survey with a single question and two choices
    survey_question = st.text_input("Please enter your survey question:")
    survey_choice1 = st.text_input("Choice 1:")
    survey_choice2 = st.text_input("Choice 2:")
    survey_choice = st.radio("Choose one of the following:", (survey_choice1, survey_choice2))
    survey_comment = st.text_input("Please provide a brief explanation for your choice:")
    survey_result = pd.DataFrame({'text': [survey_comment], 'label': [int(survey_choice == survey_choice1)]})
    survey_result.to_csv("survey_result.csv", index=False)

# Define function to load the survey result and train/test the model
def load_survey():
    # Load the survey result from the CSV file
    df = pd.read_csv("survey_result.csv")
    # Train and test the model
    train_test_model(df)

# Create a Streamlit app to run the NLP sentiment analysis
st.title("NLP Sentiment Analysis")
option = st.sidebar.selectbox("Select an option:", ["Create Survey", "Load Survey"])
if option == "Create Survey":
    create_survey()
else:
    load_survey()
