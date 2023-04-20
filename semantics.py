import streamlit as st
import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Load the English language model for spaCy
nlp = spacy.load('en_core_web_sm')

# Define a function to preprocess the text
def preprocess_text(text):
    # Tokenize the text
    doc = nlp(text)
    # Remove stop words and lemmatize the tokens
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    # Rejoin the tokens into a string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Set the title of the web app
st.title('Sentiment Analysis with Logistic Regression')

# Load the dataset
data = pd.read_csv('semantics.csv')

# Preprocess the text data
data['text'] = data['text'].apply(preprocess_text)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display the accuracy score
st.write('Model Accuracy:', accuracy)

# Create a survey with one question and two choices (agree or disagree)
survey_question = st.selectbox('Do you agree or disagree?', ('Agree', 'Disagree'))

# Get the respondent's explanation for their choice
explanation = st.text_input('Why do you feel this way?')

# Preprocess the respondent's input
processed_explanation = preprocess_text(explanation)

# Make a prediction with the model
prediction = model.predict([processed_explanation])[0]

# Display the prediction
if prediction == 1:
    st.write('Our model predicts that you agree with the statement.')
else:
    st.write('Our model predicts that you disagree with the statement.')
