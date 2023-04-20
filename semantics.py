import pandas as pd
import numpy as np
import spacy
from spacy.util import minibatch, compounding
from sklearn.utils import shuffle
import streamlit as st

# Load the data
@st.cache(persist=True)
def load_data():
    data = pd.read_csv("semantics.csv")  # replace with your data file name
    data = shuffle(data)
    data.reset_index(inplace=True, drop=True)
    return data

# Train the model
@st.cache(allow_output_mutation=True)
def train_model(data, iterations):
    # Load the small English model
    nlp = spacy.load('en_core_web_sm')

    # Create the TextCategorizer with exclusive classes and "bow" architecture
    textcat = nlp.create_pipe(
        "textcat",
        config={
            "exclusive_classes": True,
            "architecture": "bow"
        }
    )

    # Add the TextCategorizer to the pipeline and disable other pipes
    nlp.add_pipe(textcat)
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat"]
    with nlp.disable_pipes(*other_pipes):
        # Initialize the TextCategorizer and add the labels
        textcat.add_label("agree")
        textcat.add_label("disagree")

        # Convert the data into the format expected by the TextCategorizer
        train_data = []
        for i, row in data.iterrows():
            text = row['text']
            text = text.replace('\r\n', '')  # remove '\r\n' characters
            label = row['label']
            if label == 'agree':
                label_dict = {"cats": {"agree": 1, "disagree": 0}}
            elif label == 'disagree':
                label_dict = {"cats": {"agree": 0, "disagree": 1}}
            train_data.append((text, label_dict))

        # Train the TextCategorizer
        losses = {}
        for i in range(iterations):
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=spacy.optimizers.Adam(learning_rate=0.001), losses=losses)
        return nlp


# Load the data
data = load_data()

# Train the model
model = train_model(data, 10)

# Test the model
while True:
    text = input("Enter some text: ")
    doc = model(text)
    print(doc.cats)
