import tkinter as tk
from tkinter import Label, Text, Button, INSERT, END
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model_path = 'F:/Learn.h5'
model = load_model(model_path)

# Create a Tokenizer for text processing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(["fake", "real"])  # Add any additional words if needed

# Function to predict fake or real news
def predict_news():
    # Get user input
    user_input = news_entry.get("1.0", END)

    # Remove leading and trailing whitespaces
    user_input = user_input.strip()

    # Tokenize and pad the input sequence
    sequences = tokenizer.texts_to_sequences([user_input])
    padded_input = pad_sequences(sequences, maxlen=700, padding='post', truncating='post')

    # Make the prediction
    prediction = model.predict(padded_input)[0][0]

    # Update the result label
    result_label.config(text=f"Prediction: {'Fake' if prediction < 0.5 else 'Real'}")

# GUI setup
root = tk.Tk()
root.title("Fake News Detection")

# Input area
news_label = Label(root, text="Enter News:")
news_label.pack()

news_entry = Text(root, height=5, width=50)
news_entry.pack()

# Prediction button
predict_button = Button(root, text="Predict", command=predict_news)
predict_button.pack()

# Result label
result_label = Label(root, text="Prediction: ")
result_label.pack()

# Run the GUI
root.mainloop()
