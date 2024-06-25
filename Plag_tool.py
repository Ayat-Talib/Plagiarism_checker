import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Process text to remove stop words and punctuation
    tokens = [token.lemma_ for token in nlp(text.lower()) if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def calculate_similarity(texts):
    # Preprocess each text in the list
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Convert the texts into TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    
    # Compute the cosine similarity between the vectors
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Calculate the average similarity for each document pair
    total_similarity = 0
    num_pairs = 0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            total_similarity += similarity_matrix[i][j]
            num_pairs += 1
    
    average_similarity = (total_similarity / num_pairs) * 100 if num_pairs != 0 else 0
    return average_similarity

def on_submit():
    texts = [text.get("1.0", tk.END).strip() for text in text_boxes if text.get("1.0", tk.END).strip()]
    plagiarism_percentage = calculate_similarity(texts)
    result = f"The average plagiarism percentage among the documents is: {plagiarism_percentage:.2f}%"
    
    messagebox.showinfo("Plagiarism Results", result)

# Initialize the main window
root = tk.Tk()
root.title("Plagiarism Checker")
root.geometry("400x230")
root.configure(bg="Gray")

# Instructions label
instructions = tk.Label(root, text="Enter the text for each document below:", bg="black", fg="white", width=20)
instructions.pack(fill="x")

# Create text boxes for document input
text_boxes = []
for _ in range(2):  # Create 2 text boxes for input, adjust number as needed
    text_box = tk.Text(root, height=3, width=40, bg="lightgray", fg="black", bd=2)
    text_box.pack(pady=5)
    text_boxes.append(text_box)

# Create a submit button
submit_button = tk.Button(root, text="Submit Documents", command=on_submit)
submit_button.pack(pady=20)

# Run the application
root.mainloop()
