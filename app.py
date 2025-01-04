import streamlit as st
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@st.cache_resource
def load_model():
    return SentenceTransformer('./all-MiniLM-L6-v2')  # Path to the local model folder



# Load knowledge base
@st.cache_data
def load_knowledge_base(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_relevant_answer(query, knowledge_base, model):
    query_embedding = model.encode([query])
    scores = {}
    for url, item in knowledge_base.items():
        if item.get('embedding'):
            similarity_score = cosine_similarity(query_embedding, [item['embedding']])[0][0]
            scores[url] = similarity_score
    if not scores:
        return "Sorry, I cannot find an answer for your query."
    best_url = max(scores, key=scores.get)
    if scores[best_url] > 0.3:
        return knowledge_base[best_url]["text"]
    else:
        return "Sorry, I cannot find an answer for your query."

# Streamlit UI
def main():
    st.title("PDF-Based Chatbot")
    st.write("Ask any question related to the PDFs!")

    # Load the model and knowledge base
    model = load_model()
    knowledge_base = load_knowledge_base("knowledge_base.json")

    # User Input
    query = st.text_input("Enter your question:")
    if query:
        response = find_relevant_answer(query, knowledge_base, model)
        st.write("### Bot Response:")
        st.write(response)

if __name__ == "__main__":
    main()
