import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Load FastText pre-trained embeddings
def load_fasttext_model(path="cc.en.300.bin"):
    return fasttext.load_model(path)


# Compute Sentence Embedding
def sentence_embedding(sentence, model):
    words = sentence.split()
    word_vectors = [model.get_word_vector(word) for word in words if word in model]
    if not word_vectors:  # Handle edge case where no words are in the model
        return np.zeros(model.get_dimension())
    return np.mean(word_vectors, axis=0)


# Match User Question to FAQ
def fasttext_match(user_question, faqs, model):
    user_embedding = sentence_embedding(user_question, model)
    faq_embeddings = [sentence_embedding(faq[1], model) for faq in faqs]
    similarities = cosine_similarity([user_embedding], faq_embeddings).flatten()
    best_match_idx = similarities.argmax()
    return faqs[best_match_idx] if similarities[best_match_idx] > 0.7 else None


# Example Usage
if __name__ == "__main__":
    # Example FAQs
    faqs = [
        (1, "What is your return policy?", "Our return policy is..."),
        (2, "How do I track my order?", "You can track your order...")
    ]
    
    # Load the FastText model
    model = load_fasttext_model("cc.en.300.bin")  # Path to pre-trained model
    
    # User question
    user_question = "Can you explain your return policies?"
    
    # Find the best match
    best_match = fasttext_match(user_question, faqs, model)
    
    if best_match:
        print(f"Matched FAQ: {best_match[1]}")
        print(f"Answer: {best_match[2]}")
    else:
        print("No matching FAQ found.")
        # Fallback system to handle unmatched queries.