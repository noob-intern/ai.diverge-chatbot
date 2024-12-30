from gensim.models import KeyedVectors
import numpy as np


def load_word2vec_model():
    return KeyedVectors.load_word2vec_format('path/to/word2vec.bin', binary=True)


def embed_text(text, model):
    words = text.split()
    embeddings = [model[word] for word in words if word in model]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.vector_size)


def word2vec_match(user_question, faqs, model):
    user_vec = embed_text(user_question, model)
    faq_vecs = [embed_text(faq[1], model) for faq in faqs]
    similarity_scores = [np.dot(user_vec, faq_vec) / (np.linalg.norm(user_vec) * np.linalg.norm(faq_vec)) for faq_vec in faq_vecs]
    best_match_idx = np.argmax(similarity_scores)
    return faqs[best_match_idx] if similarity_scores[best_match_idx] > 0.7 else None


"""
•	Represent FAQs and user queries as vectors by averaging word embeddings.
•	Compute cosine similarity between these vectors.

Pros:
	•	Captures semantic relationships between words.
	•	Handles synonyms better.

Cons:
	•	Requires pre-trained embeddings (e.g., GloVe, Word2Vec).
	•	Slightly computationally intensive.
"""