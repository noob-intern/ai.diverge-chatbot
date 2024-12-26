import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1) Example Knowledge Base (FAQ entries)
knowledge_base = [
    "How do I place an order on your website?",
    "Which payment methods do you accept?",
    "Is there a minimum order value?",
    "Where can I track my order?",
    "What is your return policy?",
    "Do I need to pay for return shipping?",
    "What shipping options do you offer?",
    "How can I file a warranty claim?",
    "Do you ship internationally?",
    "How do I redeem a promotional code?"
]

# 2) Create TF-IDF Matrix
#    Each knowledge_base entry is treated like a "document."
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(knowledge_base)  # Shape: [num_docs, num_features]

# Example user queries
user_queries = [
    "Can I return a product that I don't like?",
    "Do you accept PayPal for payment?",
    "Where do I find the shipping details?",
    "Show me how to create an AI model."
]

# 3) Loop through each user query
for user_query in user_queries:
    # Transform the query into the same TF-IDF space
    query_vec = vectorizer.transform([user_query])  # shape: [1, num_features]

    # 4) Compute cosine similarity with each FAQ
    #    cosine_similarity returns a [1 x len(knowledge_base)] array
    similarities = cosine_similarity(query_vec, X).flatten()

    # 5) Find the best match
    best_idx = np.argmax(similarities)  # index of highest similarity
    best_score = similarities[best_idx]
    best_match = knowledge_base[best_idx]

    # 6) Print results
    print("-----------------------------------")
    print(f"User Query   : {user_query}")
    print(f"Best Match   : {best_match}")
    print(f"Best Score   : {best_score:.4f}")

    # Optional: Threshold check
    THRESHOLD = 0.2  # Example threshold; adjust per domain
    if best_score >= THRESHOLD:
        print("→ Confidence is HIGH. Serve answer from knowledge base.\n")
    else:
        print("→ Confidence is LOW. Consider fallback.\n")