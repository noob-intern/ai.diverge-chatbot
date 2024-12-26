from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Example knowledge base
knowledge_base = [
    "How do I place an order on your website?",
    "What is your return policy?",
    "Do you ship internationally?",
    "Which bike lock is the most secure?",
    "How do I file a warranty claim?"
]

# Pre-encode your knowledge base
kb_embeddings = model.encode(knowledge_base, convert_to_tensor=True)

# SWhen a user query arrives, encode it and compute similarity
user_query = "Can you explain how to return a product that I don't like?"
query_embedding = model.encode(user_query, convert_to_tensor=True)

# Calculate cosine similarity scores
cosine_scores = util.cos_sim(query_embedding, kb_embeddings)  # shape: [1, len(kb)]

# Find the best match
# cosine_scores[0] is a list of similarities to each FAQ
best_score = float(cosine_scores[0].max())  # highest similarity
best_match_index = int(cosine_scores[0].argmax())
best_match_text = knowledge_base[best_match_index]

print(f"User Query   : {user_query}")
print(f"Best Match   : {best_match_text}")
print(f"Best Score   : {best_score:.4f}")

# Threshold check
THRESHOLD = 0.5
if best_score >= THRESHOLD:
    print("Confidence is high. Serve answer from knowledge base.")
else:
    print("Confidence is low. Fallback to GPT-4 (or advanced AI).")