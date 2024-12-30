from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

knowledge_base = [
    "How do I place an order on your website?",
    "Which payment methods do you accept?",
    "Is there a minimum order value?",
    "Where can I track my order?",
    "What is your return policy?",
    "Do I need to pay for return shipping?",
    "What shipping options do you offer?",
    "How can I file a warranty claim?",
    "Which helmet size is best for me?",
    "How do I clean my helmet?",
    "When should I replace my helmet?",
    "Do you ship internationally?",
    "Can I purchase gift cards?",
    "How do I redeem a promotional code?",
    "Which lock is best for securing my bike in the city?",
    "Do you offer loyalty rewards or points for purchases?",
    "Where can I find instructions for assembling a bike rack?",
    "How do I choose the right bike lights for nighttime riding?",
    "Do you have any recommendations for child seats or trailers?",
    "Can I pick up my order in-store instead of shipping?"
]

# Encode knowledge base once (can be very large ig)
kb_embeddings = model.encode(knowledge_base, convert_to_tensor=True)


THRESHOLD = 0.5


queries = [
    # --- High Confidence (15) ---
    "What is the process for ordering on your site?",
    "I'd like to know your refund policy.",
    "Are you able to ship products across borders?",
    "Which lock will keep my bike safest?",
    "How can I file a warranty claim?",
    "Is there a way to get my money back if I'm not satisfied?",
    "What's your procedure for shipping outside the country?",
    "Where do I go to make a warranty request?",
    "How do I purchase something from your store?",
    "I'm unhappy with the product. How do I get a refund?",
    "Can I get a new item if mine is broken?",
    "What's the best way to secure my bicycle?",
    "Where can I find the guidelines for returning items?",
    "If I don't live in the US, can I still get my items delivered?",
    "In case of a faulty product, how to replace it?",

    # --- Low Confidence (10) ---
    "What's your favorite color?",
    "Where is my cat?",
    "How do I fix a broken phone screen?",
    "What time is it in Tokyo?",
    "What's your company stock price?",
    "Can you show me a picture of a cat?",
    "Is there a sale on winter coats?",
    "How do I create an AI model from scratch?",
    "Where can I find good pizza near me?",
    "Any suggestions for a birthday gift?"
]



for user_query in queries:
    # Encode user query
    query_embedding = model.encode(user_query, convert_to_tensor=True)

    # Calculate cosine similarities
    cosine_scores = util.cos_sim(query_embedding, kb_embeddings)
    best_score = float(cosine_scores[0].max())
    best_match_index = int(cosine_scores[0].argmax())
    best_match_text = knowledge_base[best_match_index]

    print("-------------------------------------------------")
    print(f"User Query   : {user_query}")
    print(f"Best Match   : {best_match_text}")
    print(f"Best Score   : {best_score:.4f}")

    if best_score >= THRESHOLD:
        print("→ Confidence is HIGH. Serve answer from knowledge base.\n")
    else:
        print("→ Confidence is LOW. Fallback to GPT-4 (or advanced AI).\n")

