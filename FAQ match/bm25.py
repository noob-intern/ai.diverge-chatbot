from rank_bm25 import BM25Okapi

knowledge_base = [
    # Ordering and Payment
    "How do I place an order on your website?",
    "Which payment methods do you accept?",
    "Is there a minimum order value?",
    "Can I apply multiple promo codes at checkout?",
    "Can I change my payment method after placing an order?",
    "What currencies do you support?",
    "Do you accept international credit cards?",

    # Shipping and Delivery
    "What shipping options do you offer?",
    "Where can I track my order?",
    "Do you ship internationally?",
    "Is expedited shipping available?",
    "How long does standard shipping take?",
    "What happens if my package is lost in transit?",
    "Is there an additional fee for rural deliveries?",

    # Returns and Refunds
    "What is your return policy?",
    "Do I need to pay for return shipping?",
    "Can I return an item without the original packaging?",
    "How long does the refund process take?",
    "Can I exchange a product for a different size or color?",
    "Do you cover return shipping if the product is defective?",
    "How do I initiate a return?",

    # Account and Security
    "How do I reset my password?",
    "Can I update my email address?",
    "Is two-factor authentication available?",
    "How do I delete my account permanently?",
    "Is my personal information secure?",
    "Why was my account temporarily locked?",
    "Can I use social media accounts to sign up?",

    # General / Misc
    "Are gift cards available for purchase?",
    "Do you offer bulk or corporate discounts?",
    "Where can I find product reviews?",
]

# 2) Tokenize the knowledge base for BM25
tokenized_kb = [doc.lower().split() for doc in knowledge_base]
bm25 = BM25Okapi(tokenized_kb)

# 3) Prepare 25 user queries (15 high confidence, 10 low confidence)
# "High Confidence" queries are likely to have a near-direct match in the knowledge_base
# "Low Confidence" queries are more ambiguous, unusual, or tangential

queries = [
    # ---- High Confidence (15) ----
    ("HC", "Can I return a product I dislike?"),
    ("HC", "Where can I check my shipment status?"),
    ("HC", "Is there a minimum purchase amount required?"),
    ("HC", "How do I reset the password on my account?"),
    ("HC", "Do you ship outside the country?"),
    ("HC", "Which forms of payment are accepted?"),
    ("HC", "Is there an option for faster delivery?"),
    ("HC", "Do you have gift cards?"),
    ("HC", "Can I use two different promo codes on one order?"),
    ("HC", "How do I delete my account for good?"),
    ("HC", "Is there any fee for rural area shipments?"),
    ("HC", "Can I exchange an item for a larger size?"),
    ("HC", "How do I start a return process?"),
    ("HC", "What if my package never arrives?"),
    ("HC", "Can I switch my credit card after placing an order?"),

    # ---- Low Confidence (10) ----
    ("LC", "What is your corporate social responsibility approach?"),
    ("LC", "Could I have updates via carrier pigeon?"),
    ("LC", "Do you have any suggestions for gift wrapping techniques?"),
    ("LC", "How many employees does your company have?"),
    ("LC", "Can you send me an email about space travel discounts?"),
    ("LC", "What’s the meaning of life?"),
    ("LC", "Do I need special glasses to read your website?"),
    ("LC", "Is there a lounge area in your store for my dog?"),
    ("LC", "Can I trade in my old TV for store credit?"),
    ("LC", "When does your CEO usually wake up in the morning?"),
]

# 4) Process each query, compute BM25 scores, and find the best match
for i, (confidence_level, user_query) in enumerate(queries, start=1):
    tokenized_query = user_query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    best_idx = scores.argmax()
    best_score = scores[best_idx]
    best_match = knowledge_base[best_idx]

    print(f"Query #{i} ({confidence_level}): {user_query}")
    print(f"  Best Match  : {best_match}")
    print(f"  BM25 Score  : {best_score:.2f}\n")