from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# FastAPI App
app = FastAPI()

# OpenAI API Key
openai.api_key = "your_openai_api_key_here"

# Database Connection
def get_faqs():
    conn = sqlite3.connect("faqs.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM faqs")
    faqs = cursor.fetchall()
    conn.close()
    return faqs

# Input Model
class Query(BaseModel):
    question: str
    past_conversation: str = ""

# Match Question to FAQ
def match_faq(question, faqs):
    questions = [faq[1] for faq in faqs]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions)
    question_vec = vectorizer.transform([question])
    similarity_scores = cosine_similarity(question_vec, tfidf_matrix).flatten()
    max_score_idx = similarity_scores.argmax()
    return faqs[max_score_idx] if similarity_scores[max_score_idx] > 0.7 else None

# OpenAI Fallback
def openai_response(question, past_conversation):
    prompt = f"User question: {question}\nConversation so far: {past_conversation}\nAnswer the question professionally."
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# API Endpoint
@app.post("/ask")
async def ask(query: Query):
    faqs = get_faqs()
    matched_faq = match_faq(query.question, faqs)

    if matched_faq:
        return {"answer": matched_faq[2]}  # Return FAQ answer

    # Fallback to OpenAI
    try:
        ai_answer = openai_response(query.question, query.past_conversation)
        return {"answer": ai_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching OpenAI response: {str(e)}")
    

# Run the FastAPI app
# uvicorn main:app --reload