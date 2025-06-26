import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
from transformers.pipelines import pipeline
import json
import re
from functools import lru_cache

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MODEL_CACHE_DIR'] = os.getenv('MODEL_CACHE_DIR', './model_cache')
app.config['MAX_RECOMMENDATIONS'] = int(os.getenv('MAX_RECOMMENDATIONS', 3))

# Cache models between requests
@lru_cache(maxsize=None)
def load_models():
    embedder = SentenceTransformer("intfloat/e5-large-v2", cache_folder=app.config['MODEL_CACHE_DIR'])
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    return embedder, generator

embedder, generator = load_models()

# Load course data
with open("courses.json", "r") as f:
    courses_list = json.load(f)

# Prepare embeddings
descriptions = [
    f"passage: {course['title']} - {course['domain']} - {course['description']} - Keywords: {', '.join(course.get('keywords', []))}"
    for course in courses_list
]
course_embeddings = embedder.encode(descriptions, convert_to_tensor=True)

def clean_message(message: str) -> str:
    message = re.sub(r'\s+', ' ', message)
    sentences = [s.strip() for s in message.split('.') if s.strip()]
    seen = set()
    unique_sentences = []
    for sentence in sentences:
        if sentence.lower() not in seen:
            seen.add(sentence.lower())
            unique_sentences.append(sentence)
    if unique_sentences:
        unique_sentences[0] = unique_sentences[0][0].upper() + unique_sentences[0][1:]
        return '. '.join(unique_sentences) + ('' if message.endswith('.') else '.')
    return "This course matches your interest."

@app.route("/recommend", methods=["POST"])
def recommend():
    raw_query = request.json.get("query", "").strip()
    if not raw_query:
        return jsonify({"error": "Please enter a learning interest."}), 400

    query_terms = re.findall(r'\b\w{3,}\b', raw_query.lower())
    query_embedding = embedder.encode(
        f"query: Find courses about {raw_query.lower()} covering {', '.join(query_terms)}",
        convert_to_tensor=True
    )

    similarities = util.cos_sim(query_embedding, course_embeddings)[0]
    candidates = [
        (idx, score.item()) for idx, score in enumerate(similarities)
        if score.item() > 0.5  # Minimum similarity threshold
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)

    recommendations = []
    for idx, score in candidates[:5]:  # Top 5 candidates
        course = courses_list[idx]
        prompt = f"""User wants to learn: {raw_query}.
        Course: {course['title']} ({course['domain']}, {course['level']})
        Description: {course['description']}
        Keywords: {', '.join(course.get('keywords', []))}
        Explain in one sentence why this course matches:"""
        
        try:
            message = generator(prompt, max_length=60)[0]['generated_text']
        except:
            message = f"Relevant to {raw_query} based on your interests."

        recommendations.append({
            **course,
            "similarity": round(score, 3),
            "message": clean_message(message)
        })
        if len(recommendations) >= app.config['MAX_RECOMMENDATIONS']:
            break

    return jsonify({
        "recommendations": recommendations or [],
        "query": raw_query
    })

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)