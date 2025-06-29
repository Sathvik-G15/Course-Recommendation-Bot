import os
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
from functools import lru_cache

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MODEL_CACHE_DIR'] = os.getenv('MODEL_CACHE_DIR', './model_cache')
app.config['MAX_RECOMMENDATIONS'] = int(os.getenv('MAX_RECOMMENDATIONS', 3))

# Load embedding model
@lru_cache(maxsize=None)
def load_model():
    return SentenceTransformer("sentence-transformers/nli-distilroberta-base-v2", cache_folder=app.config['MODEL_CACHE_DIR'])

embedder = load_model()

# Load courses
with open("courses.json", "r") as f:
    courses_list = json.load(f)

# Prepare embeddings
descriptions = [
    f"{course['title']} - {course['domain']} - {course['description']} - Keywords: {', '.join(course.get('keywords', []))}"
    for course in courses_list
]
course_embeddings = embedder.encode(descriptions, convert_to_tensor=True)

# Abbreviation expansion
def expand_abbreviations(text):
    abbreviations = {
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "cv": "computer vision",
        "dl": "deep learning",
        "nlp": "natural language processing",
        "cnn": "convolutional neural networks",
        "rnn": "recurrent neural networks",
        "qa": "question answering",
        "eda": "exploratory data analysis",
        "sql": "structured query language",
        "js": "javascript",
        "db": "database",
        "api": "application programming interface",
        "devops": "development operations",
        "gcp": "google cloud platform",
        "aws": "amazon web services",
        "html": "hypertext markup language",
        "css": "cascading style sheets",
        "oop": "object oriented programming",
        "stl": "standard template library",
        "dsa": "data structures and algorithms"
    }
    words = text.lower().split()
    expanded = [abbreviations.get(word, word) for word in words]
    return " ".join(expanded)

# Simple reason generator
def generate_reason(query, course):
    return f"This course matches your interest in '{query}' based on its coverage of topics like {', '.join(course.get('keywords', [])[:3])}."

@app.route("/recommend", methods=["POST"])
def recommend():
    raw_query = request.json.get("query", "").strip()
    if not raw_query:
        return jsonify({"error": "Please enter a learning interest."}), 400

    expanded_query = expand_abbreviations(raw_query)
    query_terms = re.findall(r'\b\w{3,}\b', expanded_query)

    query_embedding = embedder.encode(
        f"{expanded_query} {', '.join(query_terms)}",
        convert_to_tensor=True
    )

    similarities = util.cos_sim(query_embedding, course_embeddings)[0]
    candidates = [
        (idx, score.item()) for idx, score in enumerate(similarities)
        if score.item() > 0.4
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)

    recommendations = []
    for idx, score in candidates[:5]:
        course = courses_list[idx]
        message = generate_reason(raw_query, course)
        recommendations.append({
            **course,
            "similarity": round(score, 3),
            "message": message
        })
        if len(recommendations) >= app.config['MAX_RECOMMENDATIONS']:
            break

    return jsonify({
        "recommendations": recommendations or [],
        "query": raw_query
    })


