import os
import json
import re
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(directory=".", path="index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(".", path)

@app.route("/recommend", methods=["POST"])
def recommend():
    raw_query = request.json.get("query", "").strip()
    if not raw_query:
        return jsonify({"error": "Please enter a learning interest."}), 400

    # Load data and models on-demand to stay within memory limits
    with open("courses.json", "r") as f:
        courses_list = json.load(f)

    embedder = SentenceTransformer("intfloat/e5-small-v2")  # lighter model
    generator = pipeline("text2text-generation", model="google/flan-t5-small")  # lighter model

    descriptions = [
        f"passage: {course['title']} - {course['domain']} - {course['description']} - Keywords: {', '.join(course.get('keywords', []))}"
        for course in courses_list
    ]
    course_embeddings = embedder.encode(descriptions, convert_to_tensor=True)

    query_terms = re.findall(r'\b\w{3,}\b', raw_query.lower())
    query_embedding = embedder.encode(
        f"query: Find courses about {raw_query.lower()} covering {', '.join(query_terms)}",
        convert_to_tensor=True
    )

    similarities = util.cos_sim(query_embedding, course_embeddings)[0]
    candidates = [
        (idx, score.item()) for idx, score in enumerate(similarities)
        if score.item() > 0.5
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)

    recommendations = []
    for idx, score in candidates[:3]:
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
            "message": message
        })

    return jsonify({
        "recommendations": recommendations or [],
        "query": raw_query
    })
