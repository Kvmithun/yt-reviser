# ============================================================
# YOUTUBE REVISER - QDRANT + GROQ RAG
# ============================================================

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq

# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY or not GROQ_API_KEY:
    raise ValueError("Missing QDRANT_URL, QDRANT_API_KEY or GROQ_API_KEY")

# ============================================================
# 2. COLLECTION
# ============================================================

COLLECTION_NAME = "youtube_reviser_25_videos_2026_08_20"

# ============================================================
# 3. CONNECT TO QDRANT
# ============================================================

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
print("Connected to Qdrant Cloud")

# ============================================================
# 4. LOAD EMBEDDING MODEL
# ============================================================

model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded")

# ============================================================
# 5. CONNECT TO GROQ
# ============================================================

groq = Groq(api_key=GROQ_API_KEY)
print("Connected to Groq")

# ============================================================
# 6. FORMAT TIMESTAMP
# ============================================================

def format_timestamp(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}" if seconds >= 3600 else f"{seconds // 60:02d}:{seconds % 60:02d}"

# ============================================================
# 7. SEARCH QDRANT
# ============================================================

def search_qdrant(question, top_k=5):
    query_vector = model.encode(question).tolist()
    return qdrant.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=top_k, with_payload=True).points

# ============================================================
# 8. BUILD CONTEXT
# ============================================================

def build_context(results):
    context = ""
    for i, result in enumerate(results):
        p = result.payload
        context += f"""
SOURCE {i + 1}
Video: {p.get('title', 'Unknown')}
Video URL: {p.get('url', '')}
Timestamp: {format_timestamp(p.get('start_time', 0))} - {format_timestamp(p.get('end_time', 0))}
Timestamp URL: {p.get('timestamp_url', p.get('url', ''))}
Transcript: {p.get('text', p.get('english_text', ''))}
"""
    return context

# ============================================================
# 9. ASK GROQ
# ============================================================

def ask_groq(question, context):
    prompt = f"""
You are YouTube Reviser.

Answer the user's question using ONLY the retrieved
YouTube transcript context.

Do not invent information.

If the answer is not present, say:
"I couldn't find enough information in the videos."

Answer clearly in English.

Question:
{question}

Retrieved Context:
{context}
"""
    response = groq.chat.completions.create(model="openai/gpt-oss-120b", messages=[{"role": "user", "content": prompt}], temperature=0.2, max_completion_tokens=1000)
    return response.choices[0].message.content

# ============================================================
# 10. COMPLETE RAG PIPELINE
# ============================================================

def youtube_reviser(question, top_k=5):
    results = search_qdrant(question, top_k)

    if not results:
        print("No relevant information found.")
        return

    context = build_context(results)
    answer = ask_groq(question, context)

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)

    print("\n" + "=" * 70)
    print("VIDEO SOURCES")
    print("=" * 70)

    seen = set()

    for result in results:
        p = result.payload
        video_id = p.get("video_id")

        if video_id in seen:
            continue

        seen.add(video_id)

        print(f"\nVideo       : {p.get('title', 'Unknown')}")
        print(f"Timestamp   : {format_timestamp(p.get('start_time', 0))}")
        print(f"Video URL   : {p.get('url', '')}")
        print(f"Watch Here  : {p.get('timestamp_url', p.get('url', ''))}")
        print(f"Similarity  : {result.score:.4f}")

# ============================================================
# 11. ASK QUESTION
# ============================================================

question = """Where is the Dutch National Flag algorithm explained?
   Give me the video name and exact timestamp"""

youtube_reviser(question, top_k=5)