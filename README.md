# YouTube Reviser — RAG-Based Video Revision Assistant

## 1. Project Overview

**YouTube Reviser** is a Retrieval-Augmented Generation (RAG) application that turns a collection of YouTube educational videos into a searchable revision assistant.

The system processes YouTube videos, extracts their **Hindi transcripts**, translates the transcript chunks from **Hindi to English**, creates vector embeddings, and stores those embeddings in **Qdrant** along with useful metadata such as:

- Video title
- YouTube video ID
- YouTube URL
- Transcript text
- Original Hindi transcript
- Chunk ID
- Start timestamp
- End timestamp
- Direct timestamp URL

When a user asks a question, the system converts the question into an embedding, searches Qdrant for semantically relevant transcript chunks, and sends the retrieved context to an LLM. The final answer can therefore include the **answer, video name, relevant timestamp, and direct YouTube link**.

---


# 3. Problem Statement

The original learning resource was a YouTube playlist containing approximately **127 DSA videos**. The videos were primarily in **Hindi**, which made English-based revision more difficult. More importantly, with such a large playlist, it was difficult to remember **which video contained a particular DSA concept or question**.

For example, while revising topics such as:

- Two Pointers
- Sliding Window
- Binary Search
- Trees
- Graphs
- Dynamic Programming

it was difficult to remember which of the 127 videos explained the required concept. Finding the exact explanation manually meant opening multiple videos, searching through transcripts or watching portions of the videos, and locating the relevant timestamp.

The project was created to solve this problem by turning the videos into a **semantic, searchable revision knowledge base**.

## Why the Dataset Was Reduced to 25 Videos

The initial idea was to process all **127 videos**. However, processing the complete playlist was not practical within the available development constraints.

Processing transcripts for all 127 videos would take approximately **3 hours on the available TPU setup**, in addition to the storage and processing requirements for the generated transcripts, translated chunks, embeddings, and Qdrant data.

Therefore, the dataset was intentionally reduced to:

```text
127 available DSA videos
          ↓
25 selected videos
```

This was a practical engineering decision based on:

- **Processing-time constraints**
- **Storage constraints**
- **Development/resource constraints**
- The need to build and validate the complete RAG pipeline within the available time

The goal was not to claim that 25 videos represent the entire playlist. Instead, the 25-video dataset was used as a manageable working dataset to implement and validate the complete system.

The same pipeline can later be scaled from:

```text
25 videos → 127 videos → larger playlists
```

without changing the fundamental RAG architecture.

## The Problem Being Solved

Before this system:

```text
127 DSA Videos
      ↓
Remember which video?
      ↓
Open multiple videos
      ↓
Search/watch manually
      ↓
Find the concept
      ↓
Find the timestamp
```

After building YouTube Reviser:

```text
User Question
      ↓
Semantic Search
      ↓
Relevant Transcript Chunk
      ↓
Video Name
      ↓
Exact Timestamp
      ↓
YouTube Link
```

The core problem can therefore be stated as:

> **How can a learner efficiently search and revise concepts from a large Hindi DSA YouTube playlist without remembering which video contains each concept?**

YouTube Reviser addresses this by converting the selected videos into a **timestamp-aware RAG knowledge base**.

---

# 4. Project Goal

The main goal is:

> **Instead of watching an entire playlist again, allow a user to ask a question and immediately find the relevant explanation from the videos.**

For example:

**User:**

> What is the two-pointer technique?

The system should retrieve the relevant transcript and return something similar to:

> The two-pointer technique uses two indices to traverse a data structure efficiently...

**Source:**

- Video: Episode 4 — Master DSA Patterns...
- Timestamp: 06:29
- YouTube: Direct link to that point in the video

This makes the project a **personal YouTube revision assistant** rather than simply a chatbot.

---

# 5. Source Playlist

The project uses the following YouTube playlist:

**DSA Patterns 2025**

Playlist ID:

```text
PLbJhGqY-mq47k_WLUtzVjmarUm1EuXPj2
```

The playlist contained approximately **127 videos** when it was collected.

For this project, the first:

```text
25 videos
```

were selected for the initial RAG dataset.

---

# 6. High-Level Architecture

```text
                    YOUTUBE PLAYLIST
                           |
                           v
                    yt-dlp / Video
                    Collection
                           |
                           v
                      25 Videos
                           |
                           v
              YouTube Transcript API
                           |
                           v
                  Hindi Transcripts
                           |
                           v
                 Hindi → English
                      Argos
                           |
                           v
                  English Chunks
                  + Timestamps
                           |
                           v
              Sentence Transformer
                           |
                           v
                  384-D Embeddings
                           |
                           v
                       QDRANT
                +------------------+
                | Vector           |
                | Payload          |
                +------------------+
                           |
                           |
                    USER QUESTION
                           |
                           v
                Query Embedding
                           |
                           v
                 Qdrant Similarity
                     Search
                           |
                           v
                 Relevant Chunks
                           |
                           v
                  Context Building
                           |
                           v
                       GROQ LLM
                           |
                           v
                    Final Answer
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
          Answer       Video Name     Timestamp
                                           |
                                           v
                                     YouTube Link
```

---

# 7. Complete Data Flow

## Stage 1 — Video Collection

The first stage is collecting the videos from the YouTube playlist.

### Technology

```text
yt-dlp
```

The playlist metadata provides:

- Video ID
- Video title
- Video URL

Example:

```json
{
  "video_id": "PvyEr3CekZE",
  "title": "Episode - 4 | Master DSA Patterns...",
  "url": "https://www.youtube.com/watch?v=PvyEr3CekZE"
}
```

The playlist contained approximately 127 videos, but only 25 were selected for the project.

---

# 8. Stage 2 — Transcript Extraction

For each selected video, the project obtains the **Hindi transcript**.

### Technology

```text
YouTube Transcript API
```

The project specifically targets:

```text
Hindi
```

There is no requirement for an existing English transcript.

The pipeline is:

```text
YouTube Video
      |
      v
Hindi Transcript
```

Each transcript segment contains timing information.

Example:

```json
{
  "text": "मुख्यमंत्री हैं...",
  "start": 389.919,
  "end": 428.0
}
```

The timestamps are important because they allow the system to later point the user to the exact section of the YouTube video.

---

# 9. Stage 3 — Hindi → English Translation

The extracted Hindi transcript is translated into English.

### Technology

```text
Argos Translate
```

### Translation direction

```text
Hindi → English
```

Argos Translate is an open-source/offline translation library used in this project.

The purpose of translating the transcript is to create a consistent English text representation for the downstream embedding and retrieval pipeline.

Example:

```text
Hindi:

"मुख्यमंत्री हैं चलिए हम तो छोटा आदमी हैं..."

             ↓

Argos Translate

             ↓

English:

"Chief Minister. Let us have a little man..."
```

The original Hindi text is still preserved in the payload.

---

# 10. Stage 4 — Chunking

A complete transcript is too large to treat as one retrieval unit.

Therefore, the transcript is divided into smaller chunks.

The project uses approximately:

```text
500 characters per chunk
```

Each chunk keeps its original timing information.

Example:

```json
{
  "chunk_id": 14,
  "start_time": 389.919,
  "end_time": 428.0,
  "hindi_text": "...",
  "english_text": "..."
}
```

This is important because the retrieved chunk can be mapped back to the exact location in the video.

---

# 11. Stage 5 — Timestamp URL Generation

For every chunk, a direct YouTube timestamp URL is created.

Example:

```text
https://www.youtube.com/watch?v=PvyEr3CekZE&t=389s
```

Therefore:

```text
Transcript Chunk
      |
      +---- Video ID
      |
      +---- Start Time
      |
      +---- End Time
      |
      +---- Timestamp URL
```

This is one of the main features of the project.

The system does not just tell the user **which video** contains the answer.

It can tell the user **where in the video** the answer was discussed.

---

# 12. Stage 6 — Embeddings

After translation and chunking, the English transcript chunks are converted into vector embeddings.

### Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model produces:

```text
384-dimensional vectors
```

Example:

```text
English transcript chunk
          |
          v
Sentence Transformer
          |
          v
[0.021, -0.183, 0.442, ...]
          |
          v
384-dimensional vector
```

The embedding represents the semantic meaning of the transcript chunk.

This allows semantic search rather than only keyword matching.

For example:

```text
Query:
"How does the two pointer approach work?"

```

can retrieve a chunk containing:

```text
"Use two indices, one from the beginning and another from the end..."
```

even if the exact words in the query do not appear in the transcript.

---

# 13. Stage 7 — Vector Database

### Vector database

```text
Qdrant
```

### Collection

```text
youtube_reviser_25_videos_2026_08_20
```

### Configuration

```text
Vector size: 384
Distance: COSINE
```

The current collection contains approximately:

```text
25 videos
1471 transcript chunks / points
```

The exact number of points is based on the chunking of the 25 processed videos.

---

# 14. What Is Stored in Qdrant?

Each Qdrant point contains two major components:

```text
POINT
│
├── VECTOR
│
└── PAYLOAD
```

## Vector

The vector contains the semantic representation of the English transcript chunk.

```text
384 floating-point values
```

## Payload

The payload stores the information needed to understand where the retrieved chunk came from.

Current payload structure:

```json
{
  "text": "English transcript chunk",

  "hindi_text": "Original Hindi transcript chunk",

  "video_id": "PvyEr3CekZE",

  "title": "Episode - 4 | Master DSA Patterns...",

  "url": "https://www.youtube.com/watch?v=PvyEr3CekZE",

  "chunk_id": 14,

  "start_time": 389.919,

  "end_time": 428.0,

  "timestamp_url": "https://www.youtube.com/watch?v=PvyEr3CekZE&t=389s"
}
```

Therefore the vector answers:

> **Which chunk is semantically relevant?**

The payload answers:

> **Where did this chunk come from?**

---

# 15. Stage 8 — User Query

When the user asks a question:

```text
What is the two pointer technique?
```

the question is converted into an embedding using the same embedding model:

```text
User Question
      |
      v
all-MiniLM-L6-v2
      |
      v
384-dimensional query vector
```

---

# 16. Stage 9 — Similarity Search

The query vector is sent to Qdrant.

Qdrant compares the query vector with the stored transcript vectors using:

```text
Cosine Similarity
```

The most semantically relevant chunks are retrieved.

Example:

```text
Query
  |
  v
Qdrant
  |
  +--> Chunk 1 — Score 0.84
  |
  +--> Chunk 2 — Score 0.81
  |
  +--> Chunk 3 — Score 0.79
  |
  +--> Chunk 4 — Score 0.77
  |
  +--> Chunk 5 — Score 0.75
```

The top-K chunks are then used as the context for the LLM.

---

# 17. Stage 10 — Context Construction

The retrieved Qdrant payloads are combined into a context.

Example:

```text
SOURCE 1

Video:
Episode - 4 | Master DSA Patterns...

Timestamp:
https://www.youtube.com/watch?v=PvyEr3CekZE&t=389s

Transcript:
"Two pointers is a technique..."


SOURCE 2

Video:
Episode - 4 | Master DSA Patterns...

Timestamp:
https://www.youtube.com/watch?v=PvyEr3CekZE&t=450s

Transcript:
"...move the left and right pointers..."
```

This context is passed to the LLM.

---

# 18. Stage 11 — LLM

### LLM

The current RAG code uses:

```text
Groq
```

with:

```text
openai/gpt-oss-120b
```

The LLM receives:

```text
User Question
+
Retrieved Context
```

and generates the final response.

The prompt instructs the model to:

1. Use only retrieved information.
2. Avoid inventing information.
3. Say it does not know when the answer is not present.
4. Mention the relevant video and timestamp.

---

# 19. Final RAG Workflow

The complete system can therefore be summarized as:

```text
                    INGESTION

YouTube Playlist
       |
       v
yt-dlp
       |
       v
25 Videos
       |
       v
YouTube Transcript API
       |
       v
Hindi Transcript
       |
       v
Argos Translate
       |
       v
English Transcript
       |
       v
Chunking + Timestamp
       |
       v
all-MiniLM-L6-v2
       |
       v
384-D Embeddings
       |
       v
Qdrant
       |
       +-------------------+
       |                   |
     Vector             Payload
       |                   |
       |             video_id
       |             title
       |             URL
       |             timestamp
       |             transcript
       +-------------------+


                    QUERY

User Question
       |
       v
all-MiniLM-L6-v2
       |
       v
Query Vector
       |
       v
Qdrant Similarity Search
       |
       v
Top-K Relevant Chunks
       |
       v
Payload + Transcript
       |
       v
Context
       |
       v
Groq / GPT-OSS-120B
       |
       v
Final Answer
       |
       +------ Video Name
       |
       +------ Timestamp
       |
       +------ YouTube Link
```

---

# 20. Technologies Used

| Component | Technology |
|---|---|
| Video collection | yt-dlp |
| Transcript extraction | YouTube Transcript API |
| Source transcript language | Hindi |
| Translation | Argos Translate |
| Translation | Hindi → English |
| Chunking | Custom Python chunking |
| Embedding model | all-MiniLM-L6-v2 |
| Embedding dimension | 384 |
| Vector database | Qdrant |
| Distance metric | Cosine |
| LLM | Groq |
| LLM model | openai/gpt-oss-120b |
| Configuration | python-dotenv |
| Language | Python |

---

# 21. Dataset Statistics

Current project configuration:

```text
Playlist videos available: ~127
Videos selected:            25
Languages:                  Hindi → English
Embedding model:            all-MiniLM-L6-v2
Embedding dimension:        384
Vector database:             Qdrant
Distance:                    Cosine
Qdrant points/chunks:        1471
```

The 1471 Qdrant points represent **transcript chunks**, not 1471 videos.

Approximately:

```text
25 videos
   ↓
multiple transcript chunks/video
   ↓
1471 chunks
   ↓
1471 embeddings
   ↓
1471 Qdrant points
```

---

# 22. Why RAG?

A normal LLM does not automatically know the exact contents of your private video dataset.

RAG solves this by retrieving relevant information from your own knowledge base before generating the answer.

Without RAG:

```text
Question
   ↓
LLM
   ↓
General answer
```

With RAG:

```text
Question
   ↓
Embedding
   ↓
Qdrant
   ↓
Relevant transcript
   ↓
LLM
   ↓
Answer grounded in videos
```

This reduces the chance of the model answering from unrelated general knowledge.

---

# 23. Why Qdrant?

Qdrant is used because the project needs semantic vector search.

The database stores:

```text
Embedding
+
Metadata
```

This is particularly useful for this project because retrieving a transcript chunk is not enough.

We also need:

```text
Video title
Video URL
Timestamp
Video ID
Original Hindi text
```

Qdrant payload allows this metadata to remain attached to the vector.

---

# 24. Why Keep the Timestamp?

The timestamp is a key feature of the project.

Suppose the answer is found in:

```text
start_time = 389.919 seconds
```

The system generates:

```text
https://www.youtube.com/watch?v=PvyEr3CekZE&t=389s
```

Therefore the user can directly jump to the relevant explanation instead of searching through the entire video.

---

# 25. Example User Experience

### User

```text
Explain two pointer technique.
```

### Retrieval

```text
Qdrant
   ↓
Relevant transcript chunks
```

### Retrieved metadata

```text
Video:
Episode - 4 | Master DSA Patterns...

Start:
389.919 seconds

Timestamp:
06:29

URL:
YouTube timestamp link
```

### LLM

The LLM uses the retrieved transcript to formulate the answer.

### Final response

```text
The two-pointer technique uses two indices
to efficiently process elements from different
positions in a data structure.

Source:
Episode - 4 | Master DSA Patterns...

Timestamp:
06:29

Watch from this point:
YouTube timestamp link
```

---

# 26. Project's Main Selling Point

The project is not simply:

> "Chat with a YouTube video."

The stronger idea is:

> **A RAG-powered YouTube revision assistant that searches across educational videos and takes the user directly to the timestamp where the answer was explained.**

The system combines:

```text
Semantic Search
+
Transcript Retrieval
+
Translation
+
LLM
+
Timestamp Retrieval
```

to create a revision-oriented search experience.

---

# 27. Current Project Status

### Completed

- [x] YouTube playlist collection
- [x] Selected 25 videos
- [x] Hindi transcript extraction
- [x] Hindi → English translation
- [x] Transcript chunking
- [x] Timestamp preservation
- [x] English embeddings
- [x] 384-dimensional vectors
- [x] Qdrant collection
- [x] Payload storage
- [x] Similarity search
- [x] Groq LLM integration
- [x] RAG context generation
- [x] Video metadata retrieval
- [x] Timestamp URL retrieval

### Current Qdrant dataset

```text
Collection:
youtube_reviser_25_videos_2026_08_20

Videos:
25

Points:
1471

Vector dimension:
384

Distance:
Cosine
```

---

# 28. Future Improvements

Possible improvements include:

### 1. Better retrieval

Use:

```text
Hybrid Search
```

combining semantic vector search with keyword search.

### 2. Reranking

Retrieve more candidates and use a reranker to select the most relevant chunks.

### 3. Video-level grouping

If multiple chunks from the same video are retrieved, group them together so the final answer presents cleaner sources.

### 4. Multi-video answers

The system can combine information from multiple videos when the answer is spread across the playlist.

### 5. Quiz mode

Generate questions from retrieved transcript chunks.

### 6. Flashcards

Generate:

```text
Question → Answer
```

pairs from the videos.

### 7. Topic detection

Automatically identify topics such as:

```text
Two Pointers
Sliding Window
Binary Search
Dynamic Programming
Graphs
Trees
```

### 8. User progress tracking

Track which videos/topics the user has revised.

### 9. Weak-topic detection

Identify topics where the user repeatedly asks questions or performs poorly in quizzes.

---

# 29. Final Architecture

```text
                         YOUTUBE
                            |
                            v
                       25 VIDEOS
                            |
                            v
                  HINDI TRANSCRIPTS
                            |
                            v
                 ARGOS TRANSLATION
                    Hindi → English
                            |
                            v
                  CHUNK + TIMESTAMP
                            |
                            v
              all-MiniLM-L6-v2
                       384-D
                            |
                            v
                         QDRANT
                  /                  \
              VECTOR                PAYLOAD
                |                      |
             Search             Video Metadata
                |                      |
                +----------+-----------+
                           |
                           v
                       RETRIEVAL
                           |
                           v
                    RELEVANT CONTEXT
                           |
                           v
                    GROQ / GPT-OSS
                           |
                           v
                     FINAL ANSWER
                           |
             +-------------+-------------+
             |             |             |
          Answer       Video Name    Timestamp
                                         |
                                         v
                                  YouTube Link
```

---

## 30. One-Line Project Description

> **YouTube Reviser is a RAG-based revision assistant that processes 25 Hindi educational YouTube videos, translates and embeds their transcripts, stores them in Qdrant with timestamp metadata, and uses semantic retrieval + an LLM to answer questions with the exact video and timestamp where the concept was explained.**
