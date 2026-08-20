import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

import yt_dlp
import argostranslate.package
import argostranslate.translate
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


PLAYLIST_URL = (
    "https://youtube.com/playlist?list=PLbJhGqY-mq47k_WLUtzVjmarUm1EuXPj2"
)
NUM_VIDEOS = 25
MAX_CHARS = 500
TEMP_DIR = Path("youtube_subtitles")
TEMP_DIR.mkdir(exist_ok=True)

COMPLETE_TRANSCRIPT_FILE = Path("youtube_complete_transcripts.json")
RAG_JSON_FILE = Path("youtube_rag_chunks.json")
FAILED_FILE = Path("failed_videos.json")

COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "youtube_reviser")
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
RECREATE_COLLECTION = os.environ.get("QDRANT_RECREATE", "").lower() in {
    "1",
    "true",
    "yes",
}


def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def timestamp_url(video_url: str, seconds: float) -> str:
    return f"{video_url}&t={int(seconds)}s"


def extract_playlist_videos() -> list[dict]:
    print_header("STEP 1 - EXTRACTING PLAYLIST")

    playlist_opts = {
        "extract_flat": True,
        "skip_download": True,
        "quiet": False,
    }

    with yt_dlp.YoutubeDL(playlist_opts) as ydl:
        playlist_info = ydl.extract_info(PLAYLIST_URL, download=False)

    videos = []
    for entry in playlist_info.get("entries", []):
        if not entry:
            continue

        video_id = entry.get("id")
        if not video_id:
            continue

        videos.append(
            {
                "video_id": video_id,
                "title": entry.get("title", "Unknown"),
                "url": f"https://www.youtube.com/watch?v={video_id}",
            }
        )

    print("Total videos found:", len(videos))
    if len(videos) < NUM_VIDEOS:
        raise RuntimeError(
            f"Playlist contains only {len(videos)} videos. Need at least {NUM_VIDEOS}."
        )

    selected = videos[:NUM_VIDEOS]
    print("Videos selected:", len(selected))
    return selected


def setup_argos() -> None:
    print_header("STEP 2 - HINDI TO ENGLISH MODEL")

    installed_languages = argostranslate.translate.get_installed_languages()
    hindi = None
    english = None

    for language in installed_languages:
        if language.code == "hi":
            hindi = language
        elif language.code == "en":
            english = language

    if hindi and english:
        try:
            translation = hindi.get_translation(english)
            if translation:
                print("OK: Hindi to English model already installed")
                return
        except Exception:
            pass

    print("Downloading Hindi to English model...")
    argostranslate.package.update_package_index()
    packages = argostranslate.package.get_available_packages()
    package = next(
        (pkg for pkg in packages if pkg.from_code == "hi" and pkg.to_code == "en"),
        None,
    )
    if package is None:
        raise RuntimeError("Hindi to English Argos package not found")

    download_path = package.download()
    argostranslate.package.install_from_path(download_path)
    print("OK: Hindi to English model installed")


def translate_hindi_to_english(text: str) -> str | None:
    try:
        result = argostranslate.translate.translate(text, "hi", "en")
        return result.strip()
    except Exception as exc:
        print("Translation error:", exc)
        return None


def cleanup_old_subtitles(video_id: str) -> None:
    for path in TEMP_DIR.iterdir():
        if path.name.startswith(f"{video_id}."):
            try:
                path.unlink()
            except OSError:
                pass


def download_hindi_transcript(video: dict) -> Path | None:
    video_id = video["video_id"]
    cleanup_old_subtitles(video_id)

    opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["hi"],
        "subtitlesformat": "json3",
        "outtmpl": str(TEMP_DIR / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": False,
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(video["url"], download=True)

        for path in TEMP_DIR.iterdir():
            if path.name.startswith(f"{video_id}.") and path.suffix == ".json3":
                return path
    except Exception as exc:
        print("YouTube error:", exc)

    return None


def parse_json3(subtitle_file: Path) -> list[dict]:
    with subtitle_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    segments = []
    for event in data.get("events", []):
        start_ms = event.get("tStartMs")
        duration_ms = event.get("dDurationMs", 0)
        if start_ms is None:
            continue

        text_parts = []
        for seg in event.get("segs", []):
            text = seg.get("utf8", "")
            if text:
                text_parts.append(text)

        text = "".join(text_parts)
        text = re.sub(r"<[^>]+>", "", text).replace("\n", " ").strip()
        if not text:
            continue

        start = float(start_ms) / 1000
        end = start + (float(duration_ms) / 1000)
        segments.append({"text": text, "start": start, "end": end})

    return segments


def create_chunks(segments: list[dict], max_chars: int = 500) -> list[dict]:
    chunks = []
    current_text = []
    current_start = None
    current_end = None

    for segment in segments:
        text = segment["text"]
        start = segment["start"]
        end = segment["end"]

        if current_start is None:
            current_start = start

        candidate = (" ".join(current_text) + " " + text).strip()
        if current_text and len(candidate) > max_chars:
            chunks.append(
                {
                    "text": " ".join(current_text),
                    "start": current_start,
                    "end": current_end,
                }
            )
            current_text = [text]
            current_start = start
            current_end = end
        else:
            current_text.append(text)
            current_end = end

    if current_text:
        chunks.append(
            {
                "text": " ".join(current_text),
                "start": current_start,
                "end": current_end,
            }
        )

    return chunks


def process_videos(videos: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    print_header("STEP 3 - GETTING 25 HINDI TRANSCRIPTS")

    complete_transcripts = []
    rag_documents = []
    failed_videos = []

    for index, video in enumerate(videos, start=1):
        print()
        print("-" * 70)
        print(f"VIDEO {index}/{NUM_VIDEOS}")
        print(video["title"])

        try:
            subtitle_file = download_hindi_transcript(video)
            if subtitle_file is None:
                raise RuntimeError("Hindi transcript could not be retrieved")
            print("OK: Hindi transcript downloaded")

            segments = parse_json3(subtitle_file)
            if not segments:
                raise RuntimeError("Transcript is empty")
            print("Transcript segments:", len(segments))

            full_hindi_transcript = " ".join(segment["text"] for segment in segments)
            chunks = create_chunks(segments, MAX_CHARS)
            print("Chunks:", len(chunks))

            translated_chunks = []
            for chunk_id, chunk in enumerate(chunks):
                english_text = translate_hindi_to_english(chunk["text"])
                if not english_text:
                    raise RuntimeError(f"Translation failed at chunk {chunk_id}")

                translated_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "start_time": chunk["start"],
                        "end_time": chunk["end"],
                        "timestamp_url": timestamp_url(video["url"], chunk["start"]),
                        "hindi_text": chunk["text"],
                        "english_text": english_text,
                    }
                )

            complete_transcripts.append(
                {
                    "video_id": video["video_id"],
                    "title": video["title"],
                    "url": video["url"],
                    "language": "hi",
                    "translated_language": "en",
                    "full_hindi_transcript": full_hindi_transcript,
                    "chunks": translated_chunks,
                }
            )

            for chunk in translated_chunks:
                rag_documents.append(
                    {
                        "video_id": video["video_id"],
                        "title": video["title"],
                        "url": video["url"],
                        "chunk_id": chunk["chunk_id"],
                        "start_time": chunk["start_time"],
                        "end_time": chunk["end_time"],
                        "timestamp_url": chunk["timestamp_url"],
                        "hindi_text": chunk["hindi_text"],
                        "english_text": chunk["english_text"],
                    }
                )

            try:
                subtitle_file.unlink()
            except OSError:
                pass

            print(f"OK: VIDEO {index} COMPLETE")
            time.sleep(1)
        except Exception as exc:
            print(f"FAILED VIDEO {index}: {exc}")
            failed_videos.append(
                {
                    "video_id": video["video_id"],
                    "title": video["title"],
                    "url": video["url"],
                    "error": str(exc),
                }
            )

    return complete_transcripts, rag_documents, failed_videos


def save_json(path: Path, data: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def validate_dataset(complete_transcripts: list[dict], failed_videos: list[dict], rag_documents: list[dict]) -> int:
    print_header("STEP 5 - DATASET VALIDATION")
    successful_video_ids = {video["video_id"] for video in complete_transcripts}
    successful_count = len(successful_video_ids)
    print("Required videos:", NUM_VIDEOS)
    print("Successful videos:", successful_count)
    print("Failed videos:", len(failed_videos))
    print("Total RAG chunks:", len(rag_documents))

    if successful_count < NUM_VIDEOS:
        raise RuntimeError(
            f"Only {successful_count}/{NUM_VIDEOS} videos succeeded. See failed_videos.json."
        )

    print("OK: 25/25 COMPLETE")
    return successful_count


def upload_to_qdrant(documents: list[dict], successful_count: int) -> None:
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set")

    print_header("STEP 6 - LOADING ENGLISH RAG CHUNKS")
    texts = [document["english_text"].strip() for document in documents]
    if not texts:
        raise RuntimeError("No English text available for embeddings")
    print("RAG documents:", len(documents))

    print_header("STEP 7 - LOADING EMBEDDING MODEL")
    model = SentenceTransformer(EMBEDDING_MODEL)
    vector_size = model.get_sentence_embedding_dimension()
    print("Model:", EMBEDDING_MODEL)
    print("Vector size:", vector_size)

    print_header("STEP 8 - CREATING EMBEDDINGS")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print("Embeddings:", len(embeddings))
    print("Dimensions:", len(embeddings[0]))

    print_header("STEP 9 - CONNECTING QDRANT")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print("OK: Connected to Qdrant")

    print_header("STEP 10 - CREATING QDRANT COLLECTION")
    existing = [collection.name for collection in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        if not RECREATE_COLLECTION:
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' already exists. "
                "Use a new QDRANT_COLLECTION name or set QDRANT_RECREATE=true to replace it."
            )

        print("Deleting existing collection:", COLLECTION_NAME)
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print("OK: Collection created:", COLLECTION_NAME)

    print_header("STEP 11 - CREATING VECTOR + PAYLOAD")
    points = []
    for index, document in enumerate(documents):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[index].tolist(),
                payload={
                    "text": document["english_text"],
                    "hindi_text": document.get("hindi_text"),
                    "video_id": document["video_id"],
                    "title": document["title"],
                    "url": document["url"],
                    "chunk_id": document["chunk_id"],
                    "start_time": document["start_time"],
                    "end_time": document["end_time"],
                    "timestamp_url": document["timestamp_url"],
                },
            )
        )
    print("Points created:", len(points))

    print_header("STEP 12 - UPLOADING TO QDRANT")
    batch_size = 100
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch, wait=True)
        uploaded = min(start + batch_size, len(points))
        print(f"Uploaded {uploaded}/{len(points)}")

    print_header("STEP 13 - VERIFYING QDRANT")
    collection_info = client.get_collection(COLLECTION_NAME)
    print("Collection:", COLLECTION_NAME)
    print("Vector size:", vector_size)
    print("Distance: COSINE")
    print("Videos:", successful_count)
    print("RAG chunks:", len(documents))
    print("Vectors:", len(points))
    print("Qdrant points:", collection_info.points_count)
    if collection_info.points_count != len(points):
        raise RuntimeError("Qdrant point count does not match")


def main() -> int:
    videos = extract_playlist_videos()
    setup_argos()
    complete_transcripts, rag_documents, failed_videos = process_videos(videos)

    print_header("STEP 4 - SAVING JSON")
    save_json(COMPLETE_TRANSCRIPT_FILE, complete_transcripts)
    save_json(RAG_JSON_FILE, rag_documents)
    save_json(FAILED_FILE, failed_videos)
    print("Saved:", COMPLETE_TRANSCRIPT_FILE)
    print("Saved:", RAG_JSON_FILE)
    print("Saved:", FAILED_FILE)

    successful_count = validate_dataset(
        complete_transcripts, failed_videos, rag_documents
    )
    upload_to_qdrant(rag_documents, successful_count)

    print_header("YOUTUBE REVISER RAG DATASET READY")
    print("25 Hindi videos             OK")
    print("Hindi transcripts           OK")
    print("Hindi to English            OK")
    print("Timestamped chunks          OK")
    print("English embeddings          OK")
    print("Qdrant vectors              OK")
    print("Qdrant payloads             OK")
    print("Collection:", COLLECTION_NAME)
    print("Total Qdrant points:", len(rag_documents))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print()
        print("ERROR:", exc)
        raise SystemExit(1)
