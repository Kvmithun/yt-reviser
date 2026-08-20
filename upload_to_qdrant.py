import argparse
import glob
import json
import os
import sys
import uuid


DEFAULT_JSON_FILES = [
    "youtube_hindi_to_english.json",
    "youtube_english_transcripts.json",
    "youtube_complete_transcripts.json",
]

REQUIRED_FIELDS = [
    "video_id",
    "title",
    "url",
    "chunk_id",
    "start_time",
    "end_time",
    "timestamp_url",
    "english_text",
]


def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def load_documents(explicit_path: str | None) -> tuple[str, list[dict]]:
    print_header("STEP 1 - FINDING JSON")

    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    else:
        candidates.extend(DEFAULT_JSON_FILES)
        for path in sorted(glob.glob("*.json")):
            if path not in candidates:
                candidates.append(path)

    for filename in candidates:
        if not os.path.exists(filename):
            print(f"Not found: {filename}")
            continue

        try:
            with open(filename, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            print(f"Error reading {filename}: {exc}")
            continue

        print(f"Checked: {filename}")
        print(
            "Documents:",
            len(data) if isinstance(data, list) else "Not a list",
        )

        if not isinstance(data, list) or not data:
            continue

        if "english_text" not in data[0]:
            print("Skipping: missing english_text in first document")
            continue

        print_header("VALID JSON FOUND")
        print("File:", filename)
        print("Documents:", len(data))
        return filename, data

    raise RuntimeError(
        "No valid non-empty JSON file found. Add a transcript JSON file with "
        "the required fields, or pass one with --json-file."
    )


def validate_documents(documents: list[dict]) -> None:
    print_header("STEP 2 - VALIDATING JSON")

    for index, document in enumerate(documents):
        for field in REQUIRED_FIELDS:
            if field not in document:
                raise RuntimeError(f"Document {index} is missing field: {field}")
            if field == "english_text" and not str(document[field]).strip():
                raise RuntimeError(f"Document {index} has empty english_text")

    print("OK: JSON structure is valid")
    print("Sample document:")
    print(json.dumps(documents[0], indent=2, ensure_ascii=False))


def build_points(documents: list[dict], embeddings) -> list:
    from qdrant_client.models import PointStruct

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
    return points


def run_dry(documents: list[dict], json_file: str) -> None:
    print_header("DRY RUN OUTPUT")
    print("Mode: dry-run")
    print("File:", json_file)
    print("Documents:", len(documents))
    print("Texts ready:", sum(1 for doc in documents if doc["english_text"].strip()))
    print("First title:", documents[0]["title"])
    print("First chunk id:", documents[0]["chunk_id"])
    print("Status: input validation passed")


def run_full(documents: list[dict], collection_name: str, model_name: str) -> None:
    try:
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
    except ModuleNotFoundError as exc:
        missing = exc.name or "dependency"
        raise RuntimeError(
            f"Missing dependency: {missing}. Install with "
            "'pip install sentence-transformers qdrant-client'"
        ) from exc

    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        raise RuntimeError(
            "QDRANT_URL and QDRANT_API_KEY must be set in the environment for full upload."
        )

    print_header("STEP 3 - LOADING EMBEDDING MODEL")
    model = SentenceTransformer(model_name)
    vector_size = model.get_sentence_embedding_dimension()
    print("Model:", model_name)
    print("Vector size:", vector_size)

    print_header("STEP 4 - CONNECTING QDRANT")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    print("OK: connected to Qdrant")

    print_header("STEP 5 - CREATING COLLECTION")
    existing = [item.name for item in client.get_collections().collections]
    if collection_name in existing:
        print("Deleting existing collection:", collection_name)
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print("OK: collection created:", collection_name)

    print_header("STEP 6 - PREPARING TEXT")
    texts = [document["english_text"].strip() for document in documents]
    print("Texts:", len(texts))

    print_header("STEP 7 - CREATING EMBEDDINGS")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print("OK: embeddings created:", len(embeddings))
    print("Dimensions:", len(embeddings[0]))

    print_header("STEP 8 - CREATING VECTOR + PAYLOAD")
    points = build_points(documents, embeddings)
    print("OK: points created:", len(points))

    print_header("STEP 9 - UPLOADING TO QDRANT")
    batch_size = 100
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=collection_name, points=batch, wait=True)
        uploaded = min(start + batch_size, len(points))
        print(f"Uploaded {uploaded}/{len(points)}")

    print_header("STEP 10 - VERIFYING QDRANT")
    collection_info = client.get_collection(collection_name)
    print("Collection:", collection_name)
    print("Vector size:", vector_size)
    print("Distance: COSINE")
    print("JSON documents:", len(documents))
    print("Vectors uploaded:", len(points))
    print("Qdrant points:", collection_info.points_count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate transcript JSON and optionally upload embeddings to Qdrant."
    )
    parser.add_argument("--json-file", help="Path to a transcript JSON file")
    parser.add_argument(
        "--collection-name",
        default="youtube_reviser",
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate and print a local summary",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        json_file, documents = load_documents(args.json_file)
        validate_documents(documents)

        if args.dry_run:
            run_dry(documents, json_file)
        else:
            run_full(documents, args.collection_name, args.embedding_model)
    except Exception as exc:
        print()
        print(f"ERROR: {exc}")
        return 1

    print_header("DONE")
    print("Workflow completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
