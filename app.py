import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import time

from flask import Flask, request, jsonify, send_file
import os
from os import path as osp
from pathlib import Path
import cv2
import lancedb
import logging
import webvtt
from tqdm import tqdm
from utils import download_video, get_transcript_vtt, encode_image, bt_embeddings, get_video_id_from_url
import torch
from PIL import Image
import socket

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to find an available port
def find_available_port(start_port=5000):
    logger.info(f"Searching for an available port starting from {start_port}")
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                s.close()
                logger.info(f"Found available port: {port}")
                return port
            except OSError:
                logger.debug(f"Port {port} is in use, trying next...")
                port += 1
                if port > 65535:
                    logger.error("No available ports found")
                    raise ValueError("No available ports found")

# Determine available port
PORT = find_available_port()
logger.info(f"Server will run on port: {PORT}")

app = Flask(__name__)

# Configuration
VIDEO_DIR = "./shared_data/videos/video1"
LANCEDB_HOST_FILE = "./shared_data/.lancedb"
TBL_NAME = "vectorstore"
N_CONTEXT = 7
Path(VIDEO_DIR).mkdir(parents=True, exist_ok=True)
Path(osp.dirname(LANCEDB_HOST_FILE)).mkdir(parents=True, exist_ok=True)
logger.info(f"Configured directories: VIDEO_DIR={VIDEO_DIR}, LANCEDB_HOST_FILE={LANCEDB_HOST_FILE}")

# Custom frame extraction and metadata generation using OpenCV
def extract_and_save_frames_and_metadata(video_filepath, video_transcript_filepath, extracted_frames_path):
    logger.info(f"Opening video file: {video_filepath}")
    cap = cv2.VideoCapture(video_filepath)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_filepath}")
        raise ValueError(f"Could not open video file: {video_filepath}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    logger.info(f"Video details - FPS: {fps}, Total Frames: {total_frames}, Duration: {duration} seconds")
    metadatas = []
    logger.info(f"Reading transcript from: {video_transcript_filepath}")
    captions = webvtt.read(video_transcript_filepath)
    logger.info(f"Found {len(captions)} captions in transcript")

    for frame_idx in range(0, total_frames, frame_interval):
        logger.debug(f"Processing frame index: {frame_idx}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Failed to read frame at index {frame_idx}, breaking loop")
            break
        
        timestamp = frame_idx / fps
        frame_path = osp.join(extracted_frames_path, f"frame_{int(timestamp)}.jpg")
        logger.debug(f"Saving frame to: {frame_path}")
        cv2.imwrite(frame_path, frame)
        
        # Find matching caption
        transcript = ""
        for caption in captions:
            if caption.start_in_seconds <= timestamp <= caption.end_in_seconds:
                transcript = caption.text
                break
        metadatas.append({
            "extracted_frame_path": frame_path,
            "transcript": transcript or "No transcript available",
            "timestamp": timestamp
        })
    
    cap.release()
    logger.info(f"Extracted and saved {len(metadatas)} frames with metadata")
    return metadatas

# Modified bt_embeddings with MPS support
def get_embeddings(texts, image_paths=None):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device for embeddings: {device}")
    
    if image_paths:
        images = [Image.open(path).convert("RGB") for path in image_paths]
        base64_images = [encode_image(img) for img in images]
        embeddings = [bt_embeddings(text, img) for text, img in zip(texts, base64_images)]
    else:
        embeddings = [bt_embeddings(text) for text in texts]
    
    logger.info(f"Generated {len(embeddings)} embeddings")
    return [emb for emb in embeddings]  # Flatten if needed

# Simple vector store management
class SimpleVectorStore:
    def __init__(self, uri, table_name):
        logger.info(f"Connecting to LanceDB at: {uri}")
        self.db = lancedb.connect(uri)
        self.table_name = table_name
        self.table = self.db.open_table(table_name) if table_name in self.db else None
        if self.table:
            logger.info(f"Opened existing table: {table_name}")
        else:
            logger.info(f"No existing table found: {table_name}")

    def from_data(self, texts, image_paths, metadatas, mode="overwrite"):
        if mode == "overwrite" and self.table_name in self.db:
            logger.info(f"Overwriting existing table: {self.table_name}")
            self.db.drop_table(self.table_name)
        logger.info(f"Building new table: {self.table_name} with {len(texts)} entries")
        embeddings = get_embeddings(texts, image_paths)
        data = [
            {"vector": emb, "text": text, "image_path": img, **meta}
            for emb, text, img, meta in zip(embeddings, texts, image_paths, metadatas)
        ]
        self.table = self.db.create_table(self.table_name, data=data)
        logger.info(f"Table {self.table_name} built successfully with {len(data)} records")
        return self

    def as_retriever(self, k=3):
        class Retriever:
            def __init__(self, table):
                self.table = table

            def invoke(self, query):
                logger.info(f"Querying table {TBL_NAME} with: {query}")
                query_emb = get_embeddings([query])[0]
                # Simulate some processing time (e.g., 2-3 seconds)
                time.sleep(2)  # Adjust based on actual query complexity
                results = self.table.search(query_emb).limit(k).to_list()
                logger.info(f"Retrieved {len(results)} results from table {TBL_NAME}")
                return [
                    type('Result', (), {
                        'page_content': res['text'],
                        'metadata': {k: v for k, v in res.items() if k != 'vector'}
                    }) for res in results
                ]
        return Retriever(self.table)

@app.route("/")
def index():
    logger.info("Serving index.html")
    with open("index.html", "r") as f:
        return f.read()

@app.route("/process-video", methods=["POST"])
def process_video():
    logger.info("Received /process-video request")
    try:
        data = request.get_json()
        youtube_url = data.get("youtube_url")
        if not youtube_url:
            logger.warning("No URL provided in request")
            return jsonify({"error": "URL is required"}), 400
        video_id = get_video_id_from_url(youtube_url)
        if not video_id:
            logger.warning(f"Invalid YouTube URL provided: {youtube_url}")
            return jsonify({"error": "Invalid YouTube URL"}), 400

        logger.info(f"Starting processing for URL: {youtube_url} (Video ID: {video_id})")
        # Download video and transcript
        logger.info("Initiating video download...")
        video_filepath = download_video(youtube_url, VIDEO_DIR)
        logger.info(f"Video downloaded to: {video_filepath}")
        logger.info("Initiating transcript download...")
        video_transcript_filepath = get_transcript_vtt(youtube_url, VIDEO_DIR)
        logger.info(f"Transcript downloaded to: {video_transcript_filepath}")
        extracted_frames_path = osp.join(VIDEO_DIR, "extracted_frame")
        Path(extracted_frames_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created/verified extracted frames directory: {extracted_frames_path}")

        # Extract frames and metadata
        logger.info("Starting frame extraction and metadata generation...")
        metadatas = extract_and_save_frames_and_metadata(video_filepath, video_transcript_filepath, extracted_frames_path)

        # Process transcripts
        logger.info("Processing transcripts...")
        video_trans = [vid["transcript"] for vid in metadatas]
        video_img_path = [vid["extracted_frame_path"] for vid in metadatas]
        updated_video_trans = [
            " ".join(video_trans[i - int(N_CONTEXT / 2) : i + int(N_CONTEXT / 2)])
            if i - int(N_CONTEXT / 2) >= 0
            else " ".join(video_trans[0 : i + int(N_CONTEXT / 2)])
            for i in range(len(video_trans))
        ]
        logger.info(f"Processed {len(updated_video_trans)} transcript entries")

        # Update metadata with URL-friendly paths
        logger.info("Updating metadata with URL-friendly paths...")
        for i, metadata in enumerate(metadatas):
            metadata["transcript"] = updated_video_trans[i]
            rel_path = osp.relpath (metadata["extracted_frame_path"], start=".")
            metadata["extracted_frame_path"] = f"/images/{rel_path.replace(os.sep, '/')}"
            metadatas[i] = metadata
        logger.info("Metadata updated successfully")

        # Initialize vector store
        logger.info("Building LanceDB vector store...")
        vector_store = SimpleVectorStore(LANCEDB_HOST_FILE, TBL_NAME)
        vector_store.from_data(updated_video_trans, video_img_path, metadatas)

        logger.info("Processing completed successfully")
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/query", methods=["POST"])
def query():
    logger.info("Received /query request")
    try:
        data = request.get_json()
        query_text = data.get("query")
        if not query_text:
            logger.warning("No query text provided")
            return jsonify({"error": "Query is required"}), 400

        logger.info(f"Starting query: {query_text}")
        vector_store = SimpleVectorStore(LANCEDB_HOST_FILE, TBL_NAME)
        if not vector_store.table:
            logger.warning("Vector store not found, please process a video first")
            return jsonify({"error": "Vector store not found. Please process a video first"}), 400
        retriever = vector_store.as_retriever(k=3)
        results = retriever.invoke(query_text)
        formatted_results = [
            {
                "caption": res.page_content,
                "image_path": res.metadata["extracted_frame_path"]
            }
            for res in results
        ]
        logger.info(f"Query completed with {len(formatted_results)} results")
        return jsonify({"results": formatted_results})
    except Exception as e:
        logger.error(f"Error querying vector store: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/images/<path:filename>")
def serve_image(filename):
    logger.info(f"Received request to serve image: {filename}")
    try:
        filepath = osp.join(".", filename)
        logger.debug(f"Serving image from: {filepath}")
        return send_file(filepath)
    except Exception as e:
        logger.error(f"Error serving image {filename}: {str(e)}", exc_info=True)
        return jsonify({"error": "Image not found"}), 404

if __name__ == "__main__":
    logger.info("Starting Flask application")
    app.run(debug=True, host="0.0.0.0", port=PORT)