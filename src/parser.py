import os
import re
import pandas as pd

HEADER_PATTERN = re.compile(
    r"Expert\s+\d+\s+[–-]\s*(.+?)\s*"
    r"Role:\s*(.+?)\s*"
    r"Market:\s*(.+?)\s*$",
    re.DOTALL
)

TURN_PATTERN = re.compile(
    r"(?P<timestamp>\d{2}:\d{2})\s*"
    r"(?P<speaker>[^:\n]+):\s*"
    r"(?P<text>.*?)(?=\n\s*\d{2}:\d{2}\s*\n|\Z)",
    re.DOTALL
)

def parse_transcript(filepath, country):
    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read().replace("\r\n", "\n").strip()

    header_text = text.split("00:00", 1)[0].strip()
    header_match = HEADER_PATTERN.search(header_text)

    if not header_match:
        raise ValueError(f"Could not read header in: {os.path.basename(filepath)}")

    expert_name = header_match.group(1).strip()
    expert_role = header_match.group(2).strip()
    transcript_id = os.path.splitext(os.path.basename(filepath))[0].lower()

    turns = []
    for match in TURN_PATTERN.finditer(text):
        turns.append({
            "timestamp": match.group("timestamp").strip(),
            "speaker": match.group("speaker").strip(),
            "text": " ".join(match.group("text").split())
        })

    chunks = []
    current_question = None

    for turn in turns:
        if turn["speaker"].lower() == "interviewer":
            current_question = turn["text"]
        else:
            chunks.append({
                "chunk_id": f"{transcript_id}_{turn['timestamp'].replace(':', '')}",
                "transcript_id": transcript_id,
                "country": country,
                "expert_name": expert_name,
                "expert_role": expert_role,
                "timestamp": turn["timestamp"],
                "question": current_question,
                "speaker": turn["speaker"],
                "answer": turn["text"],
                "source_file": os.path.basename(filepath)
            })

    return chunks

def build_dataset(raw_dir="data/raw"):
    files = {
        "Transcript_1_France.txt": "France",
        "Transcript_2_Germany.txt": "Germany",
        "Transcript_3_UK.txt": "United Kingdom"
    }

    all_chunks = []

    for filename, country in files.items():
        filepath = os.path.join(raw_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing transcript: {filepath}")

        all_chunks.extend(parse_transcript(filepath, country))

    return pd.DataFrame(all_chunks)

if __name__ == "__main__":
    df = build_dataset()
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/processed_chunks.csv", index=False)
    print(df[["country", "expert_name", "timestamp", "question", "answer"]])