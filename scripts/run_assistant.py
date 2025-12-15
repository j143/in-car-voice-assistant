import argparse
import json
from pathlib import Path

from pipeline.end_to_end import VoiceAssistantPipeline


def main():
    parser = argparse.ArgumentParser(description="Run the in-car voice assistant pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Process a plain text command (bypasses STT)")
    group.add_argument("--audio", type=Path, help="Path to raw PCM16 audio file to process")
    args = parser.parse_args()

    pipeline = VoiceAssistantPipeline(use_rag=True)

    if args.text is not None:
        result = pipeline.process_text(args.text)
    else:
        audio_bytes = args.audio.read_bytes()
        result = pipeline.process_audio(audio_bytes)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
