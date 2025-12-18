import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path so pipeline module can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.end_to_end import VoiceAssistantPipeline


def main():
    parser = argparse.ArgumentParser(description="Run the in-car voice assistant pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Process a plain text command (bypasses STT)")
    group.add_argument("--audio", type=Path, help="Path to raw PCM16 audio file to process")
    parser.add_argument("--classifier", choices=["rule", "svm"], default="rule", help="Classifier backend")
    parser.add_argument("--rag", choices=["kb", "faiss", "off"], default="kb", help="RAG backend")
    args = parser.parse_args()

    use_rag = args.rag != "off"
    pipeline = VoiceAssistantPipeline(use_rag=use_rag, classifier_type=args.classifier, rag_type=(args.rag if use_rag else "kb"))

    if args.text is not None:
        result = pipeline.process_text(args.text)
    else:
        audio_bytes = args.audio.read_bytes()
        result = pipeline.process_audio(audio_bytes)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
