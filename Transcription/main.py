import argparse
import json
import sys
from pathlib import Path
from transcriber import WhisperTranscriber
from utils.audio import AudioProcessor
from utils.output import format_output


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Speech-to-text with speaker diarization"
    )
    parser.add_argument(
        "--audio", type=str, required=True, help="Path to the audio file"
    )
    parser.add_argument(
        "--language", type=str, default="it", help="Language of the audio (default: it)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        help="Whisper model size (default: large-v2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path (default: input_file_transcript.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for processing (cuda/cpu)",
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    try:
        args = parse_arguments()

        # Set output path if not specified
        if not args.output:
            output_path = Path(args.audio).with_suffix(".json")
        else:
            output_path = Path(args.output)

        print(f"Processing audio file: {args.audio}")

        # Initialize components
        audio_processor = AudioProcessor()
        transcriber = WhisperTranscriber(
            model_size=args.model, device=args.device, language=args.language
        )

        # Process audio
        print("Processing audio...")
        processed_audio = audio_processor.process(args.audio)

        # Perform transcription
        print("Transcribing audio...")
        transcription = transcriber.transcribe(processed_audio)

        # Format results
        print("Formatting results...")
        final_output = format_output(transcription)

        # Save results
        print(f"Saving results to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)

        print("Processing complete!")

    except Exception as e:
        print(f"Errore durante l'esecuzione: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import torch

    main()
