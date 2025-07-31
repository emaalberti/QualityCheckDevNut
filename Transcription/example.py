"""
Esempio di utilizzo del sistema di trascrizione audio.
"""

import os
import json
from pathlib import Path


def main():
    """Funzione principale per l'esempio."""
    # Verifica se esiste un file audio di esempio
    audio_file = "audio.m4a"
    if not Path(audio_file).exists():
        print(f"File audio di esempio '{audio_file}' non trovato.")
        print("Utilizza un tuo file audio o rinomina il tuo file in 'audio.m4a'.")
        return

    # Esegui la trascrizione
    print("\n=== Avvio trascrizione ===\n")

    # Comando di esecuzione
    command = f"python main.py --audio {audio_file} --language it"
    print(f"Esecuzione comando: {command}\n")

    # Esegui il comando
    os.system(command)

    # Verifica se il file di output è stato creato
    output_file = Path(audio_file).with_suffix(".json")
    if output_file.exists():
        print(f"\nFile di output creato: {output_file}")

        # Mostra il risultato
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print("\n=== Trascrizione ===\n")
        print(data["transcription"])
        print(f"\nNumero di segmenti elaborati: {data['metadata']['num_segments']}")
    else:
        print(f"\nErrore: File di output {output_file} non creato.")


if __name__ == "__main__":
    main()
