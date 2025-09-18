# Project Overview

This repository is structured into three main components: **Transcription**, **Scoring**, and **Fine-Tuning**.  
Each module corresponds to a different step of the pipeline, combining speech-to-text, evaluation, and model adaptation.

---

## Transcription
This module is based on **OpenAI Whisper**, a state-of-the-art speech recognition system.  
We adapted the original implementation to our needs: audio files are transcribed and the outputs are stored in the `Documents` directory.  
Whisper leverages **transformer-based sequence-to-sequence modeling** and was trained on a large-scale multilingual dataset, allowing robust transcription across different languages and acoustic conditions.

---

## Scoring
The **Scoring** module retrieves the transcriptions from the `Documents` directory and generates a dataset by evaluating them through **LLMs running on Ollama**.  
The scoring process relies on **prompt engineering** and **model-based evaluation**, a method where large language models act as judges of transcription quality.  
This allows us to quantify aspects such as accuracy, semantic consistency, and fluency, turning raw transcriptions into labeled data suitable for further analysis.

---

## Fine-Tuning
The **Fine-Tuning** module is an application of the **Unsloth** framework.  
It implements **parameter-efficient fine-tuning techniques (PEFT)** such as **LoRA (Low-Rank Adaptation)** to adapt a pretrained language model to our domain-specific dataset with reduced computational cost.  
This approach enables us to improve performance on transcription-related tasks while minimizing training time and hardware requirements.  

By combining Whisper for transcription, LLM-based scoring with Ollama, and PEFT fine-tuning through Unsloth, the pipeline showcases an end-to-end methodology for building and adapting speech-to-text systems enhanced by large language models.


