

# 🏛️ Persian Orpheus TTS

**Persian Orpheus TTS** is a research project that brings Persian voice generation to life — by teaching an English-based speech model to **“think in English but speak in Persian.”**

Through a novel **Finglish-to-Persian fine-tuning approach**, we successfully trained a model capable of generating natural, fluent Persian speech, leveraging the linguistic and acoustic power of the original English base model.

---

## 🎯 Overview

Traditional Persian TTS systems are often limited by dataset scarcity and phonetic inconsistency.
In this work, we developed a **Persian speech synthesis model** trained on a **Finglish (Persian written with Latin characters)** dataset — bridging the gap between English phonetic understanding and Persian articulation.

Our fine-tuned model, **Orpheus-llama3b-fa**, produces smooth, natural Persian audio while preserving the high-quality prosody of the English base model.

---

## 🧠 Model

Fine-tuned model available here:
👉 [**David-ger/Orpheus-llama3b-fa-finetuned-gpt5-mini-4865**](https://huggingface.co/David-ger/Orpheus-llama3b-fa-finetuned-gpt5-mini-4865)

* **Base model:** Orpheus LLaMA-3B (English)
* **Fine-tuning method:** Finglish-to-Persian dataset alignment
* **Objective:** Persian phoneme alignment with English latent representations
* **Frameworks used:** PyTorch, Transformers, Accelerate

---

## 🗃️ Dataset

Dataset used for fine-tuning:
📦 [**David-ger/Persian-tts-finglish-orpheus**](https://huggingface.co/David-ger/Persian-tts-finglish-orpheus)

This dataset contains paired samples of:

* Finglish text → Persian phonetic transcription
* Synthetic and real Persian audio recordings
* Over 20 hours of aligned training data

---

## 🔊 Example Output

You can listen to an example output from the fine-tuned model below:

🎧 [`example.wav`](./example.wav)

You can listen to the generated Persian speech below:

<audio controls>
  <source src="example.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>

This clip demonstrates the model’s ability to generate **fluent, natural-sounding Persian** with expressive intonation and accurate pronunciation.

---

## 🚀 Usage

```python
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
import soundfile as sf

model = AutoModelForSpeechSeq2Seq.from_pretrained("David-ger/Orpheus-llama3b-fa-finetuned-gpt5-mini-4865")
processor = AutoProcessor.from_pretrained("David-ger/Orpheus-llama3b-fa-finetuned-gpt5-mini-4865")

text = "سلام دنیا! امروز روز زیباییه."
inputs = processor(text=text, return_tensors="pt")

with torch.no_grad():
    audio = model.generate(**inputs)

sf.write("output.wav", audio.cpu().numpy(), 22050)
print("✅ Audio generated successfully: output.wav")
```

---

## 📚 Citation

If you use this model or dataset in your research, please cite:

```bibtex
@misc{persian_orpheus_tts_2025,
  author       = {David-ger},
  title        = {Persian Orpheus TTS: Finglish-based Persian Speech Fine-tuning},
  year         = {2025},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/David-ger/Orpheus-llama3b-fa-finetuned-gpt5-mini-4865}},
}
```

---

## 💬 Acknowledgements

This project is inspired by the idea of **cross-lingual speech adaptation**, showing how models can reuse existing phonetic intelligence to master a new language.
Special thanks to open-source contributors and the Hugging Face community for their support.

---

## 🪶 License

This project is released under the **MIT License**.
Feel free to experiment, modify, and extend it for research or creative projects.

---

Would you like me to make a **Persian version** of this README too (با توضیحات فارسی برای ریپو گیت‌هاب)؟
