# 🧬 Biomedical Image Search Engine

> A minimal, zero-shot biomedical image search engine powered by CLIP + FAISS. Query using an image or a text description.

---

## 🚀 Live Demo

👉 [**Launch on Hugging Face Spaces**](https://huggingface.co/spaces/Scnf/biomedica-search)

Upload a biomedical image **or** type a medical concept to retrieve the most visually or semantically similar slides.

📁 Need examples? Download 9 sample slides (one per class):  
👉 [**Google Drive Folder**](https://drive.google.com/drive/folders/1eCWP_UnL2etBhWhtAKVr1YHSNolkIIRI?usp=sharing)

---

## 🧠 What It Does

This tool retrieves the top-5 most relevant histopathology images from a subset of the **PathMNIST** dataset based on a **query**:

- 🖼 Upload an image → find visually similar samples
- 💬 Type a biomedical description → get semantically related slides

Under the hood, both modalities are encoded into the **same embedding space** using OpenAI’s **CLIP** model.

---

## 🔍 Key Features

- **Multimodal search**: text or image input
- **Zero-shot**: no task-specific fine-tuning
- **Real-time inference** on the Hugging Face platform
- **Lightweight dataset** (500 samples from PathMNIST)
- Optimized with **FAISS** for fast vector retrieval

---

## 🛠️ Tech Stack

| Component        | Tool                         |
|------------------|-------------------------------|
| Embedding model  | OpenAI [CLIP](https://openai.com/research/clip)         |
| Similarity search| [FAISS](https://github.com/facebookresearch/faiss)      |
| Dataset          | [PathMNIST](https://medmnist.com/) (via MedMNIST)       |
| Web UI           | [Gradio](https://www.gradio.app/)                       |
| Hosting          | Hugging Face Spaces           |

---

## 📁 Project Structure

<pre> ```bash biomedica-search/ ├── app.py # Main Gradio app ├── README.md # One-pager già pronto ├── requirements.txt # Tutte le dipendenze (Gradio, Transformers, etc.) ├── data/ # Vuoto (solo .gitkeep o README) ├── examples/ # 9 immagini di esempio │ └── class_0.png ... ├── save_images.py # Script per generare immagini di esempio ├── .gitignore # Ignora tutto ciò che è inutile └── .gitattributes # (opzionale, solo se usi Git LFS) ``` </pre>        



## 📚 Background

This project was inspired by the recent Stanford research paper:

> **[BIOMEDICA: Large-Scale Zero-Shot Biomedical Image Classification and Captioning]([https://arxiv.org/abs/2311.17088](https://minwoosun.github.io/biomedica-website/))**

It highlights how general-purpose models like CLIP can perform competitively on medical datasets **without domain-specific tuning**. This project is a lightweight reproduction of that idea, with real-time inference and an accessible interface.

---

## 🧪 How to Use It

1. **Go to the [demo](https://huggingface.co/spaces/Scnf/biomedica-search)**
2. Upload a sample slide **or** type a prompt like:
   - "pink stained epithelial cells"
   - "microscopic blood smear with nucleus"
3. Wait a few seconds, and receive 5 images from the dataset ranked by similarity

---

## 📁 Example Inputs

You can test it with:

- ✅ Your own biomedical image (uploaded in JPG or PNG)
- ✅ Free-text descriptions ("lung carcinoma", "purple-stained cells")
- ✅ [Pre-curated examples](https://drive.google.com/drive/folders/1eCWP_UnL2etBhWhtAKVr1YHSNolkIIRI?usp=sharing) from the 9 PathMNIST classes

---

## 👨‍🔬 Author

**Alessandro Scanferla**  
Founder-minded · Tech explorer · Passionate about biotech and AI  
💼 [LinkedIn](https://www.linkedin.com/in/alessandroscanferla-/)

This project was created to demonstrate engineering skills applied to scientific domains, especially in preparation for my application to the EPFL MTE program.

---

## 🧩 Future Work

- Switch to real clinical datasets (e.g., BIOMEDICA or TCGA)
- Add class labels or captions to retrieved images
- Enable hybrid queries (text + image)
- Fine-tune on downstream tasks (e.g., diagnosis, segmentation)

---

## 📜 License

MIT License — free to use, remix, and build on.

---

_If this sparks any ideas or you'd like to collaborate, feel free to reach out!_
