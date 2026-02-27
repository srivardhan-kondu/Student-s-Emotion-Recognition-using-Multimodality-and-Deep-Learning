# 🎓 Student's Emotion Recognition using Multimodality and Deep Learning

> An AI-powered system that **automatically detects how a student is feeling** by analyzing their **face**, **voice**, and **written words** — all at the same time — and combines the results into one accurate emotion reading.

---

## 📋 Table of Contents

- [What Does This Project Do?](#-what-does-this-project-do)
- [Why Is This Useful?](#-why-is-this-useful)
- [How Does It Work? (Simple Explanation)](#-how-does-it-work-simple-explanation)
- [The 3 AI Models Explained](#-the-3-ai-models-explained)
- [How Results Are Combined](#-how-results-are-combined)
- [Project Structure](#-project-structure)
- [How to Install and Run](#-how-to-install-and-run)
- [How to Use the Dashboard](#-how-to-use-the-dashboard)
- [Model Performance](#-model-performance)
- [Technology Used](#-technology-used)
- [Datasets Used for Training](#-datasets-used-for-training)

---

## 🎯 What Does This Project Do?

This system **reads a student's emotion** using three different methods simultaneously:

| Input Method | What It Analyzes | Example |
|---|---|---|
| 📷 **Face Photo / Camera** | Facial expressions | Is the student smiling? Frowning? Looking scared? |
| 🎤 **Voice / Audio Recording** | Tone, pitch, energy of speech | Is the voice loud and angry? Quiet and sad? |
| 📝 **Text / Typed Message** | Meaning of written words | "I love this class!" vs "I hate this assignment" |

It can detect **6 emotions**:

| Emotion | What It Looks Like |
|---|---|
| 😊 **Happy** | Smiling face, cheerful voice, positive words |
| 😢 **Sad** | Downturned lips, slow quiet voice, negative words |
| 😠 **Angry** | Furrowed brows, loud harsh voice, aggressive words |
| 😐 **Neutral** | Relaxed face, calm voice, factual statements |
| 😨 **Fear** | Wide eyes, trembling voice, anxious words |
| 😲 **Surprise** | Raised eyebrows, sudden pitch change, unexpected news |

---

## 💡 Why Is This Useful?

**The Problem:**
- Teachers in large classrooms cannot monitor every student's emotions
- Online learners often feel disconnected and their struggles go unnoticed
- A student might *say* they are fine but *look* confused or sad

**Our Solution:**
- This AI monitors student emotions automatically in real-time
- By combining face + voice + text, it is far more accurate than using just one method
- Educators can use this data to identify struggling students early and offer help

**Example of Why Multiple Methods Matter:**

> Imagine a student is smiling 😊 (face says: happy)  
> but says "Yeah right, this makes total sense" in a sarcastic tone 😠 (voice says: angry)  
> and types "I don't understand anything" 😢 (text says: sad)  
>
> A system using only the face would say "Happy" — completely wrong!  
> Our system **combines all three** and correctly identifies the student is **frustrated**.

---

## 🧠 How Does It Work? (Simple Explanation)

Think of it like three expert judges each watching the same student:

```
👁️ JUDGE 1 (Face Expert)     → Looks at the face photo       → Says "I think: HAPPY (85%)"
🎤 JUDGE 2 (Voice Expert)     → Listens to the audio          → Says "I think: HAPPY (92%)"
📝 JUDGE 3 (Text Expert)      → Reads what student wrote      → Says "I think: HAPPY (98%)"
                                                                          ↓
                                                            🧮 COMBINE ALL THREE
                                                                          ↓
                                                         ✅ FINAL ANSWER: HAPPY (89%)
```

**Step-by-step flow:**

1. **You provide input** — upload a photo, record audio, or type some text (you can use all three or just one)
2. **Three AI models independently analyze** each type of input
3. **Each model gives its prediction** — e.g., "I'm 80% sure this is 'happy'"
4. **A Fusion Engine combines all predictions** — it's smart enough to trust the more confident model more
5. **One final answer is given** — the most likely emotion with a confidence percentage

---

## 🔬 The 3 AI Models Explained

### 👁️ Model 1 — Facial Emotion Recognition (CNN)

**What is a CNN?**
A CNN (Convolutional Neural Network) is a type of AI that was designed to **understand images**, just like how your eyes and brain process visual information.

**How it works:**
1. The system first **detects the face** in the photo using a face detector (like a smartphone's face scanner)
2. The face is **converted to grayscale** (black & white) and shrunk to a tiny 48×48 pixel image
3. The CNN **scans the image in small patches** — looking for patterns like:
   - Curved lips (smile) → likely happy
   - Furrowed eyebrows → likely angry
   - Wide eyes → likely surprised or scared
4. It outputs a **percentage score for each of the 6 emotions**

**Analogy:** Imagine teaching a child to recognize emotions by showing them thousands of faces with labels — "this is happy," "this is sad." After enough examples, the child learns the patterns. That's exactly what this CNN did — it was trained on **35,887 face images**.

---

### 🎤 Model 2 — Speech Emotion Recognition (BiLSTM with Attention)

**What is this model?**
This AI listens to audio and understands emotion from **HOW something is said**, not what words are used.

**Step 1 — Feature Extraction (Converting Sound to Numbers)**

The audio file is analyzed to extract numerical features:
- **MFCC** (Mel-Frequency Cepstral Coefficients) — captures the *tone and pitch* of the voice
- **Delta MFCC** — captures how quickly the tone *changes* over time
- **Delta-Delta MFCC** — captures the *acceleration* of those changes (like a sudden shout)

Think of it like this: instead of reading words, the AI reads the "shape" of the sound wave.

**Step 2 — BiLSTM (Bidirectional Long Short-Term Memory)**

This part of the model reads the audio features **both forward and backward in time**:
- Forward: "After this quiet moment, the voice got louder" → building anger?
- Backward: "Before the shout, there was a calm pause" → deliberate emphasis?

Combined, it understands **the full emotional arc** of the speech.

**Step 3 — Attention Mechanism (Focusing on What Matters)**

Not all parts of a speech clip are equally emotional. The attention mechanism focuses on the most expressive moments:

```
Time:     [0.1s]   [0.2s]   [0.3s]   [0.4s]   [0.5s]   [0.6s]
Audio:    "I..."   "just"   "can't"  "TAKE"    "this"   "anymore"
Focus:     5%       10%      15%       40%       15%       15%
                                        ↑
                    Model pays MOST attention to the loud emotional word
```

**Trained on:** 1,440 audio clips from 24 professional actors (RAVDESS dataset)

---

### 📝 Model 3 — Text Emotion Recognition (BERT)

**What is BERT?**
BERT (Bidirectional Encoder Representations from Transformers) is a powerful AI from Google that was trained on the **entire English Wikipedia and thousands of books**. It deeply understands human language.

**How it works:**

Traditional language AI reads left to right: "I" → "love" → "this"  
BERT reads **all directions at once**, understanding context fully:

```
Sentence: "I can't believe how good this is!"

Traditional AI: reads word by word, might miss sarcasm
BERT:           understands "can't believe" + "how good" together = genuine excitement
```

**After BERT understands the text**, a classification layer converts that understanding into one of the 6 emotions.

**Trained on:** ~58,000 real Reddit comments (GoEmotions dataset by Google)

---

## 🔀 How Results Are Combined

This is called the **Fusion Engine** — the brain that takes predictions from all 3 models and makes a final decision.

### The Smart Way (Calibrated Fusion — Default)

**Problem with raw AI outputs:** AI models are often overconfident. For example, the face model might say "99% happy" when it's really more like "75% happy." This is called **miscalibration**.

**Step 1 — Temperature Calibration (Fixing Overconfidence)**

Think of it like adjusting a car's speedometer that always reads 20% too high. We apply a correction factor called a "temperature" to make the predictions more realistic:

```
Before correction: Face says "99% happy"   ← Too confident
After correction:  Face says "78% happy"   ← More realistic
```

| Model | Temperature Applied | Why |
|---|---|---|
| Face | 1.5 | Most overconfident — needs the most correction |
| Speech | 1.3 | Moderately overconfident |
| Text | 1.2 | Slightly overconfident |

**Step 2 — Confidence Gating (Ignoring Unreliable Inputs)**

If one model is very unsure (less than 30% confident), it gets **completely skipped** in the final calculation. Why take advice from someone who says "I'm only 20% sure"?

**Step 3 — Weighted Combination**

Each model has a base importance (weight). The final answer is a weighted average:

| Modality | Base Weight | Why |
|---|---|---|
| 😊 Face | 40% | Facial expressions are the strongest signal for basic emotions |
| 🎤 Speech | 30% | Voice tone carries a lot of emotional information |
| 📝 Text | 30% | Words convey meaning but can sometimes be ambiguous |

**Example Calculation:**
```
Face:   happy=80%, confidence=0.80 → effective weight = 0.40 × 0.80 = 0.32
Speech: happy=90%, confidence=0.95 → effective weight = 0.30 × 0.95 = 0.285
Text:   happy=20%, confidence=0.25 → SKIPPED (below 30% threshold)

Final emotion = weighted average of face + speech only → HAPPY ✅
```

### Other Available Fusion Strategies

| Strategy | How It Works | Best For |
|---|---|---|
| **Calibrated** (Default) | Smart — fixes overconfidence + ignores bad inputs | Most accurate, production use |
| **Weighted** | Simple average with fixed weights | Quick, predictable results |
| **Adaptive** | Weights change based on how confident each model is | Variable quality inputs |
| **Voting** | Each model votes, majority wins | Fast consensus decisions |

---

## 📁 Project Structure

Here is every file and folder explained in plain English:

```
Student's Emotion Recognition/
│
├── 📄 run_dashboard.py          ← ⭐ THE MAIN FILE — Run this to start the app!
├── 📄 download_models.py        ← ⭐ Run this FIRST to download the AI models
├── 📄 requirements.txt          ← List of all Python libraries needed
├── 📄 README.md                 ← This file you are reading now
│
├── 📁 src/                      ← All the source code (the brain of the system)
│   │
│   ├── 📄 config.py             ← Global settings (model paths, emotion labels, etc.)
│   │
│   ├── 📁 facial_recognition/   ← Everything related to analyzing faces
│   │   ├── model_architecture.py   ← Defines the CNN neural network structure
│   │   ├── emotion_model.py        ← Loads & runs the facial model
│   │   ├── face_detector.py        ← Detects faces in photos (like auto-focus)
│   │   ├── data_preprocessing.py   ← Prepares images for training
│   │   └── train.py                ← Script to train the facial model
│   │
│   ├── 📁 speech_analysis/      ← Everything related to analyzing voice
│   │   ├── emotion_model.py        ← The BiLSTM model for voice emotion
│   │   ├── audio_features.py       ← Extracts MFCC features from audio
│   │   ├── speech_recognition.py   ← Audio file loading utilities
│   │   └── train.py                ← Script to train the speech model
│   │
│   ├── 📁 text_analysis/        ← Everything related to analyzing text
│   │   ├── emotion_model.py        ← The BERT model for text emotion
│   │   ├── text_preprocessing.py   ← Cleans and tokenizes text input
│   │   └── train.py                ← Script to train the text model
│   │
│   ├── 📁 fusion/               ← Combines predictions from all 3 models
│   │   ├── multimodal_fusion.py    ← The 4 fusion strategies + calibration logic
│   │   └── multimodal_predictor.py ← Easy-to-use API to get predictions
│   │
│   ├── 📁 dashboard/            ← The web interface (what you see in the browser)
│   │   └── app.py               ← Streamlit dashboard — the visual front-end
│   │
│   └── 📁 utils/                ← Helper utilities
│       ├── helpers.py           ← Common helper functions
│       └── voice_recorder.py    ← Tool for recording audio directly in the app
│
├── 📁 saved_models/             ← Where the trained AI models are stored
│   ├── facial_emotion_model.h5  ← Facial CNN model (~43 MB)
│   ├── speech_emotion_model.h5  ← Speech BiLSTM model (~6 MB)
│   └── text_bert_model/         ← BERT text model folder (~438 MB)
│
├── 📁 data/                     ← Training datasets (not included — too large)
├── 📁 docs/                     ← Technical documentation and training charts
└── 📁 tests/                    ← Automated tests for code quality
```

---

## 🚀 How to Install and Run

> ⚠️ **Before starting** — Make sure you have **Python 3.8 or newer** installed on your computer.  
> Check by opening your terminal/command prompt and typing: `python3 --version`  
> If you see a version number like `Python 3.10.x`, you are good to go!

---

### Step 1 — Download (Clone) the Project

**What does "clone" mean?**  
Cloning means copying all the project files from the internet (GitHub) to your computer.

Open your **Terminal** (Mac/Linux) or **Command Prompt** (Windows) and type:

```bash
git clone https://github.com/srivardhan-kondu/Student-s-Emotion-Recognition-using-Multimodality-and-Deep-Learning.git
```

Then navigate into the project folder:

```bash
cd "Student's Emotion Recognition using Multimodality and Deep Learning"
```

> 💡 **What is Git?** Git is a tool for downloading and managing code. If you don't have it,  
> download it from [https://git-scm.com/downloads](https://git-scm.com/downloads) and install it first.

---

### Step 2 — Create a Virtual Environment

**What is a virtual environment?**  
Think of it like a separate clean room for this project's software. It prevents this project's libraries from mixing with other Python projects on your computer.

**On Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

> ✅ **How do you know it worked?**  
> You will see `(venv)` appear at the start of your terminal line, like:  
> `(venv) your-computer:project $`

---

### Step 3 — Install Required Libraries

**What are libraries?**  
Libraries are pre-built tools that our code uses. For example, TensorFlow (for AI), OpenCV (for image processing), etc.

Run this command — it will automatically install everything:

```bash
pip install -r requirements.txt
```

> ⏳ This may take **5–15 minutes** depending on your internet speed.  
> You will see text scrolling on screen — that's normal, it's downloading and installing.

Then install some language processing data:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

### Step 4 — Download the Pre-trained AI Models ⚡

**Why is this step needed?**  
The AI models are the "brains" of the system. They have already been trained (which took hours of computing time), so you don't have to train them yourself. We just need to download the finished, ready-to-use models.

Run:
```bash
python download_models.py
```

> ⏳ This will download ~490 MB total from Google Drive. Time depends on your internet speed.  
> You will see a progress bar for each model file.

What gets downloaded into your `saved_models/` folder:

| File | What It Is | Size |
|---|---|---|
| `facial_emotion_model.h5` | Trained facial expression AI | ~43 MB |
| `speech_emotion_model.h5` | Trained voice tone AI | ~6 MB |
| `text_bert_model/` | Trained text understanding AI | ~438 MB |

> ✅ When it finishes you will see:  
> `✅ All models downloaded successfully!`  
> `🚀 You can now run: python run_dashboard.py`

## ⚠️ Important: If ZIP Models Do Not Extract Correctly

In some systems, the Google Drive ZIP file for the BERT model may not
extract properly using the automatic script.

If this happens, follow the steps below carefully.

------------------------------------------------------------------------

### Step 1 --- Download the Model Manually

If `download_models.py` fails to extract the BERT model correctly:

1.  Open the Google Drive link shown in the terminal manually.
2.  Download the `text_bert_model.zip` file.
3.  Extract it inside the `saved_models/` folder.

------------------------------------------------------------------------

### Step 2 --- Fix Folder Structure (Very Important)

After extraction, you might see this incorrect folder structure:

saved_models/ └── text_bert_model/ └── text_bert_model/ ├── config.json
├── pytorch_model.bin ├── tokenizer.json └── other model files

⚠️ This structure is incorrect and will cause the application to fail
when loading the text model.

You must modify it so it becomes:

saved_models/ └── text_bert_model/ ├── config.json ├── pytorch_model.bin
├── tokenizer.json └── other model files

------------------------------------------------------------------------

### How to Fix the Structure

1.  Open the inner `text_bert_model` folder.
2.  Move all files one level up into the outer `text_bert_model` folder.
3.  Delete the now-empty inner folder.

After correcting the structure, the dashboard will load the text model
successfully.


---

### Step 5 — Launch the Dashboard! 🎉

```bash
python run_dashboard.py
```

You will see this message in the terminal:
```
🚀 Starting Multimodal Emotion Recognition Dashboard...
📍 Dashboard will be available at: http://localhost:8501
```

Now open your web browser (Chrome, Firefox, etc.) and go to:

## 👉 [http://localhost:8501](http://localhost:8501)

The dashboard will open and you're ready to use it!

> 🛑 **To stop the server:** Press `Ctrl + C` in the terminal.

---

> <details>
> <summary>🔧 Advanced: Want to retrain the models yourself? (Optional — takes hours)</summary>
>
> You only need this if you want to train from scratch using your own data.
>
> **Download the datasets:**
>
> | Dataset | For | Size | Download |
> |---|---|---|---|
> | FER2013 | Face model | ~300 MB | [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) |
> | RAVDESS | Speech model | ~1.1 GB | [Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) |
> | GoEmotions | Text model | ~50 MB | [Kaggle](https://www.kaggle.com/datasets/debarshichanda/goemotions) |
>
> Place them in: `data/facial/fer2013/`, `data/speech/ravdess/`, `data/text/goemotions/`
>
> Then run:
> ```bash
> python src/facial_recognition/train.py   # ~30 min on GPU
> python src/speech_analysis/train.py      # ~20 min on GPU
> python src/text_analysis/train.py        # ~40 min on GPU
> ```
> </details>

---

## 🖥️ How to Use the Dashboard

When you open [http://localhost:8501](http://localhost:8501), you will see a dashboard with **4 tabs**:

### Tab 1 — 🎯 Multimodal (Most Powerful)

This tab uses **all three methods at once** for the most accurate result.

What you can do:
1. 📷 Upload a photo of the student's face (JPG or PNG)
2. 🎤 Upload an audio recording of the student speaking (WAV or MP3)
3. 📝 Type or paste a message the student wrote
4. Click **"Analyze Emotion"**
5. See the combined result with individual scores from each model

### Tab 2 — 👁️ Image Only

Upload **just a photo** to detect emotion from the face alone.  
Useful when you only have a webcam image.

### Tab 3 — 🎤 Audio Only

Upload **just an audio file** to detect emotion from voice alone.  
Useful for analyzing voice recordings or phone calls.

### Tab 4 — 📝 Text Only

Type or paste **any text** to detect emotion from words alone.  
Useful for analyzing student chat messages or written feedback.

---

### Sidebar Options (Left Panel)

| Option | What It Does |
|---|---|
| **Fusion Strategy** | Choose how the 3 models are combined (Calibrated is recommended) |
| **Modality Weights** | Adjust how much weight each model gets (Face / Speech / Text sliders) |
| **Prediction History** | See the last 5 predictions made |

---

## 📊 Model Performance

How accurate is each AI model?

| Model | Dataset Used | Number of Samples | Accuracy |
|---|---|---|---|
| 👁️ Facial CNN | FER2013 | 35,887 images | **57.7%** |
| 🎤 Speech BiLSTM | RAVDESS | 1,440 audio clips | **97.0%** |
| 📝 Text BERT | GoEmotions | 58,000 text samples | **65.9%** |

> **Why is facial accuracy lower?**  
> The FER2013 dataset (used for faces) is notoriously difficult — even humans only agree ~65% of the time on these images. Our model at 57.7% is typical for lightweight models. More complex systems achieve ~73% but require 10x more computing power.
>
> **Why is speech accuracy so high?**  
> The RAVDESS dataset uses professional actors with *very clear* emotional expressions. Real-world audio would be harder.

---

## 🛠️ Technology Used

| Category | Tool | What It's Used For |
|---|---|---|
| **AI Framework** | TensorFlow / Keras | Building and running the Face and Speech models |
| **AI Framework** | PyTorch | Building and running the BERT text model |
| **Language AI** | HuggingFace Transformers | BERT model library |
| **Face Detection** | OpenCV (Haar Cascade) | Detecting faces in photos |
| **Audio Processing** | librosa | Extracting audio features (MFCC) |
| **Text Processing** | NLTK | Cleaning and preparing text |
| **Web Dashboard** | Streamlit | Building the interactive browser interface |
| **Charts** | Plotly / Matplotlib | Showing emotion charts and graphs |
| **Data Science** | NumPy, Pandas, scikit-learn | Data manipulation and evaluation |

---

## 📦 Datasets Used for Training

### 👁️ FER2013 (For the Face Model)

| Property | Details |
|---|---|
| Total images | 35,887 face photos |
| Image size | 48 × 48 pixels, black & white |
| Source | Kaggle / Facial Expression Recognition Challenge |
| Emotions | 7 (we use 6) |

### 🎤 RAVDESS (For the Speech Model)

| Property | Details |
|---|---|
| Total clips | 1,440 audio recordings |
| Speakers | 24 professional actors (12 male, 12 female) |
| Audio format | WAV, 48,000 samples per second |
| Source | Ryerson University Audio-Visual Database |

### 📝 GoEmotions (For the Text Model)

| Property | Details |
|---|---|
| Total texts | ~58,000 Reddit comments |
| Original labels | 27 emotions (we map to 6) |
| Language | English |
| Source | Google Research |

---

## 📚 Additional Documentation

| Document | What's Inside |
|---|---|
| [QUICKSTART.md](QUICKSTART.md) | Even shorter setup guide |
| [DATASET_INSTRUCTIONS.md](DATASET_INSTRUCTIONS.md) | How to download datasets (for retraining) |
| [docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md) | Deep technical details for developers |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | Detailed user guide for the dashboard |
| [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) | How to deploy on a server |

---

## 📄 License

This project is developed as part of an academic research initiative at the university level.

## 🙏 Acknowledgments

- **Datasets:** FER2013, RAVDESS, GoEmotions
- **Pre-trained Models:** BERT (Google), EfficientNet (Google)
- **Libraries:** TensorFlow, PyTorch, HuggingFace, Streamlit, librosa, OpenCV
- **Research Papers:**
  - *"Real-time Convolutional Neural Networks for Emotion and Gender Classification"* — MiniXception architecture
  - *"BERT: Pre-training of Deep Bidirectional Transformers"* (Devlin et al., 2019)
  - *"Attention Is All You Need"* (Vaswani et al., 2017)
