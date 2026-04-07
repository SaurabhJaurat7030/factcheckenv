# Knowledge QA Environment (OpenEnv)

This project serves as a lightweight benchmark for evaluating grounded QA and hallucination control in LLM agents.

## 📌 Overview

This project implements an OpenEnv-compliant environment for evaluating grounded question answering (QA) and fact verification using document-based reasoning.

It simulates real-world workflows where agents must:

* Retrieve relevant information from documents
* Generate accurate answers
* Cite correct sources
* Avoid hallucinations

---

## 🧠 Motivation

Large Language Models (LLMs) often generate incorrect or unverified information.
This environment benchmarks an agent’s ability to:

* Perform **knowledge retrieval**
* Provide **factually correct answers**
* Ensure **grounded responses with evidence**
* Handle **conflicting or outdated information**

---

## 🧱 Environment Design

### 🔍 Observation

The agent receives:

* A list of documents (id + text)
* A question
* Interaction history

### ⚡ Action

The agent must return:

* `answer` (string)
* `source` (document ID)

### 🎯 Reward

The reward is computed based on:

* Answer correctness (+0.5)
* Source correctness (+0.3)
* Grounding in document (+0.2)
* Hallucination penalty (-0.3)

---

## 🎯 Tasks
> All tasks use deterministic grading to ensure reproducibility and consistent benchmarking.
### 🟢 Easy — Retrieval

* Identify the correct document
* No answer generation required

### 🟡 Medium — Question Answering

* Generate correct answer using documents

### 🔴 Hard — QA + Fact Verification

* Answer correctly
* Cite correct source
* Handle conflicting information

---

## 📊 Dataset

The dataset contains structured tasks with:

* Documents
* Questions
* Ground truth answers
* Correct sources

It also includes advanced multi-document reasoning samples for future extension.

---

## ⚖️ Reward Function

| Component       | Score |
| --------------- | ----- |
| Correct Answer  | +0.5  |
| Correct Source  | +0.3  |
| Grounded Answer | +0.2  |
| Hallucination   | -0.3  |

Final score is normalized between **0.0 and 1.0**

---

## 🤖 Baseline Results

EASY: 1.0
MEDIUM: 1.0
HARD: 0.2

AVERAGE: 0.73

> The lower performance on hard tasks demonstrates real-world challenges in multi-document reasoning, where models must ignore outdated information and select the most relevant evidence.

---

## ⚙️ Setup

### 1. Clone repository

```bash
git clone https://gitlab.com/SaurabhJaurat7030/factcheckenv.git
cd factcheckenv
```

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate environment

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Demo (No API Required)

```bash
python main.py
```

---

## 🤖 Run Baseline (Requires OpenAI API)

### Set API Key

**Windows:**

```bash
set OPENAI_API_KEY=your_api_key
```

**Mac/Linux:**

```bash
export OPENAI_API_KEY=your_api_key
```

### Run:

```bash
python baseline.py
```

---

## 🐳 Docker Usage

### Build:

```bash
docker build -t qa-env .
```

### Run:

```bash
docker run qa-env
```

---

## 📦 OpenEnv Compliance

This environment fully implements OpenEnv specifications:

* Typed **Observation, Action, Reward** models using Pydantic
* Supports:

  * `reset()`
  * `step(action)`
  * `state()`
* Includes `openenv.yaml` metadata
* Deterministic task graders
* Multi-task evaluation (easy → hard)

---
## 🧠 Design Philosophy

- Simple but scalable environment
- Deterministic and reproducible evaluation
- Focus on real-world LLM failure cases (hallucination, outdated knowledge)
---

## 🚀 Future Improvements

* Multi-document reasoning (advanced tasks)
* Better hallucination detection
* Integration with Hugging Face datasets
* Leaderboard for model comparison

---

## 🔐 Security Note

The project uses environment variables for API keys.
No API keys are stored in the repository.

---

This environment can be extended into a full benchmark suite for evaluating LLM reliability in real-world applications.

## 👨‍💻 Author

Saurabh Jaurat

---
