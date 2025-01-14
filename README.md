# MITRA - Model in the Rear Approach to SafeGuarding LLMs

<img width="1680" alt="Screenshot 2025-01-14 at 3 11 13 AM" src="https://github.com/user-attachments/assets/bc08f488-98c4-4053-8186-7375a58c258e" />

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Demo](#demo)
- [Technologies Used](#technologies-used)
  - [Frontend](#frontend)
  - [Backend](#backend)
  - [Others](#others)
- [Installation](#installation)
  - [Clone the Repository](#clone-the-repository)
  - [Navigate to the Project Directory](#navigate-to-the-project-directory)
  - [Create a Virtual Environment](#create-a-virtual-environment)
  - [Activate the Virtual Environment](#activate-the-virtual-environment)
  - [Install Dependencies](#install-dependencies)
  - [Download Qwen2.5:0.5 from Ollama](#Download-Qwen2.5:0.5-from-Ollama)
- [Usage](#usage)
  - [Activate the Virtual Environment](#activate-the-virtual-environment)
  - [Run the Flask Application](#run-the-flask-application)
- [Project Structure](#project-structure)

## Introduction

**MITRA** is a novel LLM Guardrail method which uses classification LLMs to protect the main LLM from jailbreaks and other harmful prompts. 

## Publication

A detailed publication on **MITRA - Model in the Rear Approach to Safeguarding LLMs** is **incoming soon**. It will cover the methodology, architecture, and results in-depth, providing insights into the novel techniques used to safeguard large language models (LLMs). Stay tuned!

## Features

- **Dynamic Chat Interface:** Engage in real-time conversations with MITRA through a user-friendly chat interface.
- **Safety Checks:** Incorporates multiple safety mechanisms, including toxicity detection and jailbreak prevention, to ensure responsible AI interactions.
- **Auto-Resizing Textarea:** Enhances user experience by automatically adjusting the input field based on the message length.
- **Responsive Design:** Optimized for various devices, ensuring seamless usability on desktops, tablets, and mobile phones.
- **Confetti Feedback:** Celebratory visual effects upon successful AI responses to enhance user engagement.
- **Example Prompts Animation:** Rotating example prompts guide users on how to effectively interact with MITRA.
- **Rate Limiting:** Protects the backend from abuse by limiting the number of requests a user can make within specific timeframes.

## Demo

https://github.com/user-attachments/assets/5f99db8f-fb78-4c3f-b0a6-805cb816e795

## Technologies Used

## Models Used

### 1. Toxicity Detection
- **Model**: `s-nlp/roberta_toxicity_classifier`
- **Details**: This is a RoBERTa-based model fine-tuned for toxicity detection. It identifies harmful or offensive content in text prompts with high accuracy.
- **Reference**: [Hugging Face Model Card](https://huggingface.co/s-nlp/roberta_toxicity_classifier)

### 2. Jailbreak Detection
- **Model**: `madhurjindal/Jailbreak-Detector-Large`
- **Details**: A large language model trained to detect jailbreak attempts or adversarial prompts aimed at bypassing safety protocols.
- **Reference**: [Hugging Face Model Card](https://huggingface.co/madhurjindal/Jailbreak-Detector-Large)

### 3. Embeddings
- **Model**: `all-MiniLM-L6-v2`
- **Details**: A lightweight SentenceTransformer model optimized for generating semantic embeddings. It balances speed and accuracy, making it suitable for real-time applications.
- **Reference**: [Sentence Transformers Documentation](https://www.sbert.net/docs/pretrained_models.html)

### 4. Language Model
- **Model**: `Qwen2.5:0.5b`
- **Details**: A fast and efficient large language model with 0.5 billion parameters, integrated via Langchain Ollama. This model handles safe and contextual AI responses.
- **Reference**: [Ollama Documentation](https://ollama.ai/)

### Backend
- **Python 3.8+**: Programming language for backend logic.
- **Flask**: Web framework for handling HTTP requests and serving templates.
- **Hugging Face Transformers**:
  - **Toxicity Model**: `s-nlp/roberta_toxicity_classifier` (RoBERTa-based classifier fine-tuned for toxicity detection).
  - **Jailbreak Detection Model**: `madhurjindal/Jailbreak-Detector-Large` (Large model trained to identify jailbreak attempts in prompts).
- **SentenceTransformers**:
  - **Embeddings Model**: `all-MiniLM-L6-v2` (Lightweight model optimized for generating semantic embeddings with fast inference).
- **Langchain Ollama**:
  - **LLM**: `Qwen2.5:0.5b` (Large language model with 0.5 billion parameters, optimized for speed and accuracy in inference).
- **PyTorch**: Deep learning library powering Hugging Face and SentenceTransformers models.
- **Asyncio**: Asynchronous programming for handling multiple concurrent safety checks.
- **Flask-Limiter**: Rate limiting to prevent abuse by restricting the number of API requests (if implemented).

### Frontend
- **HTML5, CSS3, JavaScript (ES6)**: For the dynamic chat interface.
- **Font Awesome**: For icons used in the frontend.
- **Google Fonts**: For enhanced typography.
- **Canvas-Confetti**: For celebratory visual effects upon successful AI responses.

### Others
- **Git & GitHub**: For version control and project collaboration.
- **Canvas-Confetti**: For confetti animations and engaging user feedback.

## Installation

Follow these steps to set up the project locally on your machine.

### Prerequisites

- **Python 3.8+** installed on your machine. You can download it from [here](https://www.python.org/downloads/).
- **Git** installed on your machine. Download from [here](https://git-scm.com/downloads).
- **Virtual Environment (Recommended):** It's good practice to use a virtual environment to manage dependencies.

### Steps

1. **Clone the Repository**

    ```bash
    git clone https://github.com/AbhayBhandarkar/MITRA.git
    ```

2. **Navigate to the Project Directory**

    ```bash
    cd MITRA
    ```

3. **Create a Virtual Environment**

    ```bash
    python3 -m venv venv
    ```

4. **Activate the Virtual Environment**

    - **On macOS and Linux:**

      ```bash
      source venv/bin/activate
      ```

    - **On Windows:**

      ```bash
      venv\Scripts\activate
      ```

5. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

6. **Download Qwen2.5:0.5 from Ollama**

    ```bash
    ollama serve
    ollama pull qwen2.5:0.5b
    ```
## Usage

### Activate the Virtual Environment

Ensure that your virtual environment is activated.

  ```bash
  source venv/bin/activate  # On macOS and Linux
  venv\Scripts\activate     # On Windows
  ```

### Run the Flask Application

  ```bash
  python app.py
  ```
### The application will start running on http://0.0.0.0:5000/ by default.

### Project Structure 

  ```bash
MITRA/
├── app.py                 # Main Flask application
├── pipeline.py            # Core pipeline for safety checks and LLM interactions
├── requirements.txt       # Python dependencies
├── static/                # Static files
│   ├── script.js          # Frontend JavaScript logic
│   └── styles.css         # Styling for the web interface
├── templates/             # HTML templates for the Flask app
│   └── index.html         # Main page
├── logs/                  # Log files
│   └── app.log            # Backend log output
├── .gitignore             # Files and directories to ignore in Git
└── README.md              # Project documentation

```

## License

This project is licensed under the **Apache License 2.0**.  
You can view the full license details in the `LICENSE` file or at the following link: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
