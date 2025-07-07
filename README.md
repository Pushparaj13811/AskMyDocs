# AI Workshop: Chat with Your Documents

This project is a single-file Python FastAPI application designed for a 3-hour hands-on AI workshop. The goal is to teach the application side of AI — how to build useful, AI-powered tools using vector databases and Large Language Model (LLM) APIs.

It provides a fully working local app that demonstrates how to embed and search documents, interact with an AI using a context window, and build a useful agentic chat app with multi-turn memory.

![App Screenshot](https://user-images.githubusercontent.com/12345/placeholder.png) <!-- Replace with an actual screenshot -->

---

### 🔬 Teaching Focus

This workshop and codebase are designed to teach practical, real-world AI application development concepts:

-   **🧠 Agentic AI Principles**: Learn how to build applications that maintain conversational memory and engage in goal-oriented interactions.
-   **🧩 Vector Space Reasoning**: Understand the core workflow of chunking text, creating embeddings, and performing similarity searches to find relevant information.
-   **🛠️ ChromaDB for Vector Storage**: Use a simple, local, and fast vector database to store and query document chunks.
-   **🔗 Consuming AI APIs**: Integrate with the Google Gemini API to generate context-aware answers based on retrieved document chunks.
-   **💬 Interactive Applications**: Build a responsive web UI with FastAPI and Jinja2 that supports session-based chat history.

### ✨ Features

-   **📄 Document Upload**: A simple web interface to upload PDF and TXT files.
-   **⚙️ Automated Processing**: Uploaded documents are automatically processed: text is extracted, split into manageable chunks, and converted into vector embeddings.
-   **💾 Vector Storage**: Vectorized chunks are stored locally using ChromaDB for fast retrieval.
-   **💬 Multi-Document Chat**: Select one or more documents from the sidebar to create a combined context for your conversation.
-   **🧠 Conversational Memory**: The application remembers the history of your current session, allowing for follow-up questions.
-   **💻 Simple & Clean UI**: A modern, responsive interface built with FastAPI's Jinja2 template system (no complex JavaScript frameworks).

---

### 🛠️ Tech Stack

-   **Backend**: Python, FastAPI
-   **Frontend**: Jinja2, HTML, CSS, JavaScript (`fetch`)
-   **Vector Database**: ChromaDB
-   **Embedding Model**: `all-MiniLM-L6-v2` (via `sentence-transformers`)
-   **LLM API**: Google Gemini

---

### 🚀 Getting Started

Follow these instructions to set up and run the project locally.

#### 1. Prerequisites

-   Python 3.8 or higher
-   An API key for the Google Gemini API.

#### 2. Clone the Repository

```bash
git clone https://github.com/your-username/chat-with-your-notes.git
cd chat-with-your-notes
```

#### 3. Set Up a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\\venv\\Scripts\\activate
```

#### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 5. Configure Environment Variables

The application requires your Google Gemini API key.

1.  Make a copy of the example environment file. (Note: This project does not have a `.env.example`, so you will create the `.env` file directly).
2.  Create a new file named `.env` in the root of the project.
3.  Add your API key to it:

    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```

    Replace `"YOUR_GEMINI_API_KEY"` with your actual key.

---

### ▶️ How to Run the Application

Once the setup is complete, you can start the web server with:

```bash
uvicorn app:app --reload
```

Alternatively, you can run the Python script directly:

```bash
python app.py
```

The application will be available at `http://127.0.0.1:8000`.

### 🔧 How It Works

1.  **File Upload**: When you upload a file, it's saved to the `/uploads` directory.
2.  **Text Processing**:
    -   `app.py` detects the file type (PDF or TXT).
    -   Text is extracted using `PyMuPDF`.
    -   The text is split into smaller, overlapping chunks.
3.  **Embedding & Storage**:
    -   The `SentenceTransformer` model converts each text chunk into a numerical vector (embedding).
    -   Each chunk and its corresponding vector are stored in a [ChromaDB](https://www.trychroma.com/) collection. A separate collection is created for each document.
4.  **Chat Interaction**:
    -   When you ask a question, your query is also converted into an embedding.
    -   ChromaDB performs a similarity search within the collections of the selected documents to find the text chunks most relevant to your question.
    -   These relevant chunks (the context) are combined with your question and the recent chat history into a detailed prompt.
5.  **AI Response**:
    -   The complete prompt is sent to the Gemini API.
    -   The AI generates a response based *only* on the provided context and conversation history.
6.  **Session Management**:
    -   FastAPI uses a browser cookie to manage your session.
    -   The question-and-answer pairs for your session are stored in memory, allowing the AI to understand follow-up questions.

---

### 📂 Project Structure

```
.
├── .gitignore          # Specifies files to ignore for Git
├── app.py              # The single-file FastAPI application
├── requirements.txt    # Project dependencies
├── templates/
│   └── index.html      # The Jinja2 template for the UI
├── uploads/            # (Created automatically) Stores uploaded files
└── chroma_db/          # (Created automatically) Stores the vector database
```

---

### 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details. 