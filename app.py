import streamlit as st
from google import genai
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import ollama
import os
import time

st.set_page_config(
    page_title="AI-Powered Study Assistant",
    page_icon="📚",
    layout="wide"
)


if "history" not in st.session_state:
    st.session_state.history = []

if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False

if "rag_filename" not in st.session_state:
    st.session_state.rag_filename = ""


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


if not api_key:
    st.error(
        "Gemini API key not found. "
        "Please check your .env file."
    )
    st.stop()

client = genai.Client(api_key=api_key)


def generate_response(prompt):

    for attempt in range(3):

        try:

            return client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )

        except Exception:

            if attempt < 2:
                time.sleep(3)

            else:
                raise


def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text



def split_text_into_chunks(text, chunk_size=800, overlap=100):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


@st.cache_resource
def get_chroma_collection():

    chroma_client = chromadb.PersistentClient(
        path="chroma_db"
    )

    collection = chroma_client.get_or_create_collection(
        name="pdf_notes"
    )

    return collection


def index_pdf_into_chroma(pdf_text, filename):

    embedding_model = load_embedding_model()

    collection = get_chroma_collection()

    chunks = split_text_into_chunks(
        pdf_text,
        chunk_size=800,
        overlap=100
    )

    if not chunks:
        return 0

    embeddings = embedding_model.encode(
        chunks
    ).tolist()

    ids = []

    for i in range(len(chunks)):

        ids.append(
            f"{filename}_{i}"
        )

    collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

    return len(chunks)


def search_notes(question, number_of_results=4):

    embedding_model = load_embedding_model()

    collection = get_chroma_collection()

    question_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=number_of_results
    )

    if not results["documents"]:
        return []

    return results["documents"][0]



def ask_llama_from_notes(question, relevant_notes):

    context = "\n\n".join(
        relevant_notes
    )

    prompt = f"""
You are an AI-powered study assistant.

Answer the student's question using ONLY the study notes
provided below.

Do not use outside information.

If the answer cannot be found in the study notes,
clearly say:

"The answer is not available in the uploaded notes."

Study notes:

{context}

Student question:

{question}

Instructions:

- Give a simple and clear answer.
- Use college-student friendly language.
- Keep the answer exam-friendly.
- Explain important terms when necessary.
"""

    response = ollama.chat(
        model="llama3.2:latest",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]



st.title("📚 AI-Powered Study Assistant")

st.write(
    "Your personal AI assistant for learning "
    "and exam preparation."
)



st.sidebar.title("📖 Study Modes")

mode = st.sidebar.selectbox(
    "Choose a study mode:",
    [
        "💬 Ask a Question",
        "📝 Summarize Notes",
        "📄 Upload PDF Notes",
        "📚 Ask From My Notes",
        "❓ Generate Quiz",
        "🧠 Explain a Topic"
    ]
)


if mode == "💬 Ask a Question":

    st.header("💬 Ask a Question")

    question = st.text_area(
        "Enter your study question:",
        placeholder=(
            "Example: Explain the photoelectric "
            "effect in simple words."
        )
    )

    if st.button("🤖 Ask AI"):

        if question.strip():

            with st.spinner("Thinking..."):

                try:

                    response = generate_response(
                        question
                    )

                    st.subheader("💡 Answer")

                    st.write(
                        response.text
                    )

                    st.session_state.history.append(
                        {
                            "mode": "Ask a Question",
                            "question": question,
                            "answer": response.text
                        }
                    )

                except Exception:

                    st.error(
                        "⚠️ Gemini is temporarily busy. "
                        "Please try again after a few seconds."
                    )

        else:

            st.warning(
                "Please enter a question first."
            )



elif mode == "📝 Summarize Notes":

    st.header("📝 Summarize Notes")

    notes = st.text_area(
        "Paste your notes here:",
        height=250,
        placeholder=(
            "Paste your study notes here..."
        )
    )

    if st.button("📝 Summarize"):

        if notes.strip():

            with st.spinner(
                "Creating summary..."
            ):

                prompt = f"""
Summarize the following study notes.

Make the summary:

- Simple
- Clear
- Exam-friendly
- Easy to remember

Notes:

{notes}
"""

                try:

                    response = generate_response(
                        prompt
                    )

                    st.subheader(
                        "📌 Summary"
                    )

                    st.write(
                        response.text
                    )

                    st.session_state.history.append(
                        {
                            "mode": "Summarize Notes",
                            "question": "Notes Summary",
                            "answer": response.text
                        }
                    )

                except Exception:

                    st.error(
                        "⚠️ Gemini is temporarily busy. "
                        "Please try again after a few seconds."
                    )

        else:

            st.warning(
                "Please enter some notes first."
            )


elif mode == "📄 Upload PDF Notes":

    st.header("📄 Upload PDF Notes")

    uploaded_file = st.file_uploader(
        "Upload your study notes as a PDF:",
        type=["pdf"]
    )

    if uploaded_file is not None:

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

        if st.button("📖 Read PDF"):

            with st.spinner(
                "Reading PDF..."
            ):

                try:

                    pdf_text = extract_pdf_text(
                        uploaded_file
                    )

                    if pdf_text.strip():

                        st.subheader(
                            "📄 Extracted Notes"
                        )

                        st.text_area(
                            "PDF content:",
                            pdf_text,
                            height=300
                        )

                        st.session_state.history.append(
                            {
                                "mode": "Upload PDF Notes",
                                "question": uploaded_file.name,
                                "answer": (
                                    "PDF text extracted successfully."
                                )
                            }
                        )

                    else:

                        st.warning(
                            "No readable text was found "
                            "in this PDF."
                        )

                except Exception:

                    st.error(
                        "⚠️ Unable to read this PDF. "
                        "Please try another PDF file."
                    )



elif mode == "📚 Ask From My Notes":

    st.header("📚 Ask From My Notes")

    st.write(
        "Upload a PDF and ask questions based only "
        "on the content of your notes."
    )

    uploaded_file = st.file_uploader(
        "Upload your study PDF:",
        type=["pdf"],
        key="rag_pdf"
    )

    if uploaded_file is not None:

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

        if st.button(
            "📚 Process My Notes"
        ):

            with st.spinner(
                "Reading and processing your notes..."
            ):

                try:

                    pdf_text = extract_pdf_text(
                        uploaded_file
                    )

                    if not pdf_text.strip():

                        st.warning(
                            "No readable text was found "
                            "in this PDF."
                        )

                    else:

                        number_of_chunks = (
                            index_pdf_into_chroma(
                                pdf_text,
                                uploaded_file.name
                            )
                        )

                        st.session_state.rag_ready = True

                        st.session_state.rag_filename = (
                            uploaded_file.name
                        )

                        st.success(
                            f"✅ Notes processed successfully! "
                            f"{number_of_chunks} text chunks "
                            f"were stored."
                        )

                        st.session_state.history.append(
                            {
                                "mode": "Ask From My Notes",
                                "question": uploaded_file.name,
                                "answer": (
                                    "PDF processed and "
                                    "stored in the knowledge base."
                                )
                            }
                        )

                except Exception as e:

                    st.error(
                        "⚠️ Unable to process the PDF."
                    )

                    st.caption(
                        f"Error: {str(e)}"
                    )

    st.divider()

    if st.session_state.rag_ready:

        st.success(
            "🟢 Your notes are ready for questions."
        )

        st.write(
            f"Current notes: "
            f"**{st.session_state.rag_filename}**"
        )

        rag_question = st.text_area(
            "Ask a question about your notes:",
            placeholder=(
                "Example: What is the photoelectric effect?"
            ),
            key="rag_question"
        )

        if st.button(
            "🔍 Ask From My Notes"
        ):

            if rag_question.strip():

                with st.spinner(
                    "Searching your notes..."
                ):

                    try:

                        relevant_notes = search_notes(
                            rag_question,
                            number_of_results=4
                        )

                        if not relevant_notes:

                            st.warning(
                                "No relevant information "
                                "was found in your notes."
                            )

                        else:

                            with st.spinner(
                                "Generating answer..."
                            ):

                                answer = (
                                    ask_llama_from_notes(
                                        rag_question,
                                        relevant_notes
                                    )
                                )

                            st.subheader(
                                "💡 Answer From Your Notes"
                            )

                            st.write(
                                answer
                            )

                            with st.expander(
                                "📖 View Relevant Notes"
                            ):

                                for note in relevant_notes:

                                    st.write(
                                        f"- {note}"
                                    )

                            st.session_state.history.append(
                                {
                                    "mode": "Ask From My Notes",
                                    "question": rag_question,
                                    "answer": answer
                                }
                            )

                    except Exception as e:

                        st.error(
                            "⚠️ Unable to generate an answer "
                            "using the local AI model."
                        )

                        st.caption(
                            f"Error: {str(e)}"
                        )

            else:

                st.warning(
                    "Please enter a question first."
                )

    else:

        st.info(
            "👆 Upload a PDF and click "
            "**Process My Notes** to begin."
        )



elif mode == "❓ Generate Quiz":

    st.header("❓ Generate Quiz")

    topic = st.text_input(
        "Enter a topic:",
        placeholder=(
            "Example: Semiconductor Physics"
        )
    )

    if st.button("🎯 Generate Quiz"):

        if topic.strip():

            with st.spinner(
                "Generating quiz..."
            ):

                prompt = f"""
Create 5 multiple-choice questions for a student
studying the topic: {topic}.

For each question:

- Give four options
- Clearly mention the correct answer
- Keep the questions suitable for a college student
"""

                try:

                    response = generate_response(
                        prompt
                    )

                    st.subheader(
                        "📝 Quiz"
                    )

                    st.write(
                        response.text
                    )

                    st.session_state.history.append(
                        {
                            "mode": "Generate Quiz",
                            "question": topic,
                            "answer": response.text
                        }
                    )

                except Exception:

                    st.error(
                        "⚠️ Gemini is temporarily busy. "
                        "Please try again after a few seconds."
                    )

        else:

            st.warning(
                "Please enter a topic first."
            )


elif mode == "🧠 Explain a Topic":

    st.header("🧠 Explain a Topic")

    topic = st.text_input(
        "Enter a topic to explain:",
        placeholder=(
            "Example: How does a semiconductor diode work?"
        )
    )

    if st.button("🧠 Explain"):

        if topic.strip():

            with st.spinner(
                "Preparing explanation..."
            ):

                prompt = f"""
Explain the following topic to a college student
using simple and easy-to-understand language.

Topic:

{topic}

Include:

1. Simple definition
2. Main concept
3. Important points
4. Simple example
"""

                try:

                    response = generate_response(
                        prompt
                    )

                    st.subheader(
                        "📚 Explanation"
                    )

                    st.write(
                        response.text
                    )

                    st.session_state.history.append(
                        {
                            "mode": "Explain a Topic",
                            "question": topic,
                            "answer": response.text
                        }
                    )

                except Exception:

                    st.error(
                        "⚠️ Gemini is temporarily busy. "
                        "Please try again after a few seconds."
                    )

        else:

            st.warning(
                "Please enter a topic first."
            )


st.sidebar.divider()

st.sidebar.subheader(
    "📚 Study History"
)

if st.session_state.history:

    st.sidebar.write(
        f"You have "
        f"{len(st.session_state.history)} "
        f"study activities in this session."
    )

    if st.sidebar.button(
        "🗑️ Clear History"
    ):

        st.session_state.history = []

        st.rerun()

else:

    st.sidebar.info(
        "No study history yet."
    )