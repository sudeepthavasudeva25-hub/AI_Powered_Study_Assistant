import streamlit as st
from google import genai
from dotenv import load_dotenv
import os
import time


# --------------------------------------------------
# STUDY HISTORY
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

# Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key not found. Please check your .env file.")
    st.stop()


# --------------------------------------------------
# CREATE GEMINI CLIENT
# --------------------------------------------------

client = genai.Client(api_key=api_key)


# --------------------------------------------------
# GEMINI RESPONSE FUNCTION WITH AUTOMATIC RETRY
# --------------------------------------------------

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


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI-Powered Study Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI-Powered Study Assistant")

st.write(
    "Your personal AI assistant for learning and exam preparation."
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("📖 Study Modes")

mode = st.sidebar.selectbox(
    "Choose a study mode:",
    [
        "💬 Ask a Question",
        "📝 Summarize Notes",
        "❓ Generate Quiz",
        "🧠 Explain a Topic"
    ]
)


# --------------------------------------------------
# ASK A QUESTION
# --------------------------------------------------

if mode == "💬 Ask a Question":

    st.header("💬 Ask a Question")

    question = st.text_area(
        "Enter your study question:",
        placeholder="Example: Explain the photoelectric effect in simple words."
    )

    if st.button("🤖 Ask AI"):

        if question.strip():

            with st.spinner("Thinking..."):

                try:

                    response = generate_response(question)

                    st.subheader("💡 Answer")
                    st.write(response.text)

                    # Save to history
                    st.session_state.history.append({
                        "mode": "Ask a Question",
                        "question": question,
                        "answer": response.text
                    })

                except Exception:

                    st.error(
                        "⚠️ Gemini is temporarily busy. "
                        "Please try again after a few seconds."
                    )

        else:

            st.warning("Please enter a question first.")


# --------------------------------------------------
# SUMMARIZE NOTES
# --------------------------------------------------

elif mode == "📝 Summarize Notes":

    st.header("📝 Summarize Notes")

    notes = st.text_area(
        "Paste your notes here:",
        height=250,
        placeholder="Paste your study notes here..."
    )

    if st.button("📝 Summarize"):

        if notes.strip():

            with st.spinner("Creating summary..."):

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

                    response = generate_response(prompt)

                    st.subheader("📌 Summary")
                    st.write(response.text)

                    # Save to history
                    st.session_state.history.append({
                        "mode": "Summarize Notes",
                        "question": "Notes Summary",
                        "answer": response.text
                    })

                except Exception:

                    st.error(
                        "⚠️ Gemini is temporarily busy. "
                        "Please try again after a few seconds."
                    )

        else:

            st.warning("Please enter some notes first.")


# --------------------------------------------------
# GENERATE QUIZ
# --------------------------------------------------

elif mode == "❓ Generate Quiz":

    st.header("❓ Generate Quiz")

    topic = st.text_input(
        "Enter a topic:",
        placeholder="Example: Semiconductor Physics"
    )

    if st.button("🎯 Generate Quiz"):

        if topic.strip():

            with st.spinner("Generating quiz..."):

                prompt = f"""
Create 5 multiple-choice questions for a student
studying the topic: {topic}.

For each question:
- Give four options
- Clearly mention the correct answer
- Keep the questions suitable for a college student
"""

                try:

                    response = generate_response(prompt)

                    st.subheader("📝 Quiz")
                    st.write(response.text)

                    # Save to history
                    st.session_state.history.append({
                        "mode": "Generate Quiz",
                        "question": topic,
                        "answer": response.text
                    })

                except Exception:

                    st.error(
                        "⚠️ Gemini is temporarily busy. "
                        "Please try again after a few seconds."
                    )

        else:

            st.warning("Please enter a topic first.")


# --------------------------------------------------
# EXPLAIN A TOPIC
# --------------------------------------------------

elif mode == "🧠 Explain a Topic":

    st.header("🧠 Explain a Topic")

    topic = st.text_input(
        "Enter a topic to explain:",
        placeholder="Example: How does a semiconductor diode work?"
    )

    if st.button("🧠 Explain"):

        if topic.strip():

            with st.spinner("Preparing explanation..."):

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

                    response = generate_response(prompt)

                    st.subheader("📚 Explanation")
                    st.write(response.text)

                    # Save to history
                    st.session_state.history.append({
                        "mode": "Explain a Topic",
                        "question": topic,
                        "answer": response.text
                    })

                except Exception:

                    st.error(
                        "⚠️ Gemini is temporarily busy. "
                        "Please try again after a few seconds."
                    )

        else:

            st.warning("Please enter a topic first.")


# --------------------------------------------------
# STUDY HISTORY
# --------------------------------------------------

st.sidebar.divider()

st.sidebar.subheader("📚 Study History")

if st.session_state.history:

    st.sidebar.write(
        f"You have {len(st.session_state.history)} "
        "study activities in this session."
    )

    if st.sidebar.button("🗑️ Clear History"):

        st.session_state.history = []

        st.rerun()

else:

    st.sidebar.info("No study history yet.")