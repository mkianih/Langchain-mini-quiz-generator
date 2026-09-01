"""Mini Quiz Generator - Hugging Face Spaces entry point (Gradio UI).

Reuses the same chain-composition pipeline as mini_quiz_generator.py, but
exposes it through a web UI instead of the terminal. The Gemini API key is
read from the GOOGLE_API_KEY environment variable, which on Hugging Face
Spaces should be set as a Repository Secret (Settings -> Variables and
secrets) rather than committed to the repo. Visitors never see the key -
only this server-side process reads it.
"""

import os

import gradio as gr
from dotenv import load_dotenv

from mini_quiz_generator import build_chain, build_llm

load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    raise RuntimeError(
        "GOOGLE_API_KEY not found. On Hugging Face Spaces, set it under "
        "Settings -> Variables and secrets -> New secret. Locally, put it "
        "in a .env file."
    )

llm = build_llm()
chain = build_chain(llm)


def generate_quiz(topic):
    if not topic or not topic.strip():
        return "Enter a topic first.", ""
    response = chain.invoke({"topic": topic.strip()})
    return response["question"], response["answer"]


demo = gr.Interface(
    fn=generate_quiz,
    inputs=gr.Textbox(label="Topic", placeholder="e.g. Photosynthesis"),
    outputs=[
        gr.Textbox(label="Question (Chain 1)"),
        gr.Textbox(label="Answer & Explanation (Chain 2)", lines=6),
    ],
    title="Mini Quiz Generator",
    description=(
        "A two-step LangChain SequentialChain pipeline: a topic becomes a "
        "beginner-level question, then that question becomes a detailed "
        "answer with explanation."
    ),
)

if __name__ == "__main__":
    demo.launch()
