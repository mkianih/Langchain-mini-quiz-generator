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


LOGO = r"""░██                                                     ░██                   ░██
░██                                                     ░██
░██          ░██████   ░████████   ░████████  ░███████  ░████████   ░██████   ░██░████████
░██               ░██  ░██    ░██ ░██    ░██ ░██    ░██ ░██    ░██       ░██  ░██░██    ░██
░██          ░███████  ░██    ░██ ░██    ░██ ░██        ░██    ░██  ░███████  ░██░██    ░██
░██         ░██   ░██  ░██    ░██ ░██   ░███ ░██    ░██ ░██    ░██ ░██   ░██  ░██░██    ░██
░██████████  ░█████░██ ░██    ░██  ░█████░██  ░███████  ░██    ░██  ░█████░██ ░██░██    ░██
                                         ░██
                                   ░███████

  ░██████              ░██                ░██████                                                          ░██
 ░██   ░██                               ░██   ░██                                                         ░██
░██     ░██ ░██    ░██ ░██░█████████    ░██         ░███████  ░████████   ░███████  ░██░████  ░██████   ░████████  ░███████  ░██░████
░██     ░██ ░██    ░██ ░██     ░███     ░██  █████ ░██    ░██ ░██    ░██ ░██    ░██ ░███           ░██     ░██    ░██    ░██ ░███
░██     ░██ ░██    ░██ ░██   ░███       ░██     ██ ░█████████ ░██    ░██ ░█████████ ░██       ░███████     ░██    ░██    ░██ ░██
 ░██   ░██  ░██   ░███ ░██ ░███          ░██  ░███ ░██        ░██    ░██ ░██        ░██      ░██   ░██     ░██    ░██    ░██ ░██
  ░██████    ░█████░██ ░██░█████████      ░█████░█  ░███████  ░██    ░██  ░███████  ░██       ░█████░██     ░████  ░███████  ░██
       ░██
        ░██"""

# The ASCII logo is 133 chars wide. Rather than let it overflow and force
# horizontal scrolling, the font-size is expressed in container-query width
# units so the art always scales down to exactly fit: a monospace glyph is
# ~0.6em wide, so 133 chars span 133 * 0.6 = ~80em, meaning 100cqw / 80 =
# 1.25cqw per character fits the container exactly. 1.2cqw leaves a margin.
# Below 700px the art would be unreadably small, so a plain text title is
# shown instead.
CSS = """
#logo-wrap {
  container-type: inline-size;
  text-align: center;
  overflow: hidden;
}
#logo-wrap pre {
  display: inline-block;
  text-align: left;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 11px;          /* fallback if cqw is unsupported */
  font-size: min(1.2cqw, 14px);
  line-height: 1.05;
  color: #4f46e5;
  white-space: pre;
  margin: 0;
  border: none;
  background: none;
  padding: 0;
}
#logo-fallback { display: none; text-align: center; }
#logo-fallback h1 { font-size: 32px; font-weight: 700; color: #4f46e5; margin: 0; }
@media (max-width: 700px) {
  #logo-wrap { display: none; }
  #logo-fallback { display: block; }
}
.gradio-container { max-width: 1000px !important; margin: 0 auto !important; }
"""

with gr.Blocks(title="Langchain Quiz Generator") as demo:
    gr.HTML(f"<div id='logo-wrap'><pre>{LOGO}</pre></div>")
    gr.HTML("<div id='logo-fallback'><h1>Langchain Quiz Generator</h1></div>")
    gr.Markdown(
        "A two-step LangChain pipeline: a topic becomes a beginner-level "
        "question, then that question becomes a detailed answer with "
        "explanation."
    )

    topic_box = gr.Textbox(label="Topic", placeholder="e.g. Photosynthesis")
    generate_btn = gr.Button("Generate Quiz", variant="primary")
    question_box = gr.Textbox(label="Question (Chain 1)", lines=3)
    answer_box = gr.Textbox(label="Answer & Explanation (Chain 2)", lines=8)

    generate_btn.click(
        generate_quiz, inputs=topic_box, outputs=[question_box, answer_box]
    )
    topic_box.submit(
        generate_quiz, inputs=topic_box, outputs=[question_box, answer_box]
    )

# Gradio otherwise follows the viewer's OS dark-mode preference, which Gradio
# itself only overrides via the ?__theme=light query parameter. This runs in
# <head>, before the app renders, so the redirect happens without a flash of
# the dark theme.
FORCE_LIGHT_HEAD = """
<script>
  (function () {
    var params = new URLSearchParams(window.location.search);
    if (params.get('__theme') !== 'light') {
      var url = new URL(window.location.href);
      url.searchParams.set('__theme', 'light');
      window.location.replace(url.href);
    }
  })();
</script>
"""

if __name__ == "__main__":
    demo.launch(css=CSS, head=FORCE_LIGHT_HEAD, theme=gr.themes.Soft())
