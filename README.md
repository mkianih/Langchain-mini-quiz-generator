---
title: Mini Quiz Generator
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Mini Quiz Generator

A two-step LangChain pipeline (Google Gemini) that turns a topic into a
beginner-level question, then turns that question into a detailed answer
with explanation. Chains are composed with LCEL (`prompt | llm`, chained via
`RunnablePassthrough.assign`) - the current LangChain 1.x way of doing what
`LLMChain`/`SequentialChain` used to do before they were removed.

- Chain 1: `topic -> question`
- Chain 2: `question -> answer`

## Running locally

Requires Python 3.10+.

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your GOOGLE_API_KEY
python3 app.py
```

## Deploying

The Gemini API key must be set as a **Repository Secret** named
`GOOGLE_API_KEY` in the Space's Settings -> Variables and secrets. It is
read server-side only (see `app.py`) and is never exposed to visitors.
