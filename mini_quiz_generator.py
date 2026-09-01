"""Mini Quiz Generator - LangChain Chain Composition.

Generates a beginner-level question and a detailed answer for any topic
using two chained LLM calls (Google Gemini), composed with LangChain's
LCEL runnable-pipe syntax (the current replacement for the retired
LLMChain / SequentialChain classes).
"""

# LangChain - Google Gemini chat model
from langchain_google_genai import ChatGoogleGenerativeAI

# LangChain - prompt template, output parser, and runnable composition
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Standard library - environment variable access
import os

# Secret management for a local environment - loads variables from a .env file
from dotenv import load_dotenv


def load_api_key():
    # Load the Google Gemini API key from a local .env file and expose it as
    # the GOOGLE_API_KEY environment variable that langchain-google-genai
    load_dotenv()

    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY not found. Add it to a .env file in this directory "
            "(GOOGLE_API_KEY=your-key-here) or export it as an environment variable."
        )


def build_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.7)


def build_chain(llm):
    # Chain 1 prompt - Question generation.
    # Takes a topic and produces a single beginner-level question.
    question_prompt = PromptTemplate(
        input_variables=["topic"],
        template=(
            "You are an educational content creator. Generate exactly one "
            "beginner-level quiz question about the following topic: {topic}\n\n"
            "Requirements:\n"
            "- The question must be appropriate for someone new to the topic.\n"
            "- Output only the question text, with no answer, numbering, or "
            "extra commentary."
        )
    )

    # Chain 2 prompt - Answer generation.
    # Takes the question produced by Chain 1 and produces a clear answer plus
    # a short explanation, so the final quiz item is fully self-contained.
    answer_prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            "You are an educational content creator. Answer the following "
            "beginner-level quiz question: {question}\n\n"
            "Requirements:\n"
            "- Give a clear, direct answer first.\n"
            "- Follow it with a short explanation (2-3 sentences) that helps a "
            "beginner understand why the answer is correct."
        )
    )

    # Chain 1: Question Generation Chain. `prompt | llm | StrOutputParser()`
    # is LCEL's chain-composition syntax - the successor to LLMChain.
    question_chain = question_prompt | llm | StrOutputParser()

    # Chain 2: Answer Generation Chain.
    answer_chain = answer_prompt | llm | StrOutputParser()

    # Connect both chains into a single pipeline. Each RunnablePassthrough.assign
    # step keeps every existing key in the running dict and adds one more, so
    # "question" (Chain 1's output) is already present in the dict Chain 2
    # receives - the same automatic data routing SequentialChain used to do.
    return (
        RunnablePassthrough.assign(question=question_chain)
        | RunnablePassthrough.assign(answer=answer_chain)
    )


def print_quiz(topic, response):
    print("\n" + "=" * 70)
    print(" GENERATED QUIZ")
    print("=" * 70)
    print(f"\nTopic: {topic}")
    print("\nQuestion:")
    print(response["question"])
    print("\nAnswer & Explanation:")
    print(response["answer"])
    print("\n" + "=" * 70)


def main():
    load_api_key()
    llm = build_llm()
    chain = build_chain(llm)

    topic = input("Enter a topic: ")

    # .invoke() takes a dict whose keys match the chain's input_variables
    # (just "topic" here) and returns a dict containing every
    # output_variable.
    response = chain.invoke({"topic": topic})

    print_quiz(topic, response)


if __name__ == "__main__":
    main()
