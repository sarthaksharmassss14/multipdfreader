from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

def summarize_conversation(qa_log: list):
    """
    Summarize the user conversation and extract user intent.
    qa_log: List of dicts with keys: question, answer, email
    """

    # Format the conversation into a string
    formatted_conversation = ""
    for i, qa in enumerate(qa_log, 1):
        formatted_conversation += f"Q{i}: {qa['question']}\nA{i}: {qa['answer']}\n"

    # Define prompt
    prompt_template = """
Given the following conversation between a user and a PDF-based assistant, summarize the conversation and identify the user's intent.

Conversation:
{conversation}

Return a summary in 2-3 lines, and also clearly specify the user's main intent in one sentence.
Respond in this format:
Summary: <summary>
Intent: <intent>
"""
    prompt = PromptTemplate(
        input_variables=["conversation"],
        template=prompt_template
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    chain = LLMChain(llm=llm, prompt=prompt)

    result = chain.run(conversation=formatted_conversation)
    return result
