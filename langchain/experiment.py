from langchain_ollama import OllamaLLM
from master_prompts import troll_prompt


def ask(llm, prompt, question=None, stops=[]):
    if question is None:
        question = input("Question: ")
    prompt = f"{prompt}\n\nQuestion: {question}\nResponse: "
    return llm.invoke(prompt, stop=stops)


llm = OllamaLLM(model="llama3:8b-text", base_url="http://127.0.0.1:11434")
stops = ["\nQuestion"]


response = ask(llm, troll_prompt, stops=stops)
print(response, end="")
