from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3:8b-text", base_url="http://127.0.0.1:11434")

master_prompt = """You are meant to give short responses, limiting yourself to relatively short, concise answers, not exceeding three sentences.
                Do not continue after finishing the answer to the first question. Terminate output at the end of answer."""

usr_in = input("Question: ")

prompt = f"{master_prompt}\n\nQuestion:{usr_in}\nResponse:"
stops = ["\nQuestion"]

response = llm.invoke(prompt, stop=stops)
print(response)
