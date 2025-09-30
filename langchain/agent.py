from langchain_ollama import OllamaLLM

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, ChatResult, ChatGeneration
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from pydantic import PrivateAttr
from master_prompts import ai_agent_prompt as mp


def _format_messages(messages: list[BaseMessage], identity: str, master_prompt: str) -> str:
    print("Formatting messages")
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = "Human"
        elif isinstance(m, AIMessage):
            role = identity
        elif isinstance(m, SystemMessage):
            role = "System"
        else:
            role = m.type  # fallback, e.g. "tool"
        lines.append(f"{role}: {m.content}")
    return master_prompt + ("\n").join(lines) + "\n"


class OllamaWrapper(BaseChatModel):
    _llm: OllamaLLM = PrivateAttr()
    _master_prompt: str = PrivateAttr()

    def __init__(self, llm: OllamaLLM, master_prompt: str):
        super().__init__()
        self._llm = llm
        self._master_prompt = master_prompt

    def _generate(self, messages: list[BaseMessage], stop=None, **kwargs):
        prompt = _format_messages(messages, "Knight", self._master_prompt)
        print(f"prompt: {prompt}")
        output = self._llm.invoke(prompt, **kwargs)
        message = AIMessage(
            content=output)

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _llm_type(self) -> str:
        return "llama"


llm = OllamaLLM(model="llama3:8b-text",
                base_url="http://127.0.0.1:11434", stops=["END"])

memory = MemorySaver()
model = OllamaWrapper(llm, mp)
tools = []
agent_executor = create_react_agent(model, tools, checkpointer=memory)


config = {"configurable": {"thread_id": "abc123"}}

input_message = HumanMessage(content="Hi, I'm Bob and I live in SF.")

for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()
