# =====================================================
# === 1. imports ======================================
# =====================================================
import cudf.pandas
cudf.pandas.install()
import pandas as pd

from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import SystemMessage

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
import operator


# =====================================================
# === 2. LLM setup ====================================
# =====================================================
# Load environment variables from .env file
load_dotenv(override=True)

llm = ChatOpenAI(
    model="qwen/qwen3-coder-480b-a35b-instruct",
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1",
    temperature=0.1,
    max_tokens=500,
)

# =====================================================
# === 3. Data =========================================
# =====================================================
# Toy, Sample DataFrame
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=10),
    'sales': [1_000_000, 120, 115, 140, 160, 155, 180, 190, 185, 200],
    'profit': [20, 25, 23, 30, 35, 33, 40, 42, 41, 45]
})


# =====================================================
# === 5. App ==========================================
# =====================================================

graph = StateGraph(MessagesState)
graph.add_edge(START,END)
app = graph.compile()

# =====================================================
# === 7. Local Run ====================================
# =====================================================
if __name__ == "__main__":
    prompt = "What are the average sales?"
    result = app.invoke({"messages": [HumanMessage(content=prompt)]})    
    print(result['messages'][-1].content)
    
    prompt = "Repeat the last question back to me."
    result = app.invoke({"messages": [HumanMessage(content=prompt)]})    
    print(result['messages'][-1].content)