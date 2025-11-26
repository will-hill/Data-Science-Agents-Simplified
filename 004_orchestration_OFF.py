# =====================================================
# === 1. imports (12 LOC) =============================
# =====================================================
import cudf.pandas
cudf.pandas.install()
import pandas as pd

from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

import os
from dotenv import load_dotenv

# =====================================================
# === 2. LLM setup (2 LOC) ============================
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
# === 3. Data ( 1 LOC ) ===============================
# =====================================================
# Toy, Sample DataFrame
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=10),
    'sales': [1_000_000, 120, 115, 140, 160, 155, 180, 190, 185, 200],
    'profit': [20, 25, 23, 30, 35, 33, 40, 42, 41, 45]
})

# =====================================================
# === 4. Tools ( 4 LOC ) ==============================
# =====================================================
# Define tool to execute Python code
@tool
def execute_python(code: str) -> str:
    """Execute Python code to analyze the DataFrame 'df'.
    Variables available: df (pandas DataFrame), pd (pandas module).
    Store your result in a variable called 'result' to return it.
    """
    try:
        local_vars = {"df": df, "pd": pd}
        exec(code, {"__builtins__": __builtins__, "pd": pd}, local_vars)
        return str(local_vars.get("result", "Code executed. Use 'result' variable."))
    except Exception as e:
        return f"Error: {str(e)}"


# =====================================================
# === 5. Nodes ( 9 LOC ) ==============================
# =====================================================
def planner_agent(state: MessagesState):
    """Decides what analysis to perform."""
    system = SystemMessage(content="""You are a planning agent. 
    Analyze the user's question and describe what calculation needs to be done.
    Be specific about the operation (mean, sum, max, etc.) and column.""")
    
    messages = [system] + state["messages"]
    planner_reply = llm.invoke(messages)
    return {
        "messages": [planner_reply]
    }


def coder_agent(state: MessagesState):
    """Generates and executes code/instructions (via tool calls)"""
    system = SystemMessage(content=f"""You are a coding agent.
    DataFrame 'df' has columns: {list(df.columns)}.
    Sample data: {df.head(3).to_string()}
    
    Write Python and pandas code to solve the task.""")
    messages = [system] + state["messages"]
    llm_with_tools = llm.bind_tools([execute_python])
    
    coder_reply = llm_with_tools.invoke(messages)
    return {"messages": [coder_reply]}


# =====================================================
# === 6. Build Graph ( 9 LOC ) ========================
# =====================================================
"""
ORCHESTRATION OFF:
A simple, static pipeline: START -> planner -> coder -> tools -> END
There is no decision point. We always run tools, even if the coder didn't actually request a tool call.
"""
graph = StateGraph(MessagesState)
graph.add_node("planner", planner_agent)
graph.add_node("coder", coder_agent)
graph.add_node("tools", ToolNode([execute_python]))

# Static linear flow (brittle)
graph.add_edge(START, "planner")
graph.add_edge("planner", "coder")

# ⚠️ BUG: Always routes to tools, even when LLM doesn't request one
graph.add_edge("coder", "tools")

graph.add_edge("tools", END)

app = graph.compile()

# =====================================================
# === 7. Local Run ====================================
# =====================================================
if __name__ == "__main__":
    prompt = "What are the average sales?"
    result = app.invoke({"messages": [HumanMessage(content=prompt)]})    
    print(result['messages'][-1].content)
    
    prompt = "Say, 'I love AI!'"
    result = app.invoke({"messages": [HumanMessage(content=prompt)]})    
    print(result['messages'][-1].content)
    
    prompt = "What's 2+2?"
    result = app.invoke({"messages": [HumanMessage(content=prompt)]})    
    print(result['messages'][-1].content)