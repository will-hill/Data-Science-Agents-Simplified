# =====================================================
# === 1. imports (12 LOC) =============================
# =====================================================
import cudf.pandas
cudf.pandas.install()
import pandas as pd

from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
import operator


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
# === 5. Nodes ( 18 LOC ) =============================
# =====================================================

class AgentState(TypedDict):
    # Conversation history (reducer appends messages)
    messages: Annotated[List[BaseMessage], operator.add]
    # High-level plan from planner node
    plan: str
    # Code string or instructions from coder node
    code: str
    # Final numeric/text result (optional, for demo)
    result: str


def planner_agent(state: AgentState):
    """Decides what analysis to perform and writes a 'plan' into state."""
    system = SystemMessage(content="""You are a planning agent. 
    Analyze the user's question and describe what calculation needs to be done.
    Be specific about the operation (mean, sum, max, etc.) and column.""")
    
    messages = [system] + state["messages"]
    planner_reply = llm.invoke(messages)
    return {
        "messages": [planner_reply],
        "plan": planner_reply.content,
    }


def coder_agent(state: AgentState):
    """Generates and executes code/instructions (via tool calls)"""
    system = SystemMessage(content=f"""You are a coding agent.
    DataFrame 'df' has columns: {list(df.columns)}.
    Sample data: {df.head(3).to_string()}
    
    Write Python and pandas code to solve the task.""")
    messages = [system] + state["messages"]
    llm_with_tools = llm.bind_tools([execute_python])    
    coder_reply = llm_with_tools.invoke(messages)
    # Extract code for state tracking (optional)
    code_str = coder_reply.content or ""
    if hasattr(coder_reply, "tool_calls") and bool(coder_reply.tool_calls):
        code_str = coder_reply.tool_calls[0].get("args", {}).get("code", "")
    
    return {
        "messages": [coder_reply],
        "code": code_str
        }  


def should_continue(state: AgentState):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and bool(last_msg.tool_calls):
        return "tools"    

    return END


# =====================================================
# === 6. Build Graph ( 9 LOC ) ========================
# =====================================================
"""
ORCHESTRATION ON:
Conditional edges and a loop:
    START -> planner -> coder
                    | (if tool_calls) -> tools -> coder
                    | (else) -> END
This graph *decides* whether to call tools or stop, based on the state (tool calls),
and we maintain structured state (plan, coder, result).
"""
graph = StateGraph(AgentState)

graph.add_node("planner", planner_agent)
graph.add_node("coder", coder_agent)
graph.add_node("tools", ToolNode([execute_python]))
# Define flow: START -> Planner -> Coder -> Tools (if needed) -> END
graph.add_edge(START, "planner")
graph.add_edge("planner", "coder")
# Orchestrated decision:
# - if coder produced tool calls -> "tools", otherwise -> END
graph.add_conditional_edges("coder", should_continue, ["tools", END])
# Loop back from tools to coder so multiple tool calls are possible
graph.add_edge("tools", "coder")
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