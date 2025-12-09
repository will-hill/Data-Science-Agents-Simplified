# MCP & LangGraph
### 30 LOC

<br/>

### 1. Grab Files

 - [ ]  ```git clone https://github.com/will-hill/Data-Science-Agents-Simplified.git```
 - [ ] ```cd Data-Science-Agents-Simplified```

<br/>

### 2. Install Dependencies

 - [ ] ```uv add --upgrade mcp langchain-mcp-adapters langchain-openai sglang langgraph jupyterlab jupyterlab-nvdashboard yfinance pandas```

<br/>

### 3. Jupyter Lab
- [ ] ```uv run jupyter lab```

<br/>

### 4. MPC Server
 - [ ] ```uv run python 006_MCP_Server.py```

<br/>

### 5. LLM Server - SGLang w/ Qwen
 - [ ] ```uv run python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct --tool-call-parser qwen```

<br/>


### 6. LangGraph App w/ MCP Client
- [ ] Run 006_MCP_LangGraph.ipynb

<br/>

### 7. Stock Data Example
- [ ] Run 006_MCP_LangGraph_Stocks.ipynb
##### This example is stock data downloaded and visualized with yfinance and plotly

<br/>
