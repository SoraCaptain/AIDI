# AGENTS.md — AIDI Project Instructions

AI-driven image analysis platform using LangChain agents + vision-language models (Qwen). See [README.md](README.md) for server commands.

## Build & Run

- **Package manager:** `uv` (not pip). Use `uv run`, `uv add`, `uv sync`.
- **Python:** ≥ 3.14 (see `.python-version`).
- **Main backend (API Gateway):** `uv run uvicorn app.api.gateway:app --host 0.0.0.0 --port 8400` — see `run_backend.sh`
- **CV server:** `cd server && uv run uvicorn cv_server:app --host 0.0.0.0 --port 8200` — see `server/run_cv.sh`
- **MCP vision server:** `uv run python -m app.mcp_servers.vision_mcp_server`
- **GroundingDINO server:** see `server/run_gdion.sh`
- **Example API calls:** see `run_send.sh` for curl examples (create task, query status, human review, get report)

There are no lint/format/test scripts defined yet.

## Architecture

```
app/api/        → FastAPI gateway (task CRUD, human review, HITL resume)
app/services/   → Graph runtime orchestration (PersistentGraphRuntime with checkpoint persistence)
app/agents/     → LangChain agents (create_agent + ChatOpenAI + tools + system_prompt)
app/tools/      → @tool-decorated functions + raw API wrappers (cv_api.py, vlm_api.py)
app/graphs/     → LangGraph StateGraph workflows (parallel multi-agent is the main one)
app/memory/     → MemoryManager: session + persistent + vector + image-vector memory
app/mcp_servers/→ FastMCP tool servers exposing CV/VLM/detection as MCP tools
app/mcp_clients/→ MCP client adapter for LangChain agents
app/schemas/    → Pydantic models (vision_result.py, report.py)
app/observability/→ Langfuse tracing (CallbackHandler + trace metadata)
server/         → FastAPI CV server (blur via Laplacian), GroundingDINO server
```

External services: CV server (Laplacian blur), VLM server (Qwen3-VL-8B), vLLM backend (Qwen3), GroundingDINO (open-vocabulary detection).

**Main data flow:** API Gateway → PersistentGraphRuntime → parallel_multi_agent_vision_graph (Planner → parallel agent nodes → Aggregator → Critic → HITL interrupt if needed → final answer). See [app/graphs/parallel_multi_agent_vision_graph.py](app/graphs/parallel_multi_agent_vision_graph.py).

## Conventions

### Agent pattern
```python
class XxxAgent:
    def __init__(self):
        self.model = ChatOpenAI(model=..., api_key=..., base_url=..., temperature=0)
        self.agent = create_agent(model=self.model, tools=[...], system_prompt=self._build_system_prompt())
    def _build_system_prompt(self) -> str:
        return "角色定义 + 工具使用规则 + 输出要求"
    def chat(self, user_input: str) -> str:
        response = self.agent.invoke({"messages": [...]})
        return response["messages"][-1].content
```

### Graph pattern (LangGraph StateGraph)
```python
class XxxState(TypedDict, total=False):
    question: str
    image_path: Optional[str]
    # ... agent outputs, control fields

def build_xxx_graph(mcp_tools, memory_manager, checkpointer) -> StateGraph:
    graph = StateGraph(XxxState)
    # Add nodes (async functions), edges, conditional edges
    # Use interrupt() for HITL, Command(resume=...) for resume
    return graph.compile(checkpointer=checkpointer)
```
See [app/graphs/parallel_multi_agent_vision_graph.py](app/graphs/parallel_multi_agent_vision_graph.py) for the full pattern.

### MCP tool pattern
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("server-name")

@mcp.tool()
def tool_name(param: type) -> dict:
    """Docstring: when to use, inputs, returns. Used for MCP tool selection."""
    # call external service via requests
    return result
```
See [app/mcp_servers/vision_mcp_server.py](app/mcp_servers/vision_mcp_server.py) for the full pattern.

### Tool pattern (LangChain)
```python
@tool
def tool_name(param: type) -> return_type:
    """Docstring: when to use, inputs, returns. Used by the agent for tool selection."""
    # call external service via requests
    return result
```

### Other conventions
- **System prompts are in Chinese** (中文). Role → tool rules → output requirements.
- **`load_dotenv()`** is called at module level in every agent file.
- **Imports order:** stdlib → third-party → local (`app.*`).
- **Type hints** throughout: `str | None`, `dict`, `list[dict]`.
- **Private methods** use `_leading_underscore`.
- **Constants** (`CV_SERVER`, `VLM_SERVER`) are module-level `UPPER_CASE`.
- **LLM JSON parsing:** Use `safe_json_loads()` (from graph module) for extracting JSON from LLM text outputs — handles markdown code blocks and partial JSON.

## Pitfalls

### Qwen3 XML tool-call parsing
When using vLLM-served Qwen3 models with `tool_choice="auto"`, tool calls arrive as XML in `content` rather than structured `tool_calls`. See `app/agents/single_agent.py` → `QwenToolCallParsingMiddleware` for the fix. Without this middleware, tools are never executed.

### CV_SERVER IP placeholder
`app/tools/cv_tools.py` has `CV_SERVER = "http://<4090D_IP>:8200"` — must be updated to the actual IP before use.

### MCP server IP placeholders
`app/mcp_servers/vision_mcp_server.py` has hardcoded IPs (`10.6.88.x`) for CV_SERVER, VLM_SERVER, and GDINO_SERVER. These are overridable via env vars but the defaults are environment-specific. Set these in `.env` before deploying to a different machine.

### Image path handling
Agents pass `image_path` through conversation context. The `ShortMemory` / `SessionMemory` class injects a system message with the current image path so the LLM can resolve pronouns ("这张图", "它").

### Image encoding for remote servers
Tools that call remote CV/VLM servers must encode local images to base64 data URLs via `utils/utils.py` → `encode_image()`. The `_resolve_image()` helper in `vision_mcp_server.py` handles this automatically for MCP tools.

### Empty scaffolded modules
Many modules under `app/agents/`, `app/tools/`, `app/graphs/`, `app/schemas/`, `app/mcp_servers/` are empty `__init__.py`-only stubs. Check file contents before assuming functionality.

### AsyncSqliteSaver lifecycle
The `PersistentGraphRuntime` in `app/services/graph_runtime_persistent.py` manages an `AsyncSqliteSaver` context manager. It must be opened at FastAPI startup (`initialize()`) and closed at shutdown (`close()`). The checkpointer is passed to `build_parallel_multi_agent_vision_graph()` via `compile(checkpointer=...)`.

### Memory manager initialization
`MemoryManager` with vector memory enabled downloads sentence-transformers models on first use. Ensure the huggingface model cache is populated or models are pre-downloaded via `scripts/download_hf_model.py`.

## Key Files
- `app/agents/vision_agent.py` — main working agent (blur + VLM)
- `app/agents/single_agent.py` — reference for Qwen3 middleware pattern
- `app/memory/short_memory.py` — conversation memory with image tracking
- `server/cv_server.py` — FastAPI blur detection endpoint
- `app/api/gateway.py` — FastAPI entry point (task CRUD, HITL resume)
- `app/graphs/parallel_multi_agent_vision_graph.py` — main production graph (Planner → parallel agents → HITL)
- `app/services/graph_runtime_persistent.py` — runtime that wires together MCP tools, MemoryManager, and graph
- `app/mcp_servers/vision_mcp_server.py` — unified MCP tool server for CV/VLM/detection
- `app/mcp_clients/vision_mcp_client.py` — MCP client adapter for LangChain agents
- `app/memory/memory_manager.py` — unified memory (session + persistent + vector + image-vector)
