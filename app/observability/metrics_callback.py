from langchain_core.callbacks import BaseCallbackHandler

from app.observability.metrics import record_tool_call


class MetricsCallbackHandler(BaseCallbackHandler):
    def __init__(self, mode: str = "native"):
        self._mode = mode

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        tool_name = serialized.get("name", "unknown")
        record_tool_call(tool_name, self._mode)
