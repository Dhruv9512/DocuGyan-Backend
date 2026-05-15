import logging
from enum import Enum
from typing import Any

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

logger = logging.getLogger(__name__)

# Simplified Event Contracts
class WSEventType(str, Enum):
    MESSAGE = "message"               # For normal text updates
    STREAM_CHUNK = "stream_chunk"     # For LLM token streaming
    ERROR = "error"                   # For pipeline failures
    COMPLETED = "completed"           # For 100% finish

class Notifier:
    """
    Industry-standard WebSocket notifier. 
    Safe to instantiate inside Celery workers and LangGraph nodes.
    """
    NODE_KEYWORD_ALIASES = {
        "orchestrator": "orchestrator",
        "docupipeline orchestrator": "orchestrator",
        "evaluator agent": "orchestrator",
        "extractor agent": "extractor",
        "docuextractor agent": "extractor",
        "question extractor": "extractor",
        "question refining": "extractor",
        "academic agent": "academic",
        "financial agent": "financial",
        "audit agent": "audit",
        "vector ingestor": "vector_rag_ingest",
        "graph ingestor": "graph_rag_ingest",
        "vectorless ingestor": "vectorless_ingest",
    }

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.group_name = f'DocuAgent_{project_id}'
        self.channel_layer = get_channel_layer()
        self._last_current_node = "orchestrator"

    def _infer_current_node(self, message_text: str | None) -> str:
        """Infer current node alias from message text when caller does not provide one."""
        if message_text:
            lowered = message_text.lower()
            for keyword, alias in self.NODE_KEYWORD_ALIASES.items():
                if keyword in lowered:
                    return alias

        return self._last_current_node

    def _infer_status(
        self,
        event_type: WSEventType,
        message_text: str | None,
        explicit_status: str | None,
    ) -> str:
        """Normalize frontend-friendly status values for every payload."""
        if explicit_status:
            return explicit_status

        if event_type == WSEventType.ERROR:
            return "error"
        if event_type == WSEventType.COMPLETED:
            return "completed"
        if event_type == WSEventType.STREAM_CHUNK:
            return "processing"

        lowered = (message_text or "").lower()
        if any(token in lowered for token in ("failed", "error", "exception", "critical failure")):
            return "error"
        if any(token in lowered for token in ("complete", "completed", "success", "done", "ingested", "processed")):
            return "completed"

        return "processing"

    def _normalize_data(self, event_type: WSEventType, data: dict[str, Any]) -> dict[str, Any]:
        """Ensure every websocket message carries the fields expected by the frontend."""
        normalized = dict(data or {})

        message_text = normalized.get("message") or normalized.get("text")
        if message_text is None and event_type == WSEventType.COMPLETED:
            message_text = "Pipeline completed successfully."

        current_node = normalized.get("current_node") or self._infer_current_node(message_text)
        self._last_current_node = current_node

        status = self._infer_status(event_type, message_text, normalized.get("status"))

        if message_text is not None:
            normalized["message"] = message_text
            normalized["text"] = message_text

        normalized["current_node"] = current_node
        normalized["status"] = status

        return normalized

    def _broadcast(self, event_type: WSEventType, data: dict):
        """Internal method to handle the async_to_sync Redis dispatch."""
        try:
            normalized_data = self._normalize_data(event_type, data)
            payload = {
                "event_type": event_type.value,
                "project_id": self.project_id,
                "data": normalized_data
            }
            message = {
                "type": "send_message", # Maps to the consumer method
                "payload": payload
            }
            async_to_sync(self.channel_layer.group_send)(self.group_name, message)
        except Exception as e:
            logger.error(f"Redis Broadcast Failed for {self.project_id}: {str(e)}")

    # ==========================================
    # Simplified API Methods for LangGraph Nodes
    # ==========================================

    def send_message(self, text: str, current_node: str | None = None, status: str | None = None):
        """Use for normal informational messages."""
        self._broadcast(WSEventType.MESSAGE, {
            "text": text,
            "current_node": current_node,
            "status": status,
        })

    def send_stream_chunk(self, chunk: str, current_node: str | None = None):
        """Use for streaming LLM outputs token-by-token."""
        self._broadcast(WSEventType.STREAM_CHUNK, {
            "text": chunk,
            "current_node": current_node,
            "status": "processing",
        })

    def send_error(self, error_msg: str, current_node: str | None = None):
        """Use to notify the UI of a failure."""
        self._broadcast(WSEventType.ERROR, {
            "text": error_msg,
            "current_node": current_node,
            "status": "error",
        })

    def send_completed(
        self,
        final_result: dict | None = None,
        current_node: str | None = None,
        message: str | None = None,
    ):
        """Use when the entire graph finishes successfully."""
        self._broadcast(WSEventType.COMPLETED, {
            "result": final_result or {},
            "text": message or "Pipeline completed successfully.",
            "current_node": current_node,
            "status": "completed",
        })