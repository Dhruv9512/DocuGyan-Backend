# DocuAgent/utils/notifier.py
import logging
from enum import Enum
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
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.group_name = f'DocuAgent_{project_id}'
        self.channel_layer = get_channel_layer()

    def _broadcast(self, event_type: WSEventType, data: dict):
        """Internal method to handle the async_to_sync Redis dispatch."""
        try:
            payload = {
                "event_type": event_type.value,
                "project_id": self.project_id,
                "data": data
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

    def send_message(self, text: str):
        """Use for normal informational messages."""
        self._broadcast(WSEventType.MESSAGE, {
            "text": text
        })

    def send_stream_chunk(self, chunk: str):
        """Use for streaming LLM outputs token-by-token."""
        self._broadcast(WSEventType.STREAM_CHUNK, {
            "text": chunk
        })

    def send_error(self, error_msg: str):
        """Use to notify the UI of a failure."""
        self._broadcast(WSEventType.ERROR, {
            "text": error_msg
        })

    def send_completed(self, final_result: dict = None):
        """Use when the entire graph finishes successfully."""
        self._broadcast(WSEventType.COMPLETED, {
            "result": final_result or {}
        })