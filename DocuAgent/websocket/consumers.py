import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer

logger = logging.getLogger(__name__)

class DocuProcessConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = self.scope.get('user')

        # 1. Authentication Check (relies on JWTAuthWebSocketMiddleware)
        if not self.user or not self.user.is_authenticated:
            logger.warning("Rejected unauthenticated WebSocket connection.")
            await self.close(code=4003) # 4003 is standard for Forbidden
            return

        self.project_id = self.scope['url_route']['kwargs']['project_id']
        
        # MUST exactly match the group_name in Notifier.__init__
        self.room_group_name = f'DocuAgent_{self.project_id}'

        # 2. Accept Connection and Join Group
        await self.channel_layer.group_add(
            self.room_group_name, 
            self.channel_name
        )
        await self.accept()
        logger.info(f"User {getattr(self.user, 'id', 'Unknown')} connected to {self.room_group_name}")

    async def disconnect(self, close_code):
        # Gracefully leave the channel layer group
        if hasattr(self, 'room_group_name'):
            await self.channel_layer.group_discard(
                self.room_group_name, 
                self.channel_name
            )
            logger.info(f"WebSocket disconnected from {self.room_group_name}")

    # -------------------------------------------------------------
    # Handle messages received from the Frontend (Client)
    # -------------------------------------------------------------
    async def receive(self, text_data=None, bytes_data=None):
        """
        Catches messages sent by the frontend via socket.send()
        """
        if text_data:
            try:
                # 1. Try to parse as JSON (Best Practice)
                data = json.loads(text_data)
                message_type = data.get('type')

                if message_type == 'ping':
                    # Respond with a 'pong' to keep the connection alive
                    await self.send(text_data=json.dumps({
                        'type': 'pong',
                        'message': 'Connection active'
                    }))
                else:
                    logger.info(f"Received unhandled message type from FE: {message_type}")

            except json.JSONDecodeError:
                # 2. Fallback: If the frontend just sends a raw string like "ping"
                if text_data.strip().lower() == 'ping':
                    await self.send(text_data=json.dumps({
                        'type': 'pong',
                        'message': 'Connection active'
                    }))
                else:
                    logger.warning(f"Received invalid JSON from client: {text_data}")

    # -------------------------------------------------------------
    # Method matches "type": "send_message" from Notifier
    # -------------------------------------------------------------
    async def send_message(self, event):
        """
        Receives messages from the Redis layer (sent by Celery/LangGraph) 
        and pushes them immediately to the frontend client.
        """
        payload = event['payload']

        # Forward the payload to the WebSocket client
        await self.send(text_data=json.dumps(payload))