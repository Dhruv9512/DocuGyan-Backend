# DocuChat/consumers.py
import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer
from DocuChat.rag.hybrid_rag import HybridRAGChatbot

logger = logging.getLogger(__name__)

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = self.scope.get('user')
        
        if not self.user or not self.user.is_authenticated:
            logger.warning("Rejected unauthenticated Chat connection.")
            await self.close(code=4003)
            return

        # Extract both parameters from the URL route
        self.project_id = self.scope['url_route']['kwargs']['project_id']
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        
        # Scope the room strictly to this session
        self.room_group_name = f'chat_{self.project_id}_{self.session_id}'

        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()
        
        # Initialize the Hybrid RAG utility for this session
        self.chatbot = HybridRAGChatbot(project_id=self.project_id)
        logger.info(f"Chat connected for project: {self.project_id}, session: {self.session_id}")

    async def disconnect(self, close_code):
        if hasattr(self, 'room_group_name'):
            await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

    async def receive(self, text_data=None, bytes_data=None):
        if text_data:
            try:
                data = json.loads(text_data)
                user_message = data.get('message')
                chat_history = data.get('history', [])

                if not user_message:
                    return

                # Send an acknowledgment
                await self.send(text_data=json.dumps({"type": "status", "content": "Thinking..."}))

                # Call the chatbot utility and stream the response
                async for token in self.chatbot.astream_chat(user_message, chat_history):
                    await self.send(text_data=json.dumps({
                        "type": "stream_chunk",
                        "content": token
                    }))

                # Send completion signal
                await self.send(text_data=json.dumps({"type": "stream_complete"}))

            except json.JSONDecodeError:
                logger.error("Invalid JSON received in chat.")
            except Exception as e:
                logger.error(f"Chat error: {e}")
                await self.send(text_data=json.dumps({"type": "error", "content": "Failed to process message."}))