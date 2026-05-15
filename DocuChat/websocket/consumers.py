import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async

from DocuChat.models import ChatSession, ChatMessage
from DocuAgent.models import DocuProcess
from DocuChat.tasks import call_hybrid_rag_chatbot, generate_session_title  # ✅ new task


# --- Database Helpers ---

@database_sync_to_async
def get_or_create_chat_session(session_id, project_id, user):
    project = DocuProcess.objects.get(project_id=project_id)
   
    session = ChatSession.objects.get_or_create(session_id=session_id, project=project, user=user)[0]
    return session, session.session_id

@database_sync_to_async
def get_message_count(session) -> int:
    chat = ChatMessage.objects.filter(session=session)
    return chat.count()


# --- Consumer ---

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.user = self.scope.get('user')

        if not self.user or not self.user.is_authenticated:
            await self.close(code=4003)
            return

        self.project_id = self.scope['url_route']['kwargs']['project_id']
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        self.room_group_name = f'chat_{self.project_id}'

        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        if hasattr(self, 'room_group_name'):
            await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

    async def receive(self, text_data=None, bytes_data=None):
        if not text_data:
            return
        try:
            data = json.loads(text_data)
            message_type = data.get('type')

            if message_type == 'ping':
                await self.send(text_data=json.dumps({'type': 'pong', 'message': 'Connection active'}))

            elif message_type == 'chat':
                user_query = data.get('query')
                sources = data.get('sources', [])

                if not user_query:
                    return

                # 1. Lazy load session
                self.db_session, self.session_id = await get_or_create_chat_session(
                    self.session_id, self.project_id, self.user
                )

                # 2. Check if first message — fire LLM title generation task
                message_count = await get_message_count(self.db_session)
                if message_count >= 0 and message_count < 3:
                    # LLM generates a meaningful title asynchronously
                    generate_session_title.delay(
                        session_id=str(self.session_id),
                        user_query=user_query[:400],  
                        channel_name=self.channel_name  
                    )

                # 3. Run the chatbot
                call_hybrid_rag_chatbot.delay(
                    session_id=str(self.session_id),
                    project_id=self.project_id,
                    user_query=user_query,
                    sources=sources,
                    user_uuid=str(self.user.user_uuid)
                )

            else:
                print(f"Unhandled message type: {message_type}")

        except json.JSONDecodeError:
            print(f"Invalid JSON: {text_data}")
        except Exception as e:
            print(f"Chat error: {e}")
            await self.send(text_data=json.dumps({"type": "error", "content": "Failed to process message."}))
    
    async def send_message(self, event):
        """
        This handler catches messages sent from Celery tasks 
        (where type="chat_message") and pushes them to the browser.
        """
        message = event['message']
        
        # Send the message to the actual WebSocket
        await self.send(text_data=message)