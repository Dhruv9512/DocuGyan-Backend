# DocuChat/tasks.py
import json
import logging
from banks import ChatMessage
from celery import shared_task
from DocuGyan.celery import stop_task
from DocuChat.rag.hybrid_rag import build_hybrid_rag_chatbot
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from DocuChat.utils.llm_calls import DocuChatLLMCalls
from DocuChat.models import ChatSession



@shared_task(bind=True, queue="DocuChat_tasks", time_limit=1800, soft_time_limit=1750, name="call_hybrid_rag_chatbot_task")
def call_hybrid_rag_chatbot(self, session_id, project_id, user_query, sources, user_uuid):
    try:
        channel_layer = get_channel_layer()
        print(f"Starting hybrid RAG chatbot task for session {session_id}")

        chatbot = build_hybrid_rag_chatbot(
            session_id=session_id,
            project_id=project_id,
            user_query=user_query,
            sources=sources,
            user_uuid=user_uuid,
        )
        response = chatbot.chat()  

        if not response:
            print("Hybrid RAG returned empty for session %s", session_id)
            async_to_sync(channel_layer.send)(
                f'chat_{project_id}',
                {
                    "type": "send_message",
                    "message": json.dumps({
                        "type": "error",
                        "content": "The chatbot failed to generate a response."
                    })
                }
            )
            stop_task(self.request.id)
            return None
        

        print(f"Hybrid RAG chatbot task completed for session {session_id}")

        return response

    except Exception as e:
        print("Error in call_hybrid_rag_chatbot: %s", str(e))
        async_to_sync(channel_layer.send)(
            f'chat_{project_id}',
            {
                "type": "send_message",
                "message": json.dumps({
                    "type": "error",
                    "content": "An error occurred while processing your request."
                })
            }
        )
        stop_task(self.request.id)
        return None

@shared_task(bind=True, queue="DocuChat_tasks", name="generate_session_title_task")
def generate_session_title(self,session_id, user_query, channel_name):
    """
    Generates an LLM-based session title and pushes it back via WebSocket.
    """
    try:
        # Call the LLM utility — handles all fallbacks internally
        title = DocuChatLLMCalls.GenerateSessionTitle(user_query)

        # Save to DB only if title is still empty (guard against race conditions)
        session = ChatSession.objects.get(session_id=session_id)
        if not session.title:
            session.title = title
            session.save()

        # Push title back to the websocket client
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.send)(
            channel_name,
            {
                "type": "send_message",
                "message": json.dumps({
                    "type": "title_updated",
                    "title": title,
                    "session_id": session_id
                })
            }
        )

    except Exception as e:
        logging.getLogger(__name__).error("generate_session_title task failed: %s", e)
        stop_task(self.request.id)
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.send)(
            channel_name,
            {
                "type": "send_message",
                "message": json.dumps({
                    "type": "error",
                    "error": "Failed to generate session title."
                })
            }
        )