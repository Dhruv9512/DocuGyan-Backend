from core.utils.utility import get_collection_name

from django.conf import settings

from DocuChat.utils.utility import is_token_limit_error
from DocuChat.utils.hybrid_rag_tools import HybridRAGTools
from DocuChat.utils.llm_calls import DocuChatLLMCalls
from DocuChat.schemas.chatbot_state import ChatState
from DocuChat.models import ChatMessage, ChatSession

from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import json

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
import logging

logger = logging.getLogger(__name__)


class HybridRAGChatbot:
    def __init__(self, session_id, project_id, user_query, sources, user_uuid):
        self.session_id = session_id
        self.project_id = project_id
        self.user_query = user_query
        self.sources = sources
        self.user_uuid = user_uuid
        self.collection_name = get_collection_name(project_id)

        logger.info(
            "[HybridRAGChatbot.__init__] Initializing | session=%s | project=%s | user=%s | sources=%s",
            session_id, project_id, user_uuid, sources
        )

        # Instance variables
        self.session = ChatSession.objects.get(session_id=self.session_id)
        logger.info("[HybridRAGChatbot.__init__] ChatSession loaded | session=%s", session_id)

        # History 
        self.history = self._get_history_chat_messages()
        self.current_chat_history = ChatMessage.objects.create(session=self.session, user_message=self.user_query, assistant_response="")
        logger.info(
            "[HybridRAGChatbot.__init__] ChatMessage record created | session=%s | history_turns=%d",
            session_id, len(self.history) // 2
        )

        # Graph and tools
        self.hybrid_rag_tools = HybridRAGTools(project_id, sources).get_tools()
        logger.info(
            "[HybridRAGChatbot.__init__] Tools loaded | session=%s | tool_count=%d",
            session_id, len(self.hybrid_rag_tools)
        )

        self.memory_saver = MemorySaver()
        self.graph = self._build_graph()
        logger.info("[HybridRAGChatbot] Graph compiled successfully | session=%s", session_id)


    # ─────────────────────────────────────────
    # Nodes
    # ─────────────────────────────────────────

    def _agent_node(self, state: ChatState) -> dict:
        logger.info(
            "[AgentNode] Invoked | session=%s | message_count=%d",
            self.session_id, len(state["messages"])
        )

        self._push_to_websocket({"type": "agent_thinking"})

        conversation_messages = [ msg for msg in state["messages"] if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None) ]

        logger.info("[AgentNode] Calling AgentLLM | session=%s", self.session_id)
        response = DocuChatLLMCalls.AgentLLM(conversation_messages, self.hybrid_rag_tools)

        has_tool_calls = bool(getattr(response, "tool_calls", None))
        logger.info(
            "[AgentNode] Response received | session=%s | has_tool_calls=%s | tool_calls=%s",
            self.session_id, has_tool_calls,
            [tc.get("name") for tc in response.tool_calls] if has_tool_calls else []
        )

        if has_tool_calls:
            self._push_to_websocket({
                "type": "agent_tool_call",
                "tools": [tc.get("name") for tc in response.tool_calls]
            })
        else:
            self._push_to_websocket({"type": "agent_done"})

        return {"messages": [response]}


    def _generate_node(self, state: ChatState) -> dict:
        retrieved = [
            msg.content
            for msg in state["messages"]
            if isinstance(msg, ToolMessage) and msg.content
        ]

        logger.info(
            "[GenerateNode] Invoked | session=%s | retrieved_chunks=%d",
            self.session_id, len(retrieved)
        )

        # ---------------------------------------------------------
        # SCENARIO A: Documents Were Found
        # ---------------------------------------------------------
        if retrieved:
            context = "\n\n".join(retrieved)
            conversation_messages = [ msg for msg in state["messages"] if isinstance(msg, (HumanMessage, AIMessage)) and not getattr(msg, "tool_calls", None) ]
            final_answer = ""

            self._push_to_websocket({"type": "stream_start"})

            try:
                for chunk_text in DocuChatLLMCalls.GenerateLLM(context, conversation_messages):
                    if chunk_text:
                        final_answer += chunk_text
                        self._push_to_websocket({
                            "type": "stream_chunk",
                            "content": chunk_text
                        })
            except Exception as e:
                logger.error("[GenerateNode] Streaming error: %s", e)
                self._push_to_websocket({"type": "stream_error", "error": str(e)})
                return None

            self._push_to_websocket({"type": "stream_end"})

            logger.info(
                "[GenerateNode] Answer streamed | session=%s | answer_length=%d",
                self.session_id, len(final_answer)
            )
        # ---------------------------------------------------------
        # SCENARIO B: No Documents Found (Or Agent Greeting)
        # ---------------------------------------------------------
        else:
            logger.warning("[GenerateNode] No documents retrieved | session=%s", self.session_id)
            
            # 1. Catch Native Greetings (If the Agent just said "Hello")
            last_message = state["messages"][-1]
            if last_message.content and not getattr(last_message, "tool_calls", None):
                final_answer = last_message.content
                self._push_to_websocket({"type": "stream_start"})
                self._push_to_websocket({"type": "stream_chunk", "content": final_answer})
                self._push_to_websocket({"type": "stream_end"})
                
            # 2. Dynamic Multilingual Fallback (Using the new FallbackLLM)
            else:
                conversation_messages = [
                    msg for msg in state["messages"]
                    if isinstance(msg, (HumanMessage, AIMessage)) 
                    and not getattr(msg, "tool_calls", None) 
                ]
                
                final_answer = ""
                self._push_to_websocket({"type": "stream_start"})
                
                try:
                    # Using the dedicated, cheaper, faster FallbackLLM
                    for chunk_text in DocuChatLLMCalls.FallbackLLM(conversation_messages):
                        if chunk_text:
                            final_answer += chunk_text
                            self._push_to_websocket({"type": "stream_chunk", "content": chunk_text})
                except Exception as e:
                    logger.error("[GenerateNode] Streaming error during fallback: %s", e)
                    self._push_to_websocket({"type": "stream_error", "error": str(e)})
                    return None

                self._push_to_websocket({"type": "stream_end"})


        # LangGraph still needs the final state object
        return {
            "messages" : [AIMessage(content=final_answer)],
            "final_answer": final_answer
        }


    # ─────────────────────────────────────────
    # Routing
    # ─────────────────────────────────────────

    def _should_retrieve(self, state: ChatState) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logger.info("[Router] Decision=retrieve | session=%s", self.session_id)
            return "retrieve"
        logger.info("[Router] Decision=generate | session=%s", self.session_id)
        return "generate"


    # ─────────────────────────────────────────
    # Graph
    # ─────────────────────────────────────────
    def _build_graph(self) -> StateGraph:
        graph = StateGraph(ChatState)

        graph.add_node("agent", self._agent_node)
        graph.add_node("retrieve", ToolNode(tools=self.hybrid_rag_tools))
        graph.add_node("generate", self._generate_node)

        graph.set_entry_point("agent")

        graph.add_conditional_edges(
            "agent",
            self._should_retrieve,
            {"retrieve": "retrieve", "generate": "generate"}
        )

        # Route directly to generate, bypassing the agent
        graph.add_edge("retrieve", "generate") 
        graph.add_edge("generate", END)

        return graph.compile(checkpointer=self.memory_saver)


    # ─────────────────────────────────────────
    # Public
    # ─────────────────────────────────────────

    def chat(self) -> str | None:
        config = {"configurable": {"thread_id": self.session_id}}
        base_messages = [HumanMessage(content=self.user_query)]

        logger.info(
            "[HybridRAGChatbot.chat] Starting | session=%s | query='%s' | "
            "total_history_messages=%d",
            self.session_id, self.user_query[:80], len(self.history)
        )

        for size in settings.HISTORY_FALLBACK_SIZES:
            # Each turn = HumanMessage + AIMessage, so multiply by 2
            trimmed_history = self.history[-(size * 2):] if size > 0 else []

            logger.info(
                "[HybridRAGChatbot.chat] Attempt | session=%s | "
                "history_turns=%d | history_messages=%d",
                self.session_id, size, len(trimmed_history)
            )

            initial_state: ChatState = {
                "messages" : trimmed_history + base_messages,
                "final_answer": ""
            }

            try:
                result = self.graph.invoke(initial_state, config=config)
                answer = result.get("final_answer", "")

                if answer:
                    logger.info(
                        "[HybridRAGChatbot.chat] Success | session=%s | "
                        "history_turns_used=%d | answer_length=%d",
                        self.session_id, size, len(answer)
                    )
                    self._store_chat_in_db(answer)
                    return answer

                # Graph completed but returned no answer — not a token error, don't retry
                logger.warning(
                    "[HybridRAGChatbot.chat] Graph returned empty answer | session=%s | "
                    "history_turns=%d",
                    self.session_id, size
                )
                self._push_to_websocket({"type": "error", "error": "Graph returned empty answer."})
                break
            except Exception as e:
                if is_token_limit_error(e):
                    logger.warning(
                        "[HybridRAGChatbot.chat] Token limit hit | session=%s | "
                        "history_turns=%d | next_size=%s | error=%s",
                        self.session_id, size,
                        settings.HISTORY_FALLBACK_SIZES[settings.HISTORY_FALLBACK_SIZES.index(size) + 1]
                        if size != settings.HISTORY_FALLBACK_SIZES[-1] else "none (exhausted)",
                        e
                    )
                    self._push_to_websocket({
                        "type": "retrying",
                        "reason": "token_limit",
                        "history_turns": size
                    })
                    continue  # Try next smaller history size

                # Non-token error — no point retrying with less history
                logger.error(
                    "[HybridRAGChatbot.chat] Non-token error, aborting retries | "
                    "session=%s | history_turns=%d | error=%s",
                    self.session_id, size, e, exc_info=True
                )
                self._push_to_websocket({"type": "error", "error": str(e)})
                break

        logger.error(
            "[HybridRAGChatbot.chat] Failed to generate answer | session=%s | "
            "all_sizes_tried=%s",
            self.session_id, settings.HISTORY_FALLBACK_SIZES
        )
        self._push_to_websocket({"type": "error", "error": "Failed to generate an answer."})
        return None
    
    # Helper method to push to websocket
    def _push_to_websocket(self, payload: dict):
        try:
            channel_layer = get_channel_layer()
            async_to_sync(channel_layer.group_send)(
                f'chat_{self.project_id}',
                {
                    "type": "send_message",
                    "message": json.dumps(payload)
                }
            )
            logger.debug(
                "[WebSocket] Pushed | session=%s | type=%s",
                self.session_id, payload.get("type")
            )
        except Exception as e:
            logger.error(
                "[WebSocket] Failed to push message | session=%s | type=%s | error=%s",
                self.session_id, payload.get("type"), e
            )

    def _get_history_chat_messages(self):
        logger.info("[HybridRAGChatbot] Loading chat history | session=%s", self.session_id)

        turns = list(
            ChatMessage.objects.filter(session=self.session)
            .order_by('-created_at')[:10]
        )[::-1]

        history = []
        for turn in turns:
            history.append(HumanMessage(content=turn.user_message))
            history.append(AIMessage(content=turn.assistant_response))

        logger.info(
            "[HybridRAGChatbot] Chat history loaded | session=%s | turns=%d | messages=%d",
            self.session_id, len(turns), len(history)
        )
        return history
    
    def _store_chat_in_db(self, final_response):
        try:
            self.current_chat_history.assistant_response = final_response
            self.current_chat_history.save(update_fields=['assistant_response'])
            logger.info("[DB] ChatMessage stored successfully | session=%s", self.session_id)
        except Exception as e:
            logger.error("[DB] Failed to store ChatMessage | session=%s | error=%s", self.session_id, e)


# ─────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────

def build_hybrid_rag_chatbot(session_id, project_id, user_query, sources, user_uuid):
    return HybridRAGChatbot(
        session_id=session_id,
        project_id=project_id,
        user_query=user_query,
        sources=sources,
        user_uuid=user_uuid
    )