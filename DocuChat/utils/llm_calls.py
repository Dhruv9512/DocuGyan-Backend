import logging
from core.utils.llm_engine import LLMEngine
from DocuChat.prompts.chatbot_prompt import TITLE_PROMPT, GENERATE_PROMPT, RETRIEVAL_DECISION_PROMPT, FALLBACK_PROMPT

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# FULL MODEL POOL — May 2026
#
# PROVIDER        MODEL                                   SPEED     TIER
# ─────────────────────────────────────────────────────────────────────────────
# Groq            openai/gpt-oss-120b                     500 t/s   Production ✅  ← strongest
# Groq            meta-llama/llama-4-scout-17b-16e-instruct 460 t/s Preview  ⚠️  ← vision+text
# Groq            llama-3.3-70b-versatile                 280 t/s   Production ✅
# Groq            qwen/qwen3-32b                          400 t/s   Preview    ⚠️  (replaces qwen-qwq-32b)
# Groq            openai/gpt-oss-20b                     1000 t/s   Production ✅
# Groq            llama-3.1-8b-instant                    560 t/s   Production ✅
# HuggingFace     Qwen/Qwen2.5-72B-Instruct               —         Free       ✅  ← strongest HF text
# HuggingFace     meta-llama/Llama-3.1-70B-Instruct       —         Free       ✅
# HuggingFace     meta-llama/Llama-3.1-8B-Instruct        —         Free       ✅
# Cerebras        gpt-oss-120b                           1800 t/s   Production ✅  ← strongest Cerebras
# Cerebras        llama3.1-8b                            2400 t/s   Production ✅  ← fastest last resort
#
# DEPRECATION NOTES (May 2026):
#   REMOVED from Groq  : llama-4-maverick (→ openai/gpt-oss-120b)
#                        qwen-qwq-32b     (→ qwen/qwen3-32b)
#                        gemma2-9b-it     (→ llama-3.1-8b-instant)
#                        llama3-groq-70b-8192-tool-use-preview (old/deprecated)
#                        llama-3.1-70b-versatile (deprecated)
#   REMOVED from Cerebras: llama-3.3-70b (deprecated Feb 16 2026)
#                           qwen-3-32b   (deprecated Feb 16 2026)
#
# WHY HuggingFace reduces hallucination:
#   HuggingFace Inference endpoints serve models at full or near-full precision.
#   Qwen2.5-72B-Instruct is one of the strongest open instruction-following
#   models available free, with excellent structured output / JSON fidelity.
#
# FREE API KEYS:
#   Groq:         console.groq.com       (no credit card)
#   HuggingFace:  huggingface.co         (no credit card)
#   Cerebras:     cloud.cerebras.ai      (no credit card)
#
# INSTALL:
#   pip install langchain-groq langchain-huggingface langchain-cerebras
# ══════════════════════════════════════════════════════════════════════════════


class DocuChatLLMCalls:
    """
    DocuChat Specialist: Handles all LLM calls for the chat interface.

    Chain catalogue:
        1. _title_chain   — Session title generation (lightweight, fast)
        2. _agent_llms    — Agent node: tool-calling capable models
        3. _generate_llms — Generate node: pure text generation models

    Provider fallback order (strongest → weakest):
        Groq (gpt-oss-120b) → Groq (llama-4-scout) → Groq (llama-3.3-70b)
        → Groq (qwen3-32b) → Groq (gpt-oss-20b) → HuggingFace (Qwen2.5-72B)
        → HuggingFace (Llama-3.1-70B) → Cerebras (gpt-oss-120b)
        → Cerebras (llama3.1-8b)
    """

    _title_chain          = None
    _title_chain_backup   = None
    _agent_llms           = None
    _agent_llms_backup    = None
    _generate_llms        = None
    _generate_llms_backup = None


    # ══════════════════════════════════════════════════════════════════════
    # WORKFLOW 1 — Session Title Generation
    # ══════════════════════════════════════════════════════════════════════

    @classmethod
    def _get_title_chain(cls, use_backup: bool = False):
        """
        Lightweight chain for title generation — small/fast models only.

        Strongest → weakest (speed matters most here):
        1. openai/gpt-oss-20b    (Groq 1000 t/s — fastest production)
        2. llama-3.1-8b-instant  (Groq 560 t/s)
        3. llama3.1-8b           (Cerebras 2400 t/s — absolute fastest last resort)
        """
        if use_backup and cls._title_chain_backup is not None:
            return cls._title_chain_backup
        if not use_backup and cls._title_chain is not None:
            return cls._title_chain
 
        primary = LLMEngine.get_groq_client(
            model_name="llama3.1-8b", temperature=0.3, use_backup=use_backup
        )
        fallback_1 = LLMEngine.get_groq_client(
            model_name="llama-3.1-8b-instant", temperature=0.3, use_backup=use_backup
        )
        fallback_2 = LLMEngine.get_cerebras_client(
            model_name="openai/gpt-oss-20b", temperature=0.3, use_backup=use_backup
        )

        chain = TITLE_PROMPT | primary.with_fallbacks([fallback_1, fallback_2])

        if use_backup:
            cls._title_chain_backup = chain
        else:
            cls._title_chain = chain

        return chain


    @classmethod
    def GenerateSessionTitle(cls, user_query: str) -> str:
        """
        Generates a short, meaningful session title from the user's first message.

        Returns:
            Concise title string (max 60 chars), e.g. "Refund Policy Overview"
        """

        try:
            result = cls._get_title_chain().invoke({"question": user_query})
            title = result.content.strip()[:60]
            logger.info("[GenerateSessionTitle] Success | Title: %s", title)
            return title

        except Exception as e:
            try:
                logger.warning("[GenerateSessionTitle] Primary failed, trying backup. Error: %s", e)
                result = cls._get_title_chain(use_backup=True).invoke({"question": user_query})
                title = result.content.strip()[:60]
                logger.info("[GenerateSessionTitle Backup] Success | Title: %s", title)
                return title

            except Exception as backup_e:
                logger.error("FATAL: GenerateSessionTitle all chains failed. Error: %s", backup_e)
                fallback_title = (user_query[:37] + "...") if len(user_query) > 40 else user_query
                logger.warning("[GenerateSessionTitle] Using truncated fallback: %s", fallback_title)
                return fallback_title


    # ══════════════════════════════════════════════════════════════════════
    # WORKFLOW 2 — Agent Node (Tool Calling)
    # ══════════════════════════════════════════════════════════════════════

    @classmethod
    def _get_agent_llms(cls, use_backup: bool = False) -> list:
        """
        9-model fallback pool. All models support function/tool calling.
        Tools are bound at call time in AgentLLM() — NOT here — so these
        cached clients can be reused across different tool sets.

        Ordered strongest → weakest:
        ── Groq Production (stable, won't be deprecated suddenly) ────────
        1. openai/gpt-oss-120b                  strongest reasoning + tools
        2. llama-3.3-70b-versatile              best Llama tool caller
        ── Groq Preview (may be discontinued without notice) ─────────────
        3. meta-llama/llama-4-scout-17b-16e-instruct  fast, multimodal
        4. qwen/qwen3-32b                       strong tool calling (replaces qwen-qwq-32b)
        ── Groq Production ───────────────────────────────────────────────
        5. openai/gpt-oss-20b                   fastest production model
        6. llama-3.1-8b-instant                 last Groq resort
        ── HuggingFace (free, strong instruction following) ──────────────
        7. Qwen/Qwen2.5-72B-Instruct            strongest free HF text model
        8. meta-llama/Llama-3.1-70B-Instruct    strong instruction following
        ── Cerebras (free, ultra-fast last resort) ───────────────────────
        9. gpt-oss-120b                         ~1800 t/s, strongest Cerebras
        10. llama3.1-8b                         ~2400 t/s, fastest last resort
        """
        if use_backup and cls._agent_llms_backup is not None:
            return cls._agent_llms_backup
        if not use_backup and cls._agent_llms is not None:
            return cls._agent_llms

        llms = [
            # ── Groq production (strongest first) ──
            LLMEngine.get_groq_client(
                model_name="openai/gpt-oss-120b", temperature=0.0, use_backup=use_backup
            ),
            LLMEngine.get_groq_client(
                model_name="llama-3.3-70b-versatile", temperature=0.0, use_backup=use_backup
            ),
            # ── Groq preview ──
            LLMEngine.get_groq_client(
                model_name="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.0, use_backup=use_backup
            ),
            LLMEngine.get_groq_client(
                model_name="qwen/qwen3-32b", temperature=0.0, use_backup=use_backup
            ),
            # ── Groq production lightweight ──
            LLMEngine.get_groq_client(
                model_name="openai/gpt-oss-20b", temperature=0.0, use_backup=use_backup
            ),
            LLMEngine.get_groq_client(
                model_name="llama-3.1-8b-instant", temperature=0.0, use_backup=use_backup
            ),
            # ── HuggingFace (free, full-precision, strong instruction follow) ──
            LLMEngine.get_huggingface_chat_client(
                model_name="Qwen/Qwen2.5-72B-Instruct", temperature=0.0, use_backup=use_backup
            ),
            LLMEngine.get_huggingface_chat_client(
                model_name="meta-llama/Llama-3.1-70B-Instruct", temperature=0.0, use_backup=use_backup
            ),
            # ── Cerebras (free, ultra-fast last resort) ──
            LLMEngine.get_cerebras_client(
                model_name="gpt-oss-120b", temperature=0.0, use_backup=use_backup
            ),
            LLMEngine.get_cerebras_client(
                model_name="llama3.1-8b", temperature=0.0, use_backup=use_backup
            ),
        ]

        model_names = [getattr(llm, 'model_name', str(llm)) for llm in llms]
        logger.info(
            "[AgentLLMs] Pool initialised | use_backup=%s | count=%d | models=%s",
            use_backup, len(llms), model_names
        )

        if use_backup:
            cls._agent_llms_backup = llms
        else:
            cls._agent_llms = llms

        return llms


    @classmethod
    def AgentLLM(cls, conversation_messages: list, tools: list) -> object:
        """
        Invokes the agent LLM with tools bound. Full fallback across all providers.

        Args:
            messages : List of LangChain message objects.
            tools    : List of LangChain-compatible tools to bind to the LLM.

        Returns:
            AIMessage response from the LLM.
        """
        try:
            llms = cls._get_agent_llms()
            primary = llms[0].bind_tools(tools)
            fallbacks = [llm.bind_tools(tools) for llm in llms[1:]]
            chain = RETRIEVAL_DECISION_PROMPT | primary.with_fallbacks(fallbacks)

            result = chain.invoke({"messages": conversation_messages})
            logger.info("[AgentLLM] Success")
            return result

        except Exception as e:
            try:
                logger.warning("[AgentLLM] Primary pool failed, trying backup. Error: %s", e)
                llms = cls._get_agent_llms(use_backup=True)
                primary = llms[0].bind_tools(tools)
                fallbacks = [llm.bind_tools(tools) for llm in llms[1:]]
                chain = RETRIEVAL_DECISION_PROMPT | primary.with_fallbacks(fallbacks)

                result = chain.invoke({"messages": conversation_messages})
                logger.info("[AgentLLM Backup] Success")
                return result

            except Exception as backup_e:
                logger.error("FATAL: AgentLLM all chains failed. Error: %s", backup_e)
                raise RuntimeError(f"All Agent LLMs failed: {e}") from backup_e


    # ══════════════════════════════════════════════════════════════════════
    # WORKFLOW 3 — Generate Node (Pure Text Generation)
    # ══════════════════════════════════════════════════════════════════════

    @classmethod
    def _get_generate_llms(cls, use_backup: bool = False) -> list:
        """
        10-model fallback pool. Pure generation — tool calling NOT required.

        Ordered strongest → weakest:
        ── Groq Production ───────────────────────────────────────────────
        1. openai/gpt-oss-120b              strongest reasoning + generation
        2. llama-3.3-70b-versatile          best quality Llama output
        ── Groq Preview ──────────────────────────────────────────────────
        3. meta-llama/llama-4-scout-17b-16e-instruct  fast + capable
        4. qwen/qwen3-32b                   quality output (replaces qwen-qwq-32b)
        ── Groq Production Lightweight ───────────────────────────────────
        5. openai/gpt-oss-20b               fastest production
        6. llama-3.1-8b-instant             last Groq resort
        ── HuggingFace (free, strong instruction following) ──────────────
        7. Qwen/Qwen2.5-72B-Instruct        strongest free HF; very accurate
        8. meta-llama/Llama-3.1-70B-Instruct strong instruction following
        9. meta-llama/Llama-3.1-8B-Instruct  lightweight HF fallback
        ── Cerebras (free, ultra-fast last resort) ───────────────────────
        10. gpt-oss-120b                    ~1800 t/s, strongest Cerebras
        11. llama3.1-8b                     ~2400 t/s, fastest last resort
        """
        if use_backup and cls._generate_llms_backup is not None:
            return cls._generate_llms_backup
        if not use_backup and cls._generate_llms is not None:
            return cls._generate_llms

        llms = [
            # ── Groq production (strongest first) ──
            LLMEngine.get_groq_client(
                model_name="openai/gpt-oss-120b", temperature=0.5, use_backup=use_backup
            ),
            LLMEngine.get_groq_client(
                model_name="llama-3.3-70b-versatile", temperature=0.5, use_backup=use_backup
            ),
            # ── Groq preview ──
            LLMEngine.get_groq_client(
                model_name="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.5, use_backup=use_backup
            ),
            LLMEngine.get_groq_client(
                model_name="qwen/qwen3-32b", temperature=0.5, use_backup=use_backup
            ),
            # ── Groq production lightweight ──
            LLMEngine.get_groq_client(
                model_name="openai/gpt-oss-20b", temperature=0.5, use_backup=use_backup
            ),
            LLMEngine.get_groq_client(
                model_name="llama-3.1-8b-instant", temperature=0.5, use_backup=use_backup
            ),
            # ── HuggingFace (free, strong instruction following) ──
            LLMEngine.get_huggingface_chat_client(
                model_name="Qwen/Qwen2.5-72B-Instruct", temperature=0.5, use_backup=use_backup
            ),
            LLMEngine.get_huggingface_chat_client(
                model_name="meta-llama/Llama-3.1-70B-Instruct", temperature=0.5, use_backup=use_backup
            ),
            LLMEngine.get_huggingface_chat_client(
                model_name="meta-llama/Llama-3.1-8B-Instruct", temperature=0.5, use_backup=use_backup
            ),
            # ── Cerebras (free, ultra-fast last resort) ──
            LLMEngine.get_cerebras_client(
                model_name="gpt-oss-120b", temperature=0.5, use_backup=use_backup
            ),
            LLMEngine.get_cerebras_client(
                model_name="llama3.1-8b", temperature=0.5, use_backup=use_backup
            ),
        ]

        model_names = [getattr(llm, 'model_name', str(llm)) for llm in llms]
        logger.info(
            "[GenerateLLMs] Pool initialised | use_backup=%s | count=%d | models=%s",
            use_backup, len(llms), model_names
        )

        if use_backup:
            cls._generate_llms_backup = llms
        else:
            cls._generate_llms = llms

        return llms


    @classmethod
    def GenerateLLM(cls, context: str, conversation_messages: list) -> object:
        """
        Streams generation output with full fallback support.

        Args:
            context : The retrieved context for the LLM to use.
            conversation_messages : The list of messages in the conversation.

        Yields:
            Text chunks (str) from the LLM stream.
        """
        try:
            llms  = cls._get_generate_llms()
            chain = GENERATE_PROMPT | llms[0].with_fallbacks(llms[1:])

            logger.info("[StreamGenerateLLM] Starting stream")
            for chunk in chain.stream({"context": context, "messages": conversation_messages}):
                yield chunk.content
            logger.info("[GenerateLLM] Success")

        except Exception as e:
            try:
                logger.warning("[GenerateLLM] Primary pool failed, trying backup. Error: %s", e)
                llms  = cls._get_generate_llms(use_backup=True)
                chain = GENERATE_PROMPT | llms[0].with_fallbacks(llms[1:])

                logger.info("[StreamGenerateLLM Backup] Starting stream")
                for chunk in chain.stream({"context": context, "messages": conversation_messages}):
                    yield chunk.content
                logger.info("[GenerateLLM Backup] Success")

            except Exception as backup_e:
                logger.error("FATAL: GenerateLLM all chains failed. Error: %s", backup_e)
                raise RuntimeError(f"All Generate LLMs failed: {e}") from backup_e


    # ══════════════════════════════════════════════════════════════════════
    # WORKFLOW 4 — Fallback LLM (No Documents Found)
    # ══════════════════════════════════════════════════════════════════════  
    @classmethod
    def _get_fallback_llms(cls, use_backup: bool = False) -> list:
        """
        10-model fallback pool. Pure generation — tool calling NOT required.

        Ordered strongest → weakest:
        ── Groq Production ───────────────────────────────────────────────
        1. openai/gpt-oss-120b              strongest reasoning + generation
        2. llama-3.3-70b-versatile          best quality Llama output
        ── Groq Preview ──────────────────────────────────────────────────
        3. meta-llama/llama-4-scout-17b-16e-instruct  fast + capable
        4. qwen/qwen3-32b                   quality output (replaces qwen-qwq-32b)
        ── Groq Production Lightweight ───────────────────────────────────
        5. openai/gpt-oss-20b               fastest production
        6. llama-3.1-8b-instant             last Groq resort
        ── HuggingFace (free, strong instruction following) ──────────────
        7. Qwen/Qwen2.5-72B-Instruct        strongest free HF; very accurate
        8. meta-llama/Llama-3.1-70B-Instruct strong instruction following
        9. meta-llama/Llama-3.1-8B-Instruct  lightweight HF fallback
        ── Cerebras (free, ultra-fast last resort) ───────────────────────
        10. gpt-oss-120b                    ~1800 t/s, strongest Cerebras
        11. llama3.1-8b                     ~2400 t/s, fastest last resort
        """
        if use_backup and cls._generate_llms_backup is not None:
            return cls._generate_llms_backup
        if not use_backup and cls._generate_llms is not None:
            return cls._generate_llms

        llms = [
            # ── Groq production (strongest first) ──
            LLMEngine.get_groq_client(
                model_name="openai/gpt-oss-120b", temperature=0.5, use_backup=use_backup
            ),
            LLMEngine.get_groq_client(
                model_name="llama-3.3-70b-versatile", temperature=0.5, use_backup=use_backup
            ),
            # ── Groq preview ──
            LLMEngine.get_groq_client(
                model_name="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.5, use_backup=use_backup
            ),
            LLMEngine.get_groq_client(
                model_name="qwen/qwen3-32b", temperature=0.5, use_backup=use_backup
            ),
            # ── Groq production lightweight ──
            LLMEngine.get_groq_client(
                model_name="openai/gpt-oss-20b", temperature=0.5, use_backup=use_backup
            ),
            LLMEngine.get_groq_client(
                model_name="llama-3.1-8b-instant", temperature=0.5, use_backup=use_backup
            ),
            # ── HuggingFace (free, strong instruction following) ──
            LLMEngine.get_huggingface_chat_client(
                model_name="Qwen/Qwen2.5-72B-Instruct", temperature=0.5, use_backup=use_backup
            ),
            LLMEngine.get_huggingface_chat_client(
                model_name="meta-llama/Llama-3.1-70B-Instruct", temperature=0.5, use_backup=use_backup
            ),
            LLMEngine.get_huggingface_chat_client(
                model_name="meta-llama/Llama-3.1-8B-Instruct", temperature=0.5, use_backup=use_backup
            ),
            # ── Cerebras (free, ultra-fast last resort) ──
            LLMEngine.get_cerebras_client(
                model_name="gpt-oss-120b", temperature=0.5, use_backup=use_backup
            ),
            LLMEngine.get_cerebras_client(
                model_name="llama3.1-8b", temperature=0.5, use_backup=use_backup
            ),
        ]

        model_names = [getattr(llm, 'model_name', str(llm)) for llm in llms]
        logger.info(
            "[GenerateLLMs] Pool initialised | use_backup=%s | count=%d | models=%s",
            use_backup, len(llms), model_names
        )

        if use_backup:
            cls._generate_llms_backup = llms
        else:
            cls._generate_llms = llms

        return llms
    
    @classmethod
    def FallbackLLM(cls, conversation_messages: list) -> object:
        """
        Streams a dynamic multilingual fallback message when no documents are found.
        """
        try:
            # You can use your standard LLM pool here. 
            llms  = cls._get_fallback_llms()
            chain = FALLBACK_PROMPT | llms[0].with_fallbacks(llms[1:])

            logger.info("[StreamFallbackLLM] Starting stream")
            for chunk in chain.stream({"messages": conversation_messages}):
                yield chunk.content
            logger.info("[FallbackLLM] Success")

        except Exception as e:
            try:
                logger.warning("[FallbackLLM] Primary pool failed, trying backup. Error: %s", e)
                llms  = cls._get_fallback_llms(use_backup=True)
                chain = FALLBACK_PROMPT | llms[0].with_fallbacks(llms[1:])

                logger.info("[StreamFallbackLLM Backup] Starting stream")
                for chunk in chain.stream({"messages": conversation_messages}):
                    yield chunk.content
                logger.info("[FallbackLLM Backup] Success")

            except Exception as backup_e:
                logger.error("FATAL: FallbackLLM all chains failed. Error: %s", backup_e)
                raise RuntimeError(f"All Fallback LLMs failed: {e}") from backup_e










