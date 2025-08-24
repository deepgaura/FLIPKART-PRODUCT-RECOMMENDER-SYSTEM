from textwrap import dedent
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from flipkart.config import Config


class RAGChainBuilder:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.model = ChatGroq(model=Config.RAG_MODEL, temperature=0.4)
        self.history_store = {}

    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        # If you want persistence across restarts, switch to SQLChatMessageHistory here.
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]

    def build_chain(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        # ---- 1) Rewrite to standalone using chat history (improves retrieval) ----
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rewrite the latest user message as a standalone question using chat_history."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        history_aware_retriever = create_history_aware_retriever(self.model, retriever, rewrite_prompt)

        # ---- 2) Summarize chat history into actionable memory (ALWAYS included) ----
        memory_prompt = ChatPromptTemplate.from_messages([
            ("system", dedent("""
                From chat_history and the latest user input, extract the user's
                preferences/constraints as short bullets: budget, brands, features,
                use-case, dislikes, previously chosen items.
                If nothing explicit, return "none".
            """).strip()),
            MessagesPlaceholder("chat_history"),
            ("human", "New request: {input}")
        ])
        memory_chain = memory_prompt | self.model | StrOutputParser()

        # ---- 3) QA prompt that uses ALL THREE signals: memory + context + general knowledge ----
        qa_system = dedent("""
            You are a product recommender. Use THREE signals:

            1) **chat_history/memory**: tailor suggestions to the user's stated prefs/constraints.
            2) **CONTEXT** (retrieved docs): the source of truth for product facts (names, specs, reviews).
               If a fact isn't in CONTEXT, don't assert it as certain.
            3) **General knowledge**: give generic buying tips, but never contradict CONTEXT.

            If the question is about the conversation itself (e.g., "what did I ask first?"),
            answer from chat_history/memory.

            Format answers in clean Markdown: brief TL;DR, short bullets, no rambling.

            MEMORY:
            {memory}

            CONTEXT:
            {context}
        """).strip()

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system),
            MessagesPlaceholder("chat_history"),
            ("human", "QUESTION: {input}")
        ])

        combine_docs_chain = create_stuff_documents_chain(self.model, qa_prompt)

        # Instead of the simple helper, build a map so we can pass {memory} too.
        # The retriever and memory_chain both receive the full inputs (incl. chat_history).
        rag_with_memory = (
            {
                "context": history_aware_retriever,   # uses rewritten question + chat_history
                "memory": memory_chain,               # summarized preferences from history
                "input": RunnablePassthrough(),       # the user's latest message
            }
            | combine_docs_chain                      # renders the final answer
        )

        # If you prefer the convenience wrapper, this is equivalent:
        # rag_with_memory = create_retrieval_chain(history_aware_retriever, combine_docs_chain)
        # ...but the helper won't forward {memory}, so we use the runnable map above.

        return RunnableWithMessageHistory(
            rag_with_memory,
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
