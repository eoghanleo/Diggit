import gradio as gr
import os
import time
import json
import uuid
from datetime import datetime
import logging
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import re
from groq import Groq
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# ——— Configuration ———
class AppConfig:
    def __init__(self):
        self.max_response_words = 300
        self.context_window = 4
        self.enable_logging = True
        self.use_groq = True

config = AppConfig()

# ——— Global State Management ———
class AppState:
    def __init__(self):
        self.property_id = None
        self.chat_history = []
        self.session_id = str(uuid.uuid4())
        self.conversation_id = None
        self.message_counter = 0
        self.last_query_info = None
        self.execution_log = []
        self.performance_metrics = {
            'response_times': [],
            'retrieval_times': [],
            'refinement_count': 0,
            'total_requests': 0,
            'errors': []
        }

# Global state instance
app_state = AppState()

# ——— Conversation Logging ———
class ConversationLogger:
    """Minimal conversation logging using VARIANT for metadata."""
    def __init__(self, session: Session):
        self.session = session
        self.table_name = "CHAT_CONVERSATIONS"
        self.messages_table = "CHAT_MESSAGES"
        self._ensure_tables_exist()

    def _ensure_tables_exist(self):
        try:
            conversations_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                CONVERSATION_ID VARCHAR PRIMARY KEY,
                PROPERTY_ID INTEGER,
                SESSION_ID VARCHAR,
                START_TIME TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                END_TIME TIMESTAMP_NTZ,
                STATUS VARCHAR DEFAULT 'ACTIVE'
            )
            """
            messages_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.messages_table} (
                MESSAGE_ID VARCHAR PRIMARY KEY,
                CONVERSATION_ID VARCHAR,
                PROPERTY_ID INTEGER,
                MESSAGE_ORDER INTEGER,
                ROLE VARCHAR,
                CONTENT TEXT,
                METADATA VARIANT,
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
            """
            self.session.sql(conversations_sql).collect()
            self.session.sql(messages_sql).collect()
        except Exception:
            pass

    def start_conversation(self, property_id: int, session_id: str) -> str:
        conversation_id = str(uuid.uuid4())
        insert_sql = f"""
        INSERT INTO {self.table_name} (
            CONVERSATION_ID, PROPERTY_ID, SESSION_ID
        ) VALUES (?, ?, ?)
        """
        try:
            self.session.sql(insert_sql, params=[conversation_id, property_id, session_id]).collect()
            return conversation_id
        except Exception:
            return None

    def log_message(self, conversation_id: str, role: str, content: str, metadata: dict = None, property_id: int = None) -> bool:
        if not conversation_id:
            return False
        message_id = str(uuid.uuid4())
        try:
            count_sql = f"SELECT COUNT(*) as msg_count FROM {self.messages_table} WHERE CONVERSATION_ID = ?"
            count_result = self.session.sql(count_sql, params=[conversation_id]).collect()
            message_order = count_result[0].MSG_COUNT + 1 if count_result else 1
        except Exception:
            message_order = 1
        insert_sql = f"""
        INSERT INTO {self.messages_table} (
            MESSAGE_ID, CONVERSATION_ID, PROPERTY_ID, MESSAGE_ORDER, ROLE, CONTENT, METADATA
        ) SELECT ?, ?, ?, ?, ?, ?, PARSE_JSON(?)
        """
        try:
            params = [
                message_id,
                conversation_id,
                property_id,
                message_order,
                role,
                content,
                json.dumps(metadata or {})
            ]
            self.session.sql(insert_sql, params=params).collect()
            return True
        except Exception as e:
            log_execution("❌ Snowflake log_message error", str(e))
            return False

    def end_conversation(self, conversation_id: str):
        if not conversation_id:
            return
        update_sql = f"""
        UPDATE {self.table_name}
        SET END_TIME = CURRENT_TIMESTAMP(), STATUS = 'COMPLETED'
        WHERE CONVERSATION_ID = ?
        """
        try:
            self.session.sql(update_sql, params=[conversation_id]).collect()
        except Exception:
            pass

# ——— Initialize Groq client ———
def get_groq_client():
    """Initialize Groq client with API key."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("⚠️ Groq API key not found. Please set GROQ_API_KEY environment variable.")
        return None
    return Groq(api_key=api_key)

groq_client = get_groq_client()

# ——— Initialize Snowpark session ———
def get_session():
    """Create and cache Snowpark session."""
    try:
        # Try to get active session first (for Snowflake environment)
        return get_active_session()
    except:
        # Fallback for local development
        connection_parameters = {
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "user": os.getenv("SNOWFLAKE_USER"),
            "authenticator": os.getenv("SNOWFLAKE_AUTHENTICATOR"),
            "private_key": os.getenv("SNOWFLAKE_PRIVATE_KEY"),
            "role": os.getenv("SNOWFLAKE_ROLE"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "database": os.getenv("SNOWFLAKE_DATABASE"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA")
        }
        return Session.builder.configs(connection_parameters).create()

session = get_session()
conversation_logger = ConversationLogger(session)

# ——— Constants ———
MODEL_NAME = 'llama3-70b-8192'
FALLBACK_MODEL = 'MIXTRAL-8X7B'
EMBED_MODEL = 'SNOWFLAKE-ARCTIC-EMBED-L-V2.0'
EMBED_FN = 'SNOWFLAKE.CORTEX.EMBED_TEXT_1024'
WORD_THRESHOLD = 100
TOP_K = 5
SIMILARITY_THRESHOLD = 0.2

# ——— Execution Logging ———
def log_execution(step: str, details: str = "", timing: float = None):
    """Log execution steps for debugging."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_entry = {
        "timestamp": timestamp,
        "step": step,
        "details": details,
        "timing": f"{timing:.3f}s" if timing else ""
    }
    app_state.execution_log.append(log_entry)
    
    if len(app_state.execution_log) > 50:
        app_state.execution_log = app_state.execution_log[-50:]

# ——— System Optimization ———
def optimize_warehouse():
    """Set warehouse for retrieval operations."""
    try:
        session.sql("USE WAREHOUSE RETRIEVAL").collect()
        session.sql("ALTER SESSION SET USE_CACHED_RESULT = TRUE").collect()
        session.sql("ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = 300").collect()
        return True
    except Exception as e:
        logging.error(f"Warehouse optimization failed: {e}")
        return False

# Initialize warehouse
optimize_warehouse()

# ——— System Prompt ———
def get_system_prompt(property_id: int) -> str:
    """Generate system prompt for the model."""
    return json.dumps({
        "role": "system",
        "content": {
            "persona": "helpful, knowledgeable plant hire equipment expert",
            "tone": "clear, professional, safety-focused",
            "focus_rule": "Answer only the most recent equipment inquiry",
            "strict_context_rule": (
                "You MUST ONLY answer using the provided context below. "
                "If the answer is not present in the context, reply: 'I'm sorry, I don't have that information in the equipment documentation.' "
                "Do NOT use any outside knowledge."
            ),
            "response_constraints": {
                "format": "plain text only",
                "length_limit": f"max {config.max_response_words} words",
                "no_hallucination": "Only use information from the provided context",
                "safety_first": "Always prioritize safety information when relevant",
                "formatting": "Always format lists as bullet points with * or - symbols, and use proper line breaks for readability"
            }
        }
    })

# ——— Question Processing ———
def process_question(raw_q: str, property_id: int, chat_history: list) -> str:
    """Process and enrich the user question with smart context detection."""
    enriched = f"Equipment inquiry for Plant Hire #{property_id}: {raw_q.strip()}"
    
    if len(chat_history) > 1:
        raw_lower = raw_q.lower()
        
        # Check for explicit reference patterns
        explicit_patterns = [
            r'\b(it|this|that)\s+(is|was|does|can|will|should|would)',
            r'\bwhat\s+about\s+(it|this|that|them)\b',
            r'\b(tell|explain|show)\s+me\s+more\b',
            r'\belse\s+about\b',
            r'\bthe\s+same\s+',
            r'\balso\b.*\?',
            r'^(and|but|so)\s+',
            r'\b(how|why|when|where)\s+do\s+(i|you)\s+(use|turn|activate|access)\s+(it|this|that|them)\b'
        ]
        
        has_explicit_reference = any(re.search(pattern, raw_lower) for pattern in explicit_patterns)
        
        if has_explicit_reference and len(chat_history) >= 2:
            last_content = chat_history[-1].get('content', '')
            if len(last_content) > 30:
                context_preview = last_content[:50].replace('\n', ' ').strip()
                if not context_preview.endswith('.'):
                    context_preview += '...'
                context = f" (Following up on: {context_preview})"
                enriched += context
    
    return enriched

# ——— Retrieval Functions ———
def retrieve_safety_information(enriched_q: str, property_id: int):
    """Retrieve safety-related information for the query."""
    try:
        log_execution("🛡️ Starting Safety Retrieval", f"Equipment {property_id}")
        start_time = time.time()
        
        session.sql("USE WAREHOUSE RETRIEVAL").collect()

        safety_sql = f"""
        SELECT
            CHUNK AS snippet,
            CHUNK_INDEX AS chunk_index,
            RELATIVE_PATH AS path,
            VECTOR_COSINE_SIMILARITY(
                LABEL_EMBED,
                {EMBED_FN}('{EMBED_MODEL}', ?)
            ) AS similarity,
            'safety' AS search_type
        FROM TEST_DB.CORTEX.RAW_TEXT
        WHERE PROPERTY_ID = ?
        AND label_embed IS NOT NULL
        AND chunk_type = 'safety'
        AND VECTOR_COSINE_SIMILARITY(
            LABEL_EMBED,
            {EMBED_FN}('{EMBED_MODEL}', ?)
        ) >= {SIMILARITY_THRESHOLD}
        ORDER BY similarity DESC
        LIMIT {TOP_K}
        """

        params = (enriched_q, property_id, enriched_q)
        results = session.sql(safety_sql, params).collect()

        log_execution("✅ Safety Retrieval complete", f"{len(results)} results in {time.time() - start_time:.2f}s")
        return results

    except Exception as e:
        log_execution("❌ Safety Retrieval error", str(e))
        return []

def retrieve_operational_information(enriched_q: str, property_id: int):
    """Retrieve operational/troubleshooting information for the query."""
    try:
        log_execution("🔧 Starting Operational Retrieval", f"Equipment {property_id}")
        start_time = time.time()
        
        session.sql("USE WAREHOUSE RETRIEVAL").collect()

        operational_sql = f"""
        SELECT
            CHUNK AS snippet,
            CHUNK_INDEX AS chunk_index,
            RELATIVE_PATH AS path,
            VECTOR_COSINE_SIMILARITY(
                LABEL_EMBED,
                {EMBED_FN}('{EMBED_MODEL}', ?)
            ) AS similarity,
            'operational' AS search_type
        FROM TEST_DB.CORTEX.RAW_TEXT
        WHERE PROPERTY_ID = ?
        AND label_embed IS NOT NULL
        AND chunk_type = 'operational'
        AND VECTOR_COSINE_SIMILARITY(
            LABEL_EMBED,
            {EMBED_FN}('{EMBED_MODEL}', ?)
        ) >= {SIMILARITY_THRESHOLD}
        ORDER BY similarity DESC
        LIMIT {TOP_K}
        """

        params = (enriched_q, property_id, enriched_q)
        results = session.sql(operational_sql, params).collect()

        log_execution("✅ Operational Retrieval complete", f"{len(results)} results in {time.time() - start_time:.2f}s")
        return results

    except Exception as e:
        log_execution("❌ Operational Retrieval error", str(e))
        return []

# ——— Response Generation ———
def get_enhanced_answer(chat_history: list, raw_question: str, property_id: int):
    """Generate answer with dual retrieval (safety + operational)."""
    try:
        log_execution("🚀 Starting Answer Generation", f"Question: '{raw_question[:50]}...'")
        
        enriched_q = process_question(raw_question, property_id, chat_history)
        
        retrieval_start = time.time()
        
        safety_results = retrieve_safety_information(enriched_q, property_id)
        operational_results = retrieve_operational_information(enriched_q, property_id)
        
        retrieval_time = time.time() - retrieval_start
        
        all_results = safety_results + operational_results
        
        log_execution("📊 Chunk Summary", f"Safety: {len(safety_results)}, Operational: {len(operational_results)}, Total: {len(all_results)}")
        
        snippets = []
        chunk_idxs = []
        paths = []
        similarities = []
        search_types = []
        
        for row in all_results:
            if hasattr(row, 'SNIPPET'):
                snippets.append(row.SNIPPET)
                chunk_idxs.append(row.CHUNK_INDEX)
                paths.append(row.PATH)
                similarities.append(row.SIMILARITY)
                search_types.append(row.SEARCH_TYPE)
            elif isinstance(row, dict):
                snippets.append(row.get('SNIPPET', row.get('snippet', '')))
                chunk_idxs.append(row.get('CHUNK_INDEX', row.get('chunk_index', 0)))
                paths.append(row.get('PATH', row.get('path', '')))
                similarities.append(row.get('SIMILARITY', row.get('similarity', 0)))
                search_types.append(row.get('SEARCH_TYPE', row.get('search_type', '')))
        
        if not snippets:
            fallback = "I don't have specific information about that equipment. Please contact your equipment supplier or safety officer for assistance."
            return enriched_q, fallback, [], [], [], [], [], False, 0, retrieval_time
        
        # Build context
        context_section = f"Equipment Information:\n"
        
        safety_snippets = [s for i, s in enumerate(snippets) if search_types[i] == 'safety']
        if safety_snippets:
            context_section += f"\n[SAFETY INFORMATION]:\n"
            for snippet in safety_snippets:
                context_section += f"{snippet}\n"
        
        operational_snippets = [s for i, s in enumerate(snippets) if search_types[i] == 'operational']
        if operational_snippets:
            context_section += f"\n[OPERATIONAL INFORMATION]:\n"
            for snippet in operational_snippets:
                context_section += f"{snippet}\n"
        
        system_prompt = get_system_prompt(property_id)
        full_prompt = (
            system_prompt + "\n\n" +
            f"Equipment Operator: {raw_question}\n\n" +
            context_section + "\n\n" +
            "Assistant: Based on the equipment information above, "
        )
        
        # Generate response
        stage1_start = time.time()
        
        if config.use_groq and groq_client:
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful, knowledgeable plant hire equipment expert. Answer only the most recent equipment inquiry using the provided information. Keep responses clear and professional, prioritize safety information when relevant, max 300 words. IMPORTANT: Always format lists as bullet points with * symbols and use proper line breaks for readability."},
                    {"role": "user", "content": f"Equipment Operator: {raw_question}\n\n{context_section}\n\nBased on the equipment information above, please answer the operator's question."}
                ]
                
                completion = groq_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=500,
                    top_p=0.9,
                    stream=False
                )
                
                initial_response = completion.choices[0].message.content.strip()
                log_execution("🚀 Groq Response", f"Tokens: {completion.usage.total_tokens}", time.time() - stage1_start)
                
            except Exception as e:
                log_execution("⚠️ Groq failed, using Cortex", str(e))
                session.sql("USE WAREHOUSE CORTEX_WH").collect()
                
                df = session.sql(
                    "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS response",
                    params=[FALLBACK_MODEL, full_prompt]
                ).collect()
                initial_response = df[0].RESPONSE.strip() if df else "I'm having trouble generating a response."
                log_execution("🤖 Cortex Fallback Response", f"{len(initial_response.split())} words", time.time() - stage1_start)
                
                session.sql("USE WAREHOUSE RETRIEVAL").collect()
        else:
            session.sql("USE WAREHOUSE CORTEX_WH").collect()
            
            df = session.sql(
                "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS response",
                params=[FALLBACK_MODEL, full_prompt]
            ).collect()
            initial_response = df[0].RESPONSE.strip() if df else "I'm having trouble generating a response."
            log_execution("🤖 LLM Response", f"{len(initial_response.split())} words", time.time() - stage1_start)
            
            session.sql("USE WAREHOUSE RETRIEVAL").collect()
        
        word_count = len(initial_response.split())
        formatted_response = format_response_with_safety(initial_response, safety_snippets, operational_snippets, raw_question)
        
        return (enriched_q, formatted_response, snippets, chunk_idxs, paths, 
                similarities, search_types, False, word_count, retrieval_time)
        
    except Exception as e:
        log_execution("❌ Generation Error", str(e))
        return raw_question, "I'm experiencing technical difficulties. Please try again.", [], [], [], [], [], False, 0, 0

def format_response_with_safety(response: str, safety_snippets: list, operational_snippets: list, question: str) -> str:
    """Format response with safety information first."""
    has_relevant_safety = len(safety_snippets) > 0
    
    uncertainty_keywords = ['wrong', 'problem', 'issue', 'error', 'broken', 'not working', 'trouble', 'help', 'unsure', 'confused', 'what should', 'how do i']
    question_lower = question.lower()
    indicates_uncertainty = any(keyword in question_lower for keyword in uncertainty_keywords)
    
    formatted_parts = []
    
    if has_relevant_safety:
        formatted_parts.append("🛡️ **SAFETY FIRST:**")
        formatted_parts.append("⚠️ Always prioritize safety when operating equipment.")
        formatted_parts.append("")
    elif indicates_uncertainty and not has_relevant_safety:
        formatted_parts.append("🛡️ **SAFETY REMINDER:**")
        formatted_parts.append("⚠️ If you're unsure about equipment operation or experiencing issues, stop work immediately and contact your supervisor or safety officer.")
        formatted_parts.append("")
    
    formatted_parts.append(response)
    
    return "\n".join(formatted_parts)

# ——— Gradio Interface Functions ———
def connect_equipment(equipment_id):
    """Connect to specific equipment."""
    try:
        eq_id = int(equipment_id)
        app_state.property_id = eq_id
        app_state.chat_history = []
        app_state.message_counter = 0
        
        if config.enable_logging:
            app_state.conversation_id = conversation_logger.start_conversation(
                eq_id, app_state.session_id
            )
        
        welcome_msg = f"""Welcome! I'm your equipment assistant for **Equipment #{eq_id}**.

I can help you with:
- 🛡️ **Safety procedures** and warnings
- 🔧 **Operating instructions** and controls
- 🔍 **Troubleshooting** common issues
- 📋 **Maintenance** requirements
- ⚠️ **Emergency procedures**

What would you like to know about your equipment?"""
        
        app_state.chat_history.append({"role": "assistant", "content": welcome_msg})
        
        return (
            gr.update(visible=False),  # Hide equipment selection
            gr.update(visible=True),   # Show chat interface
            [[None, welcome_msg]],     # Initialize chatbot with welcome message
            f"Connected to Equipment #{eq_id}"  # Update status
        )
    except ValueError:
        return (
            gr.update(visible=True),   # Keep equipment selection visible
            gr.update(visible=False),  # Hide chat interface
            [],                        # Empty chatbot
            "Please enter a valid equipment ID number"
        )

def chat_with_equipment(message, history):
    """Handle chat messages with the equipment assistant."""
    if not app_state.property_id:
        return history + [[message, "Please connect to an equipment first."]], ""
    
    if not message.strip():
        return history, ""
    
    # Add user message to app state
    app_state.chat_history.append({"role": "user", "content": message})
    
    try:
        # Generate response
        start_time = time.time()
        
        (enriched_q, response, snippets, chunk_idxs, paths, 
         similarities, search_types, used_refinement, word_count, retrieval_time) = get_enhanced_answer(
            app_state.chat_history, 
            message, 
            app_state.property_id
        )
        
        total_time = time.time() - start_time
        
        # Add assistant response to app state
        app_state.chat_history.append({"role": "assistant", "content": response})
        app_state.message_counter += 1
        
        # Log the conversation
        if config.enable_logging and app_state.conversation_id:
            conversation_logger.log_message(
                app_state.conversation_id,
                "user",
                message,
                metadata={"enriched": enriched_q},
                property_id=app_state.property_id
            )
            
            conversation_logger.log_message(
                app_state.conversation_id,
                "assistant",
                response,
                metadata={
                    "word_count": word_count,
                    "retrieval_time": retrieval_time,
                    "total_time": total_time,
                    "snippets_used": len(snippets),
                    "used_refinement": used_refinement
                },
                property_id=app_state.property_id
            )
        
        # Store query info for debugging
        app_state.last_query_info = {
            "question": message,
            "enriched": enriched_q,
            "response": response,
            "snippets": snippets,
            "paths": paths,
            "similarities": similarities,
            "search_types": search_types,
            "timing": {
                "retrieval": retrieval_time,
                "total": total_time
            }
        }
        
        # Update history for display
        history = history + [[message, response]]
        
        return history, ""
        
    except Exception as e:
        error_msg = f"I encountered an error while processing your request: {str(e)}"
        history = history + [[message, error_msg]]
        return history, ""

def switch_equipment():
    """Switch to a different equipment."""
    # End current conversation
    if app_state.conversation_id:
        conversation_logger.end_conversation(app_state.conversation_id)
    
    # Reset state
    app_state.property_id = None
    app_state.chat_history = []
    app_state.conversation_id = None
    app_state.message_counter = 0
    
    return (
        gr.update(visible=True),   # Show equipment selection
        gr.update(visible=False),  # Hide chat interface
        [],                        # Clear chatbot
        "Disconnected. Please select equipment."
    )

def get_debug_info():
    """Get debug information for display."""
    if not app_state.last_query_info:
        return "No query information available yet."
    
    info = app_state.last_query_info
    timing = info['timing']
    
    debug_text = f"""
**Last Query Performance:**
- Retrieval Time: {timing['retrieval']:.2f}s
- Total Response Time: {timing['total']:.2f}s

**Retrieved Chunks:** {len(info['snippets'])}
"""
    
    for i, (snippet, sim, search_type, path) in enumerate(zip(
        info['snippets'], info['similarities'], info['search_types'], info['paths']
    )):
        debug_text += f"\n**Chunk {i+1} - {search_type.title()}**\n"
        debug_text += f"Score: {sim:.3f}\n"
        debug_text += f"Source: {path}\n"
        preview = snippet[:150] + "..." if len(snippet) > 150 else snippet
        debug_text += f"Content: {preview}\n"
    
    return debug_text

# ——— Gradio Interface ———
def create_interface():
    with gr.Blocks(title="Plant Hire Equipment Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚜 Plant Hire Equipment Assistant")
        
        # Status display
        status_display = gr.Textbox(
            value="Please enter equipment ID to connect",
            label="Status",
            interactive=False
        )
        
        # Equipment selection interface
        with gr.Group(visible=True) as equipment_selection:
            gr.Markdown("### Connect to Your Equipment")
            with gr.Row():
                equipment_id_input = gr.Textbox(
                    label="Equipment ID",
                    placeholder="e.g., 2",
                    value="2"
                )
                connect_btn = gr.Button("Connect to Equipment", variant="primary")
        
        # Chat interface
        with gr.Group(visible=False) as chat_interface:
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        height=500,
                        label="Equipment Assistant Chat"
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Ask about your equipment...",
                            label="Your Message",
                            scale=4
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Column(scale=1):
                    switch_btn = gr.Button("🔄 Switch Equipment", variant="secondary")
                    
                    with gr.Accordion("Debug Information", open=False):
                        debug_output = gr.Markdown("No debug information available yet.")
                        refresh_debug_btn = gr.Button("Refresh Debug Info")
                    
                    with gr.Accordion("Settings", open=False):
                        logging_checkbox = gr.Checkbox(
                            label="Enable conversation logging",
                            value=True
                        )
                        groq_checkbox = gr.Checkbox(
                            label="Use Groq API (faster)",
                            value=True
                        )
        
        # Event handlers
        connect_btn.click(
            connect_equipment,
            inputs=[equipment_id_input],
            outputs=[equipment_selection, chat_interface, chatbot, status_display]
        )
        
        def handle_message(message, history):
            return chat_with_equipment(message, history)
        
        msg_input.submit(
            handle_message,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        send_btn.click(
            handle_message,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        switch_btn.click(
            switch_equipment,
            outputs=[equipment_selection, chat_interface, chatbot, status_display]
        )
        
        refresh_debug_btn.click(
            get_debug_info,
            outputs=[debug_output]
        )
        
        # Settings handlers
        def update_logging(enabled):
            config.enable_logging = enabled
            return f"Logging {'enabled' if enabled else 'disabled'}"
        
        def update_groq(enabled):
            config.use_groq = enabled
            return f"Groq API {'enabled' if enabled else 'disabled'}"
        
        logging_checkbox.change(
            update_logging,
            inputs=[logging_checkbox],
            outputs=[]
        )
        
        groq_checkbox.change(
            update_groq,
            inputs=[groq_checkbox],
            outputs=[]
        )
    
    return demo

# ——— Main Application ———
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )