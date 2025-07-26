import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session
import time
import json
import uuid
from datetime import datetime
import logging
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import re
import os
from groq import Groq


# ——— Conversation Logging ———
# To check the schema of the messages table in Snowflake, run:
# DESC TABLE CHAT_MESSAGES;
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
        # Get message order (1-based)
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

    def get_conversation_history(self, property_id: int, limit: int = 10) -> List[Dict]:
        query_sql = f"""
        SELECT 
            CONVERSATION_ID,
            SESSION_ID,
            START_TIME,
            END_TIME,
            STATUS
        FROM {self.table_name}
        WHERE PROPERTY_ID = ?
        ORDER BY START_TIME DESC
        LIMIT ?
        """
        try:
            results = self.session.sql(query_sql, params=[property_id, limit]).collect()
            def row_to_dict(row):
                if hasattr(row, 'as_dict'):
                    return row.as_dict()
                return {
                    "CONVERSATION_ID": getattr(row, "CONVERSATION_ID", None),
                    "SESSION_ID": getattr(row, "SESSION_ID", None),
                    "START_TIME": getattr(row, "START_TIME", None),
                    "END_TIME": getattr(row, "END_TIME", None),
                    "STATUS": getattr(row, "STATUS", None)
                }
            return [row_to_dict(row) for row in results]
        except Exception:
            return []

    def get_conversation_messages(self, conversation_id: str) -> List[Dict]:
        query_sql = f"""
        SELECT 
            MESSAGE_ORDER,
            ROLE,
            CONTENT,
            PROPERTY_ID,
            METADATA,
            CREATED_AT
        FROM {self.messages_table}
        WHERE CONVERSATION_ID = ?
        ORDER BY MESSAGE_ORDER
        """
        try:
            results = self.session.sql(query_sql, params=[conversation_id]).collect()
            def row_to_dict(row):
                if hasattr(row, 'as_dict'):
                    return row.as_dict()
                return {
                    "MESSAGE_ORDER": getattr(row, "MESSAGE_ORDER", None),
                    "ROLE": getattr(row, "ROLE", None),
                    "CONTENT": getattr(row, "CONTENT", None),
                    "PROPERTY_ID": getattr(row, "PROPERTY_ID", None),
                    "METADATA": getattr(row, "METADATA", None),
                    "CREATED_AT": getattr(row, "CREATED_AT", None)
                }
            return [row_to_dict(row) for row in results]
        except Exception:
            return []


# ——— App config ———
st.set_page_config(
    page_title="Plant Hire Equipment Assistant", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ——— Initialize Groq client ———
@st.cache_resource
def get_groq_client():
    """Initialize Groq client with API key."""
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("groq", {}).get("api_key")
    if not api_key:
        st.error("⚠️ Groq API key not found. Please set GROQ_API_KEY environment variable or add to Streamlit secrets.")
        return None
    return Groq(api_key=api_key)

groq_client = get_groq_client()

# ——— Initialize Snowpark session ———
@st.cache_resource
def get_session():
    """Create and cache Snowpark session."""
    try:
        # Try to get active session first (for Snowflake environment)
        return get_active_session()
    except:
        # Fallback for local development
        connection_parameters = {
            "account": st.secrets["snowflake"]["account"],
            "user": st.secrets["snowflake"]["user"],
            "authenticator" : st.secrets["snowflake"]["authenticator"],
            "private_key": st.secrets["snowflake"]["private_key"],
            "role": st.secrets["snowflake"]["role"],
            "warehouse": st.secrets["snowflake"]["warehouse"],
            "database": st.secrets["snowflake"]["database"],
            "schema": st.secrets["snowflake"]["schema"]
        }
        return Session.builder.configs(connection_parameters).create()

session = get_session()

# ——— Initialize Conversation Logger ———
@st.cache_resource
def get_conversation_logger():
    """Initialize conversation logger."""
    return ConversationLogger(session)

conversation_logger = get_conversation_logger()

# ——— Constants ———
MODEL_NAME = 'llama3-70b-8192'  # Updated Groq model name (without context size)
FALLBACK_MODEL = 'MIXTRAL-8X7B'  # Snowflake Cortex fallback
EMBED_MODEL = 'SNOWFLAKE-ARCTIC-EMBED-L-V2.0'
EMBED_FN = 'SNOWFLAKE.CORTEX.EMBED_TEXT_1024'
WORD_THRESHOLD = 100  # Increased from 50 to 100
TOP_K = 5  # Fixed value, no longer configurable
SIMILARITY_THRESHOLD = 0.2  # Fixed value, no longer configurable

# ——— Configuration ———
if 'config' not in st.session_state:
    st.session_state.config = {
        'max_response_words': 300,  # Increased from 100 to 300
        'context_window': 4,
        'enable_logging': True,
        'use_groq': True
    }

# ——— Performance Monitor ———
class PerformanceMonitor:
    def __init__(self):
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = {
                'response_times': [],
                'retrieval_times': [],
                'refinement_count': 0,
                'total_requests': 0,
                'errors': []
            }
        self.metrics = st.session_state.performance_metrics
    
    def log_request(self, metrics: Dict[str, Any]):
        self.metrics['response_times'].append(metrics.get('latency', 0))
        self.metrics['retrieval_times'].append(metrics.get('retrieval_time', 0))
        self.metrics['total_requests'] += 1
        if metrics.get('used_refinement'):
            self.metrics['refinement_count'] += 1
        
        # Keep only last 100 entries
        if len(self.metrics['response_times']) > 100:
            self.metrics['response_times'] = self.metrics['response_times'][-100:]
            self.metrics['retrieval_times'] = self.metrics['retrieval_times'][-100:]
    
    def log_error(self, error_type: str, details: str):
        self.metrics['errors'].append({
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'details': details
        })
        if len(self.metrics['errors']) > 50:
            self.metrics['errors'] = self.metrics['errors'][-50:]
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        if not self.metrics['response_times']:
            return {'status': 'No data yet'}
        
        return {
            'avg_response_time': np.mean(self.metrics['response_times']),
            'avg_retrieval_time': np.mean(self.metrics['retrieval_times']) if self.metrics['retrieval_times'] else 0,
            'p95_response_time': np.percentile(self.metrics['response_times'], 95) if len(self.metrics['response_times']) > 10 else 0,
            'refinement_rate': self.metrics['refinement_count'] / self.metrics['total_requests'] if self.metrics['total_requests'] > 0 else 0,
            'total_requests': self.metrics['total_requests'],
            'recent_errors': len(self.metrics['errors'])
        }

monitor = PerformanceMonitor()

# ——— Error Handling ———
class ChatError:
    def __init__(self, error_type: str, user_message: str, technical_details: str = None):
        self.error_type = error_type
        self.user_message = user_message
        self.technical_details = technical_details
        monitor.log_error(error_type, technical_details or user_message)
    
    def display(self):
        st.error(f"😔 {self.user_message}")
        if self.technical_details:
            with st.expander("Technical details"):
                st.code(self.technical_details)

# ——— Execution Logging ———
if 'execution_log' not in st.session_state:
    st.session_state.execution_log = []

def log_execution(step: str, details: str = "", timing: float = None):
    """Log execution steps for debugging."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_entry = {
        "timestamp": timestamp,
        "step": step,
        "details": details,
        "timing": f"{timing:.3f}s" if timing else ""
    }
    st.session_state.execution_log.append(log_entry)
    
    if len(st.session_state.execution_log) > 50:
        st.session_state.execution_log = st.session_state.execution_log[-50:]

# ——— System Initialization ———
def optimize_warehouse():
    """Set warehouse for retrieval operations."""
    try:
        # Use RETRIEVAL warehouse for all operations
        session.sql("USE WAREHOUSE RETRIEVAL").collect()
        session.sql("ALTER SESSION SET USE_CACHED_RESULT = TRUE").collect()
        session.sql("ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = 300").collect()
        return True
    except Exception as e:
        logging.error(f"Warehouse optimization failed: {e}")
        return False

# Initialize warehouse on startup
if 'warehouse_initialized' not in st.session_state:
    st.session_state.warehouse_initialized = optimize_warehouse()

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
                "length_limit": f"max {st.session_state.config['max_response_words']} words",
                "no_hallucination": "Only use information from the provided context",
                "safety_first": "Always prioritize safety information when relevant",
                "formatting": "Always format lists as bullet points with * or - symbols, and use proper line breaks for readability"
            },
            "examples": [
                {
                    "question": "How do I start the volkswagen golf TDI?",
                    "context": "[No information about volkswagen golf TDI in the context]",
                    "answer": "I'm sorry, I don't have that information in the equipment documentation."
                }
            ]
        }
    })

# ——— Refinement Prompt ———
EDITOR_PROMPT = json.dumps({
    "role": "editor",
    "task": "Make the response more concise while keeping all facts",
    "rules": [
        "Keep the professional, safety-focused tone",
        "Preserve all factual information",
        "Remove redundancy and filler",
        "Maximum 50 words unless more detail is essential"
    ]
})

# ——— Question Processing ———
def process_question(raw_q: str, property_id: int, chat_history: list) -> str:
    """Process and enrich the user question with smart context detection."""
    # Simple enrichment - add equipment context
    enriched = f"Equipment inquiry for Plant Hire #{property_id}: {raw_q.strip()}"
    
    # Smart context detection using entity tracking and explicit reference patterns
    if len(chat_history) > 1:
        raw_lower = raw_q.lower()
        
        # 1. Check for explicit reference patterns that indicate follow-up
        # Using Safari-compatible patterns without complex groups
        has_explicit_reference = False
        
        # Check for simple reference patterns using string matching
        reference_words = ['it', 'this', 'that', 'them']
        action_words = ['is', 'was', 'does', 'can', 'will', 'should', 'would']
        
        # Simple string matching instead of complex regex
        for ref_word in reference_words:
            for action_word in action_words:
                if f"{ref_word} {action_word}" in raw_lower or f"{action_word} {ref_word}" in raw_lower:
                    has_explicit_reference = True
                    break
            if has_explicit_reference:
                break
        
        # Check for other reference patterns
        if not has_explicit_reference:
            reference_phrases = [
                'what about it', 'what about this', 'what about that', 'what about them',
                'tell me more', 'explain more', 'show me more',
                'else about', 'the same', 'also'
            ]
            
            for phrase in reference_phrases:
                if phrase in raw_lower:
                    has_explicit_reference = True
                    break
            
            # Check if starts with conjunctions
            if raw_lower.startswith(('and ', 'but ', 'so ')):
                has_explicit_reference = True
        
        # 2. Entity/topic tracking - extract key nouns from previous exchange
        if not has_explicit_reference and len(chat_history) >= 2:
            # Get the last user question and assistant response
            last_user_msg = next((msg['content'] for msg in reversed(chat_history[:-1]) if msg['role'] == 'user'), "")
            last_assistant_msg = next((msg['content'] for msg in reversed(chat_history) if msg['role'] == 'assistant'), "")
            
            # Extract meaningful entities using simple string matching instead of regex
            equipment_terms = [
                'excavator', 'digger', 'loader', 'bulldozer', 'crane', 'forklift', 'dumper', 'roller',
                'engine', 'hydraulic', 'fuel', 'oil', 'filter', 'battery', 'tire', 'track',
                'operator', 'driver', 'safety', 'helmet', 'vest', 'harness', 'seatbelt',
                'manual', 'instruction', 'procedure', 'checklist', 'inspection',
                'start', 'stop', 'emergency', 'shutdown', 'restart',
                'weight', 'capacity', 'reach', 'height', 'depth', 'angle',
                'terrain', 'ground', 'slope', 'mud', 'water', 'rock',
                'maintenance', 'service', 'repair', 'part', 'spare',
                'control', 'lever', 'pedal', 'button', 'switch', 'gauge',
                'attachment', 'bucket', 'hammer', 'drill', 'grapple',
                'transport', 'trailer', 'loading', 'unloading',
                'weather', 'rain', 'wind', 'temperature', 'visibility'
            ]
            
            # Find entities in previous messages using simple string matching
            previous_entities = set()
            for term in equipment_terms:
                if term in last_user_msg.lower():
                    previous_entities.add(term)
                if term in last_assistant_msg.lower():
                    previous_entities.add(term)
            
            # Check if current question mentions any previous entities
            current_entities = set()
            for term in equipment_terms:
                if term in raw_lower:
                    current_entities.add(term)
            
            # If there's entity overlap, it might be a follow-up
            has_entity_overlap = bool(previous_entities & current_entities)
        else:
            has_entity_overlap = False
        
        # 3. Apply context only if we have strong signals
        if has_explicit_reference or (has_entity_overlap and len(raw_q.split()) < 10):
            # Get the most relevant previous content
            last_exchange = chat_history[-2:] if len(chat_history) >= 2 else chat_history
            last_content = last_exchange[-1]['content']
            
            # Only add context if the last message was substantial and not a greeting
            if len(last_content) > 30 and not any(greeting in last_content.lower() for greeting in ['welcome', 'hello', 'hi there']):
                # Extract the most relevant part of the previous message
                context_preview = last_content[:50].replace('\n', ' ').strip()
                if not context_preview.endswith('.'):
                    context_preview += '...'
                context = f" (Following up on: {context_preview})"
                enriched += context
    
    return enriched

# ——— Dual Retrieval System ———
def retrieve_safety_information(enriched_q: str, property_id: int):
    """Retrieve safety-related information for the query."""
    try:
        log_execution("🛡️ Starting Safety Retrieval", f"Equipment {property_id}")
        start_time = time.time()
        
        # Ensure we're using RETRIEVAL warehouse
        session.sql("USE WAREHOUSE RETRIEVAL").collect()

        # Pure semantic safety retrieval
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
        
        # Ensure we're using RETRIEVAL warehouse
        session.sql("USE WAREHOUSE RETRIEVAL").collect()

        # Pure semantic operational retrieval
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

# ——— Answer Generation ———
def get_enhanced_answer(chat_history: list, raw_question: str, property_id: int):
    """Generate answer with dual retrieval (safety + operational)."""
    try:
        log_execution("🚀 Starting Answer Generation", f"Question: '{raw_question[:50]}...'")
        
        enriched_q = process_question(raw_question, property_id, chat_history)
        
        # Dual retrieval: Safety first, then operational
        retrieval_start = time.time()
        
        # Get safety information
        safety_results = retrieve_safety_information(enriched_q, property_id)
        
        # Get operational information
        operational_results = retrieve_operational_information(enriched_q, property_id)
        
        retrieval_time = time.time() - retrieval_start
        
        # Combine and parse results
        all_results = safety_results + operational_results
        
        log_execution("📊 Chunk Summary", f"Safety: {len(safety_results)}, Operational: {len(operational_results)}, Total: {len(all_results)}")
        
        snippets = []
        chunk_idxs = []
        paths = []
        similarities = []
        search_types = []
        
        for row in all_results:
            # Handle both dictionary-style and attribute-style access
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
            else:
                # If row is a list/tuple, assume order: snippet, chunk_index, path, similarity, search_type
                if len(row) >= 5:
                    snippets.append(row[0])
                    chunk_idxs.append(row[1])
                    paths.append(row[2])
                    similarities.append(row[3])
                    search_types.append(row[4])
        
        if not snippets:
            fallback = "I don't have specific information about that equipment. Please contact your equipment supplier or safety officer for assistance."
            return enriched_q, fallback, [], [], [], [], [], False, 0, retrieval_time
        
        # Build prompt with safety-first approach
        context_section = f"Equipment Information:\n"
        
        # Add safety information first if available
        safety_snippets = [s for i, s in enumerate(snippets) if search_types[i] == 'safety']
        if safety_snippets:
            context_section += f"\n[SAFETY INFORMATION]:\n"
            for i, snippet in enumerate(safety_snippets, 1):
                context_section += f"{snippet}\n"
        
        # Add operational information
        operational_snippets = [s for i, s in enumerate(snippets) if search_types[i] == 'operational']
        if operational_snippets:
            context_section += f"\n[OPERATIONAL INFORMATION]:\n"
            for i, snippet in enumerate(operational_snippets, 1):
                context_section += f"{snippet}\n"
        
        log_execution("📤 LLM Context", f"Safety chunks: {len(safety_snippets)}, Operational chunks: {len(operational_snippets)}, Total sent to LLM: {len(safety_snippets) + len(operational_snippets)}")
        
        system_prompt = get_system_prompt(property_id)
        full_prompt = (
            system_prompt + "\n\n" +
            f"Equipment Operator: {raw_question}\n\n" +
            context_section + "\n\n" +
            "Assistant: Based on the equipment information above, "
        )
        
        # Generate response
        stage1_start = time.time()
        
        # Try Groq first, fallback to Cortex if needed
        use_groq = st.session_state.config.get('use_groq', True)
        if use_groq and groq_client:
            try:
                # Convert prompt to Groq format
                messages = [
                    {"role": "system", "content": "You are a helpful, knowledgeable plant hire equipment expert. Answer only the most recent equipment inquiry using the provided information. Keep responses clear and professional, prioritize safety information when relevant, max 300 words. IMPORTANT: Always format lists as bullet points with * symbols and use proper line breaks for readability."},
                    {"role": "user", "content": f"Equipment Operator: {raw_question}\n\n{context_section}\n\nBased on the equipment information above, please answer the operator's question."}
                ]
                
                # Call Groq API
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
                # Fallback to Cortex - DO NOT try Groq again with different model
                log_execution("⚠️ Groq failed, using Cortex", str(e))
                stage1_start = time.time()  # Reset timer for Cortex
                
                # Switch to CORTEX_WH only for LLM generation
                session.sql("USE WAREHOUSE CORTEX_WH").collect()
                
                df = session.sql(
                    "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS response",
                    params=[FALLBACK_MODEL, full_prompt]
                ).collect()
                initial_response = df[0].RESPONSE.strip() if df else "I'm having trouble generating a response."
                log_execution("🤖 Cortex Fallback Response", f"{len(initial_response.split())} words", time.time() - stage1_start)
                # Switch back to RETRIEVAL warehouse
                session.sql("USE WAREHOUSE RETRIEVAL").collect()
            log_execution("🤖 LLM Response", f"{len(initial_response.split())} words", time.time() - stage1_start)
            
            # Switch back to RETRIEVAL warehouse
            session.sql("USE WAREHOUSE RETRIEVAL").collect()
        
        stage1_time = time.time() - stage1_start
        word_count = len(initial_response.split())
        
        log_execution("🤖 LLM Response", f"{word_count} words", stage1_time)
        
        # Format response with safety information first
        formatted_response = format_response_with_safety(initial_response, safety_snippets, operational_snippets, raw_question)
        
        return (enriched_q, formatted_response, snippets, chunk_idxs, paths, 
                similarities, search_types, False, word_count, retrieval_time)
        
    except Exception as e:
        log_execution("❌ Generation Error", str(e))
        return raw_question, "I'm experiencing technical difficulties. Please try again.", [], [], [], [], [], False, 0, 0

def format_main_response(response: str) -> str:
    """Format main response to ensure proper bullet points and line breaks."""
    # Look for patterns that indicate lists and convert them to bullet points
    lines = response.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            formatted_lines.append('')
            continue
            
        # Check if line contains list indicators using simple string operations
        if line and line[0].isdigit() and '. ' in line[:5]:  # Numbered list
            # Convert to bullet point - find first '. ' and replace
            dot_pos = line.find('. ')
            if dot_pos > 0:
                line = '* ' + line[dot_pos + 2:]
        elif line.startswith('*'):  # Already bullet point
            pass  # Keep as is
        elif line.startswith('-'):  # Dash list
            line = '* ' + line[1:].lstrip()
        elif ':' in line and any(keyword in line.lower() for keyword in ['include', 'possible', 'causes', 'reasons', 'steps', 'check']):
            # This might be a list header, keep as is
            pass
        else:
            # Check if this line contains multiple items separated by common patterns
            bullet_chars = ['•', '·', '▪', '▫']
            for char in bullet_chars:
                if char in line:
                    line = line.replace(char, '*')
                    break
        
        formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def format_response_with_safety(response: str, safety_snippets: list, operational_snippets: list, question: str) -> str:
    """Format response with safety information first and attention-grabbing emojis."""
    # Determine if we have relevant safety information
    has_relevant_safety = len(safety_snippets) > 0
    
    # Check if question indicates uncertainty or problems
    uncertainty_keywords = ['wrong', 'problem', 'issue', 'error', 'broken', 'not working', 'trouble', 'help', 'unsure', 'confused', 'what should', 'how do i']
    question_lower = question.lower()
    indicates_uncertainty = any(keyword in question_lower for keyword in uncertainty_keywords)
    
    formatted_parts = []
    
    # Add safety information first if available
    if has_relevant_safety:
        # Process safety information through separate LLM call
        processed_safety = process_safety_information(safety_snippets, question)
        if processed_safety:
            formatted_parts.append("🛡️ **SAFETY FIRST:**")
            formatted_parts.append(processed_safety)
            formatted_parts.append("")  # Empty line for spacing
    
    # Add general safety warning if no specific safety info but uncertainty indicated
    elif indicates_uncertainty and not has_relevant_safety:
        formatted_parts.append("🛡️ **SAFETY REMINDER:**")
        formatted_parts.append("⚠️ If you're unsure about equipment operation or experiencing issues, stop work immediately and contact your supervisor or safety officer.")
        formatted_parts.append("")  # Empty line for spacing
    
    # Add the main response
    formatted_parts.append(format_main_response(response))
    
    return "\n".join(formatted_parts)

def process_safety_information(safety_snippets: list, question: str) -> str:
    """Process safety information through LLM to extract relevant safety points."""
    try:
        # Combine safety snippets
        safety_content = "\n\n".join(safety_snippets)
        
        # Create focused system prompt for safety processing
        safety_system_prompt = "You are a safety expert for construction equipment. Your task is to extract ONLY relevant safety information from the provided content.\n\nIMPORTANT RULES:\n1. Extract ONLY safety-related information (warnings, hazards, protective measures, emergency procedures)\n2. IGNORE: warranty information, addresses, contact details, administrative procedures\n3. Focus on: operational safety, personal protective equipment, hazard warnings, emergency procedures\n4. Keep each safety point concise (1-2 sentences max)\n5. Use clear, direct language\n6. Maximum 3 safety points total\n\nFormat your response as bullet points with ⚠️ emoji, like:\n⚠️ [Safety point 1]\n⚠️ [Safety point 2]\n⚠️ [Safety point 3]\n\nIf no relevant safety information is found, respond with \"No relevant safety information found.\""

        # Create user prompt with context
        safety_user_prompt = f"""Question: {question}

Safety Content:
{safety_content}

Extract only the safety information that is relevant to the question or general equipment safety."""

        # Try Groq first
        use_groq = st.session_state.config.get('use_groq', True)
        if use_groq and groq_client:
            try:
                messages = [
                    {"role": "system", "content": safety_system_prompt},
                    {"role": "user", "content": safety_user_prompt}
                ]
                
                completion = groq_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.1,  # Lower temperature for more focused extraction
                    max_tokens=300,  # Increased from 150 to 300
                    top_p=0.9,
                    stream=False
                )
                
                safety_response = completion.choices[0].message.content.strip()
                log_execution("🛡️ Safety Processing", f"Groq safety extraction completed")
                
            except Exception as e:
                # Fallback to Cortex
                log_execution("⚠️ Safety processing Groq failed, using Cortex", str(e))
                safety_response = process_safety_with_cortex(safety_system_prompt, safety_user_prompt)
        else:
            # Use Cortex directly
            safety_response = process_safety_with_cortex(safety_system_prompt, safety_user_prompt)
        
        # Clean up response
        if safety_response and safety_response.lower() != "no relevant safety information found.":
            # Ensure proper formatting with line breaks
            formatted_safety = format_safety_response(safety_response)
            return formatted_safety
        else:
            return ""
            
    except Exception as e:
        log_execution("❌ Safety processing error", str(e))
        return ""

def process_safety_with_cortex(system_prompt: str, user_prompt: str) -> str:
    """Process safety information using Cortex as fallback."""
    try:
        # Switch to CORTEX_WH for LLM generation
        session.sql("USE WAREHOUSE CORTEX_WH").collect()
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        df = session.sql(
            "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS response",
            params=[FALLBACK_MODEL, full_prompt]
        ).collect()
        
        safety_response = df[0].RESPONSE.strip() if df else ""
        log_execution("🛡️ Safety Processing", f"Cortex safety extraction completed")
        
        # Switch back to RETRIEVAL warehouse
        session.sql("USE WAREHOUSE RETRIEVAL").collect()
        
        return safety_response
        
    except Exception as e:
        log_execution("❌ Cortex safety processing error", str(e))
        # Switch back to RETRIEVAL warehouse
        session.sql("USE WAREHOUSE RETRIEVAL").collect()
        return ""

def format_safety_response(safety_response: str) -> str:
    """Format safety response with proper line breaks and indentation."""
    # Split by lines and clean up
    lines = safety_response.strip().split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if line:
            # If line starts with ⚠️, keep it as is
            if line.startswith('⚠️'):
                formatted_lines.append(line)
            # If line doesn't start with ⚠️ but contains safety content, add ⚠️
            elif any(keyword in line.lower() for keyword in ['safety', 'danger', 'hazard', 'warning', 'caution', 'emergency', 'protective']):
                formatted_lines.append(f"⚠️ {line}")
            # Otherwise, skip non-safety lines
            else:
                continue
    
    # Join with proper line breaks
    return '\n'.join(formatted_lines)

# ——— Stream Response ———
def stream_response(response: str, placeholder):
    """Simulate streaming for better UX with mobile-safe rendering."""
    # For mobile compatibility, use simpler streaming approach
    # Check if this is a safety message
    if "⚠️" in response or "SAFETY" in response.upper():
        # For safety messages, display immediately without streaming to avoid parsing issues
        placeholder.empty()
        lines = response.split('\n')
        for i, line in enumerate(lines):
            if line.strip():
                placeholder.text(line)
            time.sleep(0.1)  # Brief pause for readability
    else:
        # For regular messages, use simple character streaming
        streamed_text = ""
        for char in response:
            streamed_text += char
            placeholder.empty()
            placeholder.text(streamed_text + "▌")
            time.sleep(0.01)
        
        # Final display without cursor
        placeholder.empty()
        placeholder.text(response)

# ——— Main App ———
def main():
    # Add comprehensive mobile compatibility fixes
    st.markdown("""
    <style>
        /* iOS Safari regex fix - disable problematic auto-linking */
        .stMarkdown a[href^="http"] {
            pointer-events: none !important;
            text-decoration: none !important;
            color: inherit !important;
        }
        
        /* Prevent URL pattern detection and regex issues */
        .stMarkdown {
            word-break: break-word;
            -webkit-user-select: text;
            user-select: text;
            /* Disable automatic text processing that might cause regex issues */
            -webkit-text-size-adjust: none;
            text-size-adjust: none;
        }
        
        /* Fix for iOS rendering issues */
        .element-container {
            -webkit-transform: translateZ(0);
            transform: translateZ(0);
        }
        
        /* Ensure proper text rendering on iOS */
        * {
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        
        /* Additional mobile fixes */
        .stChatMessage {
            -webkit-transform: translateZ(0);
            transform: translateZ(0);
        }
        
        /* Prevent Safari from trying to parse regex-like patterns */
        pre, code {
            -webkit-user-select: text;
            user-select: text;
            white-space: pre-wrap;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'property_id' not in st.session_state:
        st.session_state.property_id = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = None
    if 'message_counter' not in st.session_state:
        st.session_state.message_counter = 0
    if 'last_query_info' not in st.session_state:
        st.session_state.last_query_info = None
    
    # Sidebar
    with st.sidebar:
        st.write("📊 System Information")
        
        # Equipment switcher (keeping property_id field name)
        if st.session_state.property_id:
            if st.button("🔄 Switch Equipment", type="secondary"):
                # End current conversation
                if st.session_state.conversation_id:
                    conversation_logger.end_conversation(st.session_state.conversation_id)
                # Reset state
                st.session_state.property_id = None
                st.session_state.chat_history = []
                st.session_state.conversation_id = None
                st.session_state.message_counter = 0
                st.rerun()
        
        # Show current equipment
        if st.session_state.property_id:
            st.info(f"🚜 Equipment #{st.session_state.property_id}")
        
        # Last Query Performance (New section)
        if st.session_state.last_query_info:
            with st.expander("⏱️ Last Query Performance", expanded=True):
                timing = st.session_state.last_query_info['timing']
                
                # Display timing metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📥 Retrieval Time", f"{timing['retrieval']:.2f}s")
                with col2:
                    st.metric("🤖 LLM Time", f"{timing.get('llm_time', timing['total'] - timing['retrieval']):.2f}s")
                
                st.metric("⏱️ Total Response Time", f"{timing['total']:.2f}s")
                
                # Model used
                if 'model_used' in timing:
                    st.info(f"🔧 Model: {timing['model_used']}")
        
        # Retrieved Chunks (New section)
        if st.session_state.last_query_info and st.session_state.last_query_info.get('snippets'):
            with st.expander("📄 Retrieved Chunks", expanded=True):
                snippets = st.session_state.last_query_info['snippets']
                similarities = st.session_state.last_query_info['similarities']
                search_types = st.session_state.last_query_info['search_types']
                paths = st.session_state.last_query_info['paths']
                
                for i, (snippet, sim, search_type, path) in enumerate(zip(snippets, similarities, search_types, paths)):
                    # Color code by search type
                    if search_type == 'safety':
                        st.write(f"**🛡️ Chunk {i+1} - Safety ({search_type})**")
                        color = "#ff6b6b"  # Red for safety
                    else:
                        st.write(f"**🔧 Chunk {i+1} - Operational ({search_type})**")
                        color = "#4ecdc4"  # Teal for operational
                    
                    # Show similarity score with color - use st.write instead of markdown
                    st.write(f"Score: {sim:.3f}")
                    
                    # Show source file
                    st.text(f"📁 {path}")
                    
                    # Show snippet preview (truncated)
                    preview = snippet[:150] + "..." if len(snippet) > 150 else snippet
                    st.text_area(f"Content {i+1}", preview, height=80, key=f"chunk_{i}")
                    st.divider()
        
        # Execution Log (Improved)
        with st.expander("🐛 Execution Log", expanded=False):
            if st.button("Clear Logs"):
                st.session_state.execution_log = []
                st.rerun()
            
            if st.session_state.execution_log:
                # Display logs in reverse chronological order
                for log in reversed(st.session_state.execution_log[-20:]):  # Show last 20 logs
                    # Format log entry - use st.text instead of markdown for iOS
                    time_str = f"[{log['timestamp']}]"
                    step_str = log['step']
                    
                    # Display main log line
                    log_line = f"{time_str} {step_str}"
                    if log['timing']:
                        log_line += f" ({log['timing']})"
                    
                    st.text(log_line)
                    
                    # Display details if present
                    if log['details']:
                        st.text(f"  └─ {log['details']}")
            else:
                st.text("No logs yet")
        
        # Configuration section
        with st.expander("⚙️ Settings", expanded=False):
            st.session_state.config['enable_logging'] = st.checkbox(
                "Enable conversation logging", 
                value=st.session_state.config['enable_logging']
            )
            
            st.session_state.config['use_groq'] = st.checkbox(
                "Use Groq API (faster)", 
                value=st.session_state.config.get('use_groq', True),
                help="When enabled, uses Groq API for faster responses. Falls back to Cortex if unavailable."
            )
        
        # Performance metrics (Overall)
        with st.expander("📊 Overall Performance", expanded=False):
            metrics = monitor.get_dashboard_metrics()
            if 'status' not in metrics:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Avg Response", f"{metrics['avg_response_time']:.2f}s")
                    st.metric("Total Queries", metrics['total_requests'])
                    if metrics.get('avg_llm_time', 0) > 0:
                        st.metric("Avg LLM Time", f"{metrics['avg_llm_time']:.2f}s")
                with col2:
                    st.metric("Avg Retrieval", f"{metrics['avg_retrieval_time']:.2f}s")
                    st.metric("Recent Errors", metrics['recent_errors'])
                    if metrics.get('p95_response_time', 0) > 0:
                        st.metric("P95 Response", f"{metrics['p95_response_time']:.2f}s")
            else:
                st.info(metrics['status'])
        
        # Conversation history
        if st.session_state.property_id and st.session_state.config['enable_logging']:
            with st.expander("📜 Conversation History", expanded=False):
                history = conversation_logger.get_conversation_history(st.session_state.property_id, limit=5)
                if history:
                    for conv in history:
                        start_time = conv.get('START_TIME', 'Unknown')
                        status = conv.get('STATUS', 'Unknown')
                        conv_id = conv.get('CONVERSATION_ID', 'Unknown')
                        
                        if st.button(f"📅 {start_time} ({status})", key=f"conv_{conv_id}"):
                            messages = conversation_logger.get_conversation_messages(conv_id)
                            if messages:
                                st.write("**Conversation Messages:**")
                                for msg in messages:
                                    role = msg.get('ROLE', 'Unknown')
                                    content = msg.get('CONTENT', 'No content')
                                    st.write(f"**{role.title()}:** {content}")
                else:
                    st.text("No conversation history")
                    
    # Main interface
    if not st.session_state.property_id:
        # Equipment selection screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Use regular headers instead of HTML for iOS compatibility
            st.title("🚜 Plant Hire Equipment Assistant")
            st.subheader("Welcome! Let's get you connected to your equipment.")
            
            st.write("#### Enter Equipment ID")
            
            # Manual entry option
            manual_id = st.text_input("Equipment ID", placeholder="e.g., 2", value="2")
            if st.button("Connect to Equipment", type="primary", disabled=not manual_id):
                try:
                    eq_id = int(manual_id)
                    st.session_state.property_id = eq_id
                    # Start new conversation
                    if st.session_state.config['enable_logging']:
                        st.session_state.conversation_id = conversation_logger.start_conversation(
                            eq_id, st.session_state.session_id
                        )
                    st.rerun()
                except ValueError:
                    st.error("Please enter a valid equipment ID number")
    
    else:
        # Chat interface
        st.title("🚜 Plant Hire Equipment Assistant")
        
        # Welcome message for new conversations
        if not st.session_state.chat_history:
            # Use st.info for welcome message to avoid markdown issues
            welcome_msg = f"""Welcome! I'm your equipment assistant for Equipment #{st.session_state.property_id}.

I can help you with:
* 🛡️ Safety procedures and warnings
* 🔧 Operating instructions and controls
* 🔍 Troubleshooting common issues
* 📋 Maintenance requirements
* ⚠️ Emergency procedures

What would you like to know about your equipment?"""
            st.session_state.chat_history.append({"role": "assistant", "content": welcome_msg})
        
        # Display chat history with mobile-safe rendering
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                # Use st.text for safety-critical content to avoid markdown issues
                content = msg["content"]
                if "⚠️" in content or "SAFETY" in content.upper():
                    # For safety messages, use multiple st.text calls to avoid parsing issues
                    lines = content.split('\n')
                    for line in lines:
                        if line.strip():
                            st.text(line)
                        else:
                            st.text('')  # Empty line for spacing
                else:
                    # Use st.write for other content
                    st.write(content)
        
        # Chat input
        if prompt := st.chat_input("Ask about your equipment..."):
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                # Use st.text to avoid any markdown parsing issues with user input
                st.text(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                try:
                    # Show thinking indicator
                    with st.spinner("Checking equipment information..."):
                        start_time = time.time()
                        
                        # Get answer with dual retrieval
                        (enriched_q, response, snippets, chunk_idxs, paths, 
                         similarities, search_types, used_refinement, word_count, retrieval_time) = get_enhanced_answer(
                            st.session_state.chat_history, 
                            prompt, 
                            st.session_state.property_id
                        )
                        
                        total_time = time.time() - start_time
                    
                    # Stream the response
                    stream_response(response, message_placeholder)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.session_state.message_counter += 1
                    
                    # Log the conversation
                    if st.session_state.config['enable_logging'] and st.session_state.conversation_id:
                        # Log user message
                        conversation_logger.log_message(
                            st.session_state.conversation_id,
                            "user",
                            prompt,
                            metadata={"enriched": enriched_q},
                            property_id=st.session_state.property_id
                        )
                        
                        # Log assistant response
                        conversation_logger.log_message(
                            st.session_state.conversation_id,
                            "assistant",
                            response,
                            metadata={
                                "word_count": word_count,
                                "retrieval_time": retrieval_time,
                                "total_time": total_time,
                                "snippets_used": len(snippets),
                                "used_refinement": used_refinement
                            },
                            property_id=st.session_state.property_id
                        )
                    
                    # Store query info for debugging
                    st.session_state.last_query_info = {
                        "question": prompt,
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
                    
                    # Log performance metrics
                    monitor.log_request({
                        'latency': total_time,
                        'retrieval_time': retrieval_time,
                        'used_refinement': used_refinement,
                        'word_count': word_count
                    })
                
                except Exception as e:
                    error = ChatError(
                        error_type="generation_error",
                        user_message="I encountered an error while processing your request. Please try again.",
                        technical_details=str(e)
                    )
                    error.display()

if __name__ == "__main__":
    main()