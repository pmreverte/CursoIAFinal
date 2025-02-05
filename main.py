"""
This script is a Streamlit application for configuring and interacting with an AI agent that processes PDF files and responds to user queries.
Modules:
    logging: For logging messages.
    streamlit as st: For creating the Streamlit web application.
    tempfile: For creating temporary files.
    json: For handling JSON data.
    numpy as np: For numerical operations.
    tiktoken: For tokenizing text.
    tools.explore_pdf: Custom module for handling PDF files.
    tools.search_embedding: Custom module for creating contextual texts and calculating embeddings.
    tools.nuevo_agente: Custom module for creating a new agent.
Constants:
    TEXT_INPUT_BANNER: A constant string for the user input prompt in the chat.
Functions:
    open_and_read_pdf: Opens and reads the content of a PDF file.
    get_pages_and_texts: Extracts pages and texts from a PDF file.
    concatenate_documents: Concatenates documents based on token size.
    create_contextual_texts_per_pdf: Creates contextual texts with overlap for each PDF.
    update_comparision_data: Updates comparison data with embeddings.
    calculate_embeddings: Calculates embeddings for a given text.
    nuevo_agente: Creates a new agent with a given prompt.
Streamlit App:
    The app has two main screens:
    1. Initial Configuration Screen:
        - Allows users to upload PDF files and define the initial prompt for the agent.
        - Processes the uploaded PDF files to extract and tokenize text.
        - Calculates embeddings for the text blocks.
        - Saves the embeddings data in session state.
        - Instantiates the agent with the selected prompt.
    2. Main Interaction Screen:
        - Allows users to interact with the agent through a chat interface.
        - Displays user messages and agent responses in the chat.
"""
import logging
import streamlit as st
import tempfile
import json
import numpy as np
import tiktoken

# Importar funciones necesarias (asegúrate de que estos módulos estén en tu proyecto)
from tools.explore_pdf import open_and_read_pdf, get_pages_and_texts, concatenate_documents
from tools.search_embedding import create_contextual_texts_per_pdf, update_comparision_data, calculate_embeddings
from tools.nuevo_agente import nuevo_agente

# Constante para el input de usuario en el chat
TEXT_INPUT_BANNER = "¿Sobre qué quieres preguntar?"

# =============================================================================
# PANTALLA INICIAL DE CONFIGURACIÓN
# =============================================================================

# Utilizamos una variable en session_state para controlar si la configuración ya se completó.
if "config_done" not in st.session_state:
    st.session_state.config_done = False

if not st.session_state.config_done:
    st.title("Configuración Inicial del Agente")
    st.write("Sube los archivos PDF y define el prompt inicial del agente.")
    
    # Subida de archivos PDF (puede ser múltiple)
    uploaded_files = st.file_uploader("Selecciona archivos PDF", type=["pdf"], accept_multiple_files=True)
    
    # Input para el prompt inicial. Se establece un valor por defecto.
    prompt_inicial = st.text_input(
        "Introduce el prompt inicial",
        value="Eres un asistente que ayuda. La respuesta siempre la devolverás en el idioma en el que te hablen. Debes de ser breve y conciso en tus respuestas."
    )
    
    # Botón para aceptar la configuración
    if st.button("Aceptar configuración"):
        # Guardamos el prompt en session_state y definimos el historial inicial del chat
        st.session_state.system_prompt = prompt_inicial
        st.session_state.chat_history = [{"role": "system", "content": st.session_state.system_prompt}]
        
        # -------------------------------
        # Procesamiento de archivos PDF
        # -------------------------------
        if uploaded_files is not None and len(uploaded_files) > 0:
            st.write(f"Se han subido {len(uploaded_files)} archivos PDF.")
            pages_and_texts = []
            filtered_pages_and_texts = []
            
            # Iteramos sobre cada archivo subido
            for uploaded_file in uploaded_files:
                st.write(f"Procesando: {uploaded_file.name}")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name
                
                # Abrimos y leemos el PDF
                pdf_content = open_and_read_pdf(temp_file_path)
                pages_and_texts.append(pdf_content)
                filtered_pages_and_texts.append(get_pages_and_texts(pdf_content))
            
            # Creamos textos contextuales con solapamiento (overlap) para cada PDF
            overlap_ratio = 0.2
            overlapped_texts_per_pdf = create_contextual_texts_per_pdf(filtered_pages_and_texts, overlap_ratio=overlap_ratio)
            
            # Calculamos la cantidad de tokens de cada bloque y concatenamos documentos si es necesario
            tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            input_texts_and_tokens = [
                {
                    'file_name': text['file_name'],
                    'text': texto,
                    'token_size': len(tokenizer.encode(texto))
                } 
                for text in overlapped_texts_per_pdf for texto in text['texts']
            ]
            max_tokens = 1000
            concatenated_input_texts_and_tokens = concatenate_documents(input_texts_and_tokens, max_tokens)
            
            # Calculamos las embeddings para cada bloque de texto
            text_and_embeddings = [
                {
                    'block_id': block_id,
                    'text': text['text'],
                    'embeddings': calculate_embeddings(text['text'])
                } 
                for block_id, text in enumerate(concatenated_input_texts_and_tokens)
            ]
            
            # Opcional: guardar las embeddings en session_state o en un archivo
            st.session_state.embeddings_data = text_and_embeddings
            
            # Convertir las embeddings a un formato adecuado (por ejemplo, numpy array) y actualizar los datos de comparación
            embedings_list = [{
                "text": entry["text"],
                "embeddings": np.array(entry["embeddings"])
            } for entry in text_and_embeddings]
            update_comparision_data(embedings_list)
        else:
            st.write("No se subieron archivos PDF. Puedes continuar sin ellos.")
        
        # ------------------------------------------------------------------------------
        # Instanciar el agente con el prompt seleccionado
        # ------------------------------------------------------------------------------
        st.session_state.agent = nuevo_agente(st.session_state.system_prompt)
        
        # Indicamos que la configuración se completó y recargamos la app para mostrar la siguiente pantalla
        st.session_state.config_done = True
        st.success("¡Configuración completada! Ahora puedes interactuar con el agente.")
        st.rerun()

# =============================================================================
# PANTALLA PRINCIPAL DE INTERACCIÓN CON EL AGENTE
# =============================================================================
else:
    st.title("Interacción con el Agente")
    
    # Inicializar variable para evitar procesar prompts repetidos
    if "last_user_prompt" not in st.session_state:
        st.session_state.last_user_prompt = None

    # Input del usuario a través de la nueva función de chat (disponible en versiones recientes de Streamlit)
    user_prompt = st.chat_input(TEXT_INPUT_BANNER)
    
    if user_prompt and user_prompt != st.session_state.last_user_prompt:
        st.session_state.last_user_prompt = user_prompt  # Guarda el último prompt para evitar duplicados
        
        # Mostrar el mensaje del usuario en el chat
        with st.chat_message("user"):
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})
            st.markdown(user_prompt)
        
        # Consultar al agente y mostrar la respuesta
        with st.chat_message("assistant"):
            response = st.session_state.agent.query(user_prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.markdown(response)