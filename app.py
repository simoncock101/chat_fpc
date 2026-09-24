import os
import platform
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

st.set_page_config(
    page_title="Chat Fútbol Profesional Colombiano",
    page_icon="⚽",
    layout="wide"
)

st.markdown("""
<style>

/* ================================
   FONDO GENERAL
================================ */

.stApp {
    background-color: #f4f6f8;
    color: #222222;
}


/* ================================
   ENCABEZADO
================================ */

.header {
    background: linear-gradient(135deg, #0b6623, #138a36);
    padding: 30px;
    border-radius: 18px;
    margin-bottom: 25px;
    color: white;
    text-align: center;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.15);
}

.header h1 {
    font-size: 42px;
    margin-bottom: 5px;
    color: white !important;
}

.header p {
    font-size: 18px;
    color: white !important;
}


/* ================================
   TARJETA DE INFORMACIÓN
================================ */

.card {
    background-color: white;
    padding: 22px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
    border-left: 5px solid #0b6623;
}

.card-title {
    font-size: 22px;
    font-weight: bold;
    color: #0b6623 !important;
}

.card p {
    color: #333333 !important;
}


/* ================================
   TEXTO GENERAL
================================ */

.stApp p {
    color: #333333;
}

.stApp label {
    color: #333333 !important;
}

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5 {
    color: #222222;
}


/* ================================
   RESPUESTA
================================ */

.response-box {
    background-color: #172019;
    padding: 22px 25px 12px 25px;
    border-radius: 15px 15px 0px 0px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.18);
    border-top: 5px solid #19a74a;
    margin-top: 25px;
}

.response-box h3 {
    color: #4ade80 !important;
    font-size: 22px;
    margin: 0;
}

.answer-text {
    background-color: #172019;
    color: white !important;
    padding: 5px 25px 25px 25px;
    border-radius: 0px 0px 15px 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.18);
    font-size: 17px;
    line-height: 1.7;
}

.answer-text p {
    color: white !important;
}

.answer-text li {
    color: white !important;
}

.answer-text strong {
    color: #4ade80 !important;
}


/* ================================
   SIDEBAR
================================ */

section[data-testid="stSidebar"] {
    background-color: #0d3b1e;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}


/* ================================
   CAMPO DE PREGUNTA
================================ */

.stTextArea textarea {
    background-color: white !important;
    color: #222222 !important;
    border: 2px solid #0b6623 !important;
    border-radius: 10px;
}

.stTextArea textarea::placeholder {
    color: #777777 !important;
}


/* ================================
   CAMPO API
================================ */

.stTextInput input {
    background-color: white !important;
    color: #222222 !important;
    border: 1px solid #0b6623 !important;
    border-radius: 8px;
}

.stTextInput input::placeholder {
    color: #777777 !important;
}


/* ================================
   CARGADOR DE ARCHIVOS
================================ */

[data-testid="stFileUploader"] {
    background-color: white;
    border-radius: 10px;
    padding: 10px;
    border: 1px solid #dddddd;
}

[data-testid="stFileUploader"] * {
    color: #333333 !important;
}


/* ================================
   MÉTRICAS
================================ */

div[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
}

div[data-testid="stMetricLabel"] {
    color: #555555 !important;
}

div[data-testid="stMetricValue"] {
    color: #222222 !important;
}


/* ================================
   MENSAJES DE STREAMLIT
================================ */

div[data-testid="stAlert"] {
    border-radius: 10px;
}

div[data-testid="stAlert"] p {
    color: #333333 !important;
}


/* ================================
   BOTONES
================================ */

.stButton button {
    background-color: #0b6623;
    color: white;
    border: none;
    border-radius: 8px;
}

.stButton button:hover {
    background-color: #138a36;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# =================================
# ENCABEZADO
# =================================

st.markdown("""
<div class="header">
    <h1>⚽ Chat del Fútbol Profesional Colombiano</h1>
    <p>Consulta información sobre la historia, equipos, jugadores y datos del FPC</p>
</div>
""", unsafe_allow_html=True)


# =================================
# SIDEBAR
# =================================

st.sidebar.markdown("## ⚽ FPC Chat")

st.sidebar.markdown("---")

st.sidebar.markdown("""
### Sobre este agente

Este agente utiliza inteligencia artificial para responder preguntas
basadas en el contenido del documento PDF que cargues.

Puedes preguntarle sobre:

- 🏆 Campeonatos
- ⚽ Equipos
- 👤 Jugadores
- 📊 Estadísticas
- 📚 Historia del fútbol colombiano
""")

st.sidebar.markdown("---")

st.sidebar.caption("Aplicación desarrollada con Streamlit")
st.sidebar.caption("Python: " + platform.python_version())


# =================================
# IMAGEN
# =================================

col1, col2, col3 = st.columns([1, 2, 1])

try:

    image = Image.open("nacional.jpg")

    col2.image(
        image,
        width=350
    )

except:

    col2.warning(
        "No se pudo cargar la imagen nacional.jpg"
    )


# =================================
# INFORMACIÓN
# =================================

st.markdown("""
<div class="card">

<div class="card-title">
🏟️ Conoce el fútbol profesional colombiano
</div>

<p>
Carga un documento PDF y utiliza este chat para realizar preguntas
sobre la información contenida en él. El sistema buscará los fragmentos
más relacionados con tu pregunta y utilizará inteligencia artificial
para generar una respuesta.
</p>

</div>
""", unsafe_allow_html=True)


# =================================
# API
# =================================

st.markdown("### 🔐 Configuración")

ke = st.text_input(
    "Ingresa tu Clave de OpenAI",
    type="password",
    placeholder="sk-..."
)

if ke:

    os.environ["OPENAI_API_KEY"] = ke

    st.success(
        "Clave ingresada correctamente"
    )

if not ke:

    st.warning(
        "Por favor ingresa tu clave de API de OpenAI para continuar"
    )


# =================================
# PDF
# =================================

st.markdown("### 📄 Documento")

pdf = st.file_uploader(
    "Carga aquí el documento PDF",
    type="pdf"
)


# =================================
# PROCESAMIENTO
# =================================

if pdf is not None and ke:

    pdf_reader = PdfReader(pdf)

    text = ""

    for page in pdf_reader.pages:

        extracted_text = page.extract_text()

        if extracted_text:

            text = text + extracted_text


    st.success(
        "Documento cargado correctamente"
    )


    # =================================
    # MÉTRICAS
    # =================================

    col1, col2 = st.columns(2)

    col1.metric(
        "Caracteres encontrados",
        f"{len(text):,}"
    )


    # =================================
    # DIVIDIR TEXTO
    # =================================

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=20,
        length_function=len
    )

    chunks = text_splitter.split_text(text)


    col2.metric(
        "Fragmentos creados",
        len(chunks)
    )


    st.markdown("---")


    # =================================
    # BASE DE CONOCIMIENTO
    # =================================

    embeddings = OpenAIEmbeddings()

    knowledge_base = FAISS.from_texts(
        chunks,
        embeddings
    )


    st.success(
        "Base de conocimiento lista"
    )


    # =================================
    # PREGUNTA
    # =================================

    st.markdown(
        "## 💬 Pregúntale al documento"
    )

    st.write(
        "Escribe una pregunta relacionada con el contenido del documento."
    )


    user_question = st.text_area(
        "Tu pregunta",
        placeholder="Ejemplo: ¿Cuál fue el primer campeón del fútbol profesional colombiano?",
        height=100
    )


    # =================================
    # RESPUESTA
    # =================================

    if user_question:

        docs = knowledge_base.similarity_search(
            user_question
        )


        llm = OpenAI(
            temperature=0,
            model_name="gpt-4o-mini-2024-07-18"
        )


        chain = load_qa_chain(
            llm,
            chain_type="stuff"
        )


        response = chain.run(
            input_documents=docs,
            question=user_question
        )


        # TÍTULO DE RESPUESTA

        st.markdown("""
<div class="response-box">
<h3>⚽ Respuesta del agente</h3>
</div>
""", unsafe_allow_html=True)


        # CONTENIDO DE RESPUESTA

        st.markdown(
            '<div class="answer-text">' +
            response.replace("\n", "<br>") +
            '</div>',
            unsafe_allow_html=True
        )


# =================================
# MENSAJES
# =================================

elif pdf is not None and not ke:

    st.warning(
        "⚠️ Primero debes ingresar tu clave de API de OpenAI."
    )


else:

    st.info(
        "📄 Carga un archivo PDF para comenzar a utilizar el chat."
    )
