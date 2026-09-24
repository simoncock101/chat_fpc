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
.stApp {
    background-color: #f4f6f8;
}

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
}

.header p {
    font-size: 18px;
}

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
    color: #0b6623;
}

.response-box {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
    border-top: 5px solid #0b6623;
}

section[data-testid="stSidebar"] {
    background-color: #0d3b1e;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>⚽ Chat del Fútbol Profesional Colombiano</h1>
    <p>Consulta información sobre la historia, equipos, jugadores y datos del FPC</p>
</div>
""", unsafe_allow_html=True)

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

col1, col2, col3 = st.columns([1, 2, 1])

try:
    image = Image.open("nacional.jpg")
    col2.image(image, width=350)
except:
    col2.warning("No se pudo cargar la imagen nacional.jpg")

st.markdown("""
<div class="card">
<div class="card-title">🏟️ Conoce el fútbol profesional colombiano</div>
<p>
Carga un documento PDF y utiliza este chat para realizar preguntas
sobre la información contenida en él. El sistema buscará los fragmentos
más relacionados con tu pregunta y utilizará inteligencia artificial
para generar una respuesta.
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 🔐 Configuración")

ke = st.text_input(
    "Ingresa tu Clave de OpenAI",
    type="password",
    placeholder="sk-..."
)

if ke:
    os.environ["OPENAI_API_KEY"] = ke
    st.success("Clave ingresada correctamente")

if not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

st.markdown("### 📄 Documento")

pdf = st.file_uploader(
    "Carga aquí el documento PDF",
    type="pdf"
)

if pdf is not None and ke:

    pdf_reader = PdfReader(pdf)

    text = ""

    for page in pdf_reader.pages:
        extracted_text = page.extract_text()

        if extracted_text:
            text = text + extracted_text

    st.success("Documento cargado correctamente")

    col1, col2 = st.columns(2)

    col1.metric(
        "Caracteres encontrados",
        f"{len(text):,}"
    )

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

    embeddings = OpenAIEmbeddings()

    knowledge_base = FAISS.from_texts(
        chunks,
        embeddings
    )

    st.success("Base de conocimiento lista")

    st.markdown("## 💬 Pregúntale al documento")

    st.write(
        "Escribe una pregunta relacionada con el contenido del documento."
    )

    user_question = st.text_area(
        "Tu pregunta",
        placeholder="Ejemplo: ¿Cuál fue el primer campeón del fútbol profesional colombiano?",
        height=100
    )

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

        st.markdown("""
        <div class="response-box">
        <h3>⚽ Respuesta</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(response)

elif pdf is not None and not ke:

    st.warning(
        "⚠️ Primero debes ingresar tu clave de API de OpenAI."
    )

else:

    st.info(
        "📄 Carga un archivo PDF para comenzar a utilizar el chat."
    )
