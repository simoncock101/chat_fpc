```python
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

# ---------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ---------------------------------------------------------

st.set_page_config(
    page_title="Chat Fútbol Profesional Colombiano",
    page_icon="⚽",
    layout="wide"
)

# ---------------------------------------------------------
# ESTILOS VISUALES
# ---------------------------------------------------------

st.markdown("""
<style>

    /* Fondo general */
    .stApp {
        background-color: #f4f6f8;
    }

    /* Encabezado principal */
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
        font-weight: 700;
    }

    .header p {
        font-size: 18px;
        margin-top: 5px;
        opacity: 0.9;
    }

    /* Tarjetas */
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
        margin-bottom: 8px;
    }

    /* Caja de respuesta */
    .response-box {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
        border-top: 5px solid #0b6623;
        margin-top: 15px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0d3b1e;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Botones */
    .stButton > button {
        background-color: #0b6623;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 25px;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #138a36;
        color: white;
    }

    /* Separador */
    .divider {
        height: 2px;
        background-color: #d9d9d9;
        margin: 25px 0;
    }

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# ENCABEZADO
# ---------------------------------------------------------

st.markdown("""
<div class="header">
    <h1>⚽ Chat del Fútbol Profesional Colombiano</h1>
    <p>Consulta información sobre la historia, equipos, jugadores y datos del FPC</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.markdown("## ⚽ FPC Chat")

    st.markdown("---")

    st.markdown("""
    ### Sobre este agente

    Este agente utiliza inteligencia artificial para responder
    preguntas basadas en el contenido del documento PDF que cargues.

    Puedes preguntarle sobre:

    - 🏆 Campeonatos
    - ⚽ Equipos
    - 👤 Jugadores
    - 📊 Estadísticas
    - 📚 Historia del fútbol colombiano
    """)

    st.markdown("---")

    st.caption("Aplicación desarrollada con Streamlit")
    st.caption("Python: " + platform.python_version())

# ---------------------------------------------------------
# IMAGEN
# ---------------------------------------------------------

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    try:
        image = Image.open('nacional.jpg')
        st.image(image, width=350)
    except Exception as e:
        st.warning(f"No se pudo cargar la imagen: {e}")

# ---------------------------------------------------------
# INFORMACIÓN
# ---------------------------------------------------------

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

# ---------------------------------------------------------
# API KEY
# ---------------------------------------------------------

st.markdown("### 🔐 Configuración")

ke = st.text_input(
    "Ingresa tu Clave de OpenAI",
    type="password",
    placeholder="sk-..."
)

if ke:
    os.environ['OPENAI_API_KEY'] = ke
    st.success("Clave ingresada correctamente")
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# ---------------------------------------------------------
# CARGAR PDF
# ---------------------------------------------------------

st.markdown("### 📄 Documento")

pdf = st.file_uploader(
    "Carga aquí el documento PDF",
    type="pdf"
)

# ---------------------------------------------------------
# PROCESAMIENTO DEL PDF
# ---------------------------------------------------------

if pdf is not None and ke:

    try:

        with st.spinner("Procesando documento..."):

            # Extraer texto
            pdf_reader = PdfReader(pdf)

            text = ""

            for page in pdf_reader.pages:
                extracted_text = page.extract_text()

                if extracted_text:
                    text += extracted_text

        st.success("Documento cargado correctamente")

        # Información del documento
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Caracteres encontrados",
                f"{len(text):,}"
            )

        # -------------------------------------------------
        # DIVIDIR TEXTO
        # -------------------------------------------------

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )

        chunks = text_splitter.split_text(text)

        with col2:
            st.metric(
                "Fragmentos creados",
                len(chunks)
            )

        st.markdown("---")

        # -------------------------------------------------
        # CREAR BASE DE CONOCIMIENTO
        # -------------------------------------------------

        with st.spinner("Creando base de conocimiento..."):

            embeddings = OpenAIEmbeddings()

            knowledge_base = FAISS.from_texts(
                chunks,
                embeddings
            )

        st.success("Base de conocimiento lista")

        # -------------------------------------------------
        # CHAT
        # -------------------------------------------------

        st.markdown("## 💬 Pregúntale al documento")

        st.write(
            "Escribe una pregunta relacionada con el contenido "
            "del documento."
        )

        user_question = st.text_area(
            "Tu pregunta",
            placeholder="Ejemplo: ¿Cuál fue el primer campeón del fútbol profesional colombiano?",
            height=100
        )

        # -------------------------------------------------
        # PROCESAR PREGUNTA
        # -------------------------------------------------

        if user_question:

            with st.spinner("Buscando información..."):

                docs = knowledge_base.similarity_search(
                    user_question
                )

                # Modelo
                llm = OpenAI(
                    temperature=0,
                    model_name="gpt-4o-mini-2024-07-18"
                )

                # Cadena de preguntas
                chain = load_qa_chain(
                    llm,
                    chain_type="stuff"
                )

                # Generar respuesta
                response = chain.run(
                    input_documents=docs,
                    question=user_question
                )

            # -------------------------------------------------
            # RESPUESTA
            # -------------------------------------------------

            st.markdown("""
            <div class="response-box">
                <h3>⚽ Respuesta</h3>
            """, unsafe_allow_html=True)

            st.markdown(response)

            st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# MENSAJES DE ESTADO
# ---------------------------------------------------------

elif pdf is not None and not ke:

    st.warning(
        "⚠️ Primero debes ingresar tu clave de API de OpenAI."
    )

else:

    st.info(
        "📄 Carga un archivo PDF para comenzar a utilizar el chat."
    )
```

