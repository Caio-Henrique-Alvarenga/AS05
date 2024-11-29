import streamlit as st
from streamlit_js_eval import streamlit_js_eval
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
from difflib import SequenceMatcher

# Configuração inicial do Streamlit
st.set_page_config(page_title="Assistente Conversacional", layout="wide")

# Modelo de embeddings
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1') # Modelo para perguntas e respostas

# Inicialização do estado da sessão
if 'index' not in st.session_state:
    dimension = 768  # Dimensão do modelo 'multi-qa-mpnet-base-dot-v1'
    st.session_state.index = faiss.IndexFlatL2(dimension)
    st.session_state.documents = []
    st.session_state.doc_sources = []
    st.session_state.processed = False  # Flag para verificar se os PDFs foram processados

def extract_text_from_pdf(pdf_file):
    """
    Extrai o texto de um PDF usando pdfplumber.
    """
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    except Exception as e:
        raise RuntimeError(f"Erro ao processar o PDF: {str(e)}")
    return text

def rank_by_similarity(query, results):
    """
    Ordena os resultados com base na similaridade textual entre a consulta e os trechos retornados.
    """
    def similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()

    return sorted(results, key=lambda x: similarity(query, x['text']), reverse=True)

def process_pdfs(uploaded_files):
    """
    Processa os arquivos PDF enviados pelo usuário.
    """
    for pdf_file in uploaded_files:
        try:
            # Extrai texto do PDF
            text = extract_text_from_pdf(pdf_file)
            
            # Divide o texto em sentenças
            sentences = text.split(".")  # Ajuste a divisão conforme necessário
            
            # Gera embeddings para as sentenças
            embeddings = model.encode(sentences)
            
            # Indexa embeddings no FAISS
            st.session_state.index.add(embeddings)
            st.session_state.documents.extend(sentences)
            
            # Armazena o nome do PDF associado a cada sentença
            st.session_state.doc_sources.extend([pdf_file.name] * len(sentences))
        except Exception as e:
            st.error(f"Erro ao processar o arquivo {pdf_file.name}: {e}")
    
    # Marca os PDFs como processados
    st.session_state.processed = True

def search_query(query_text):
    """
    Busca no índice FAISS com base na consulta do usuário.
    """
    if not st.session_state.documents:
        st.warning("Por favor, envie PDFs antes de fazer perguntas.")
        return []

    # Gera embedding da consulta
    query_embedding = model.encode([query_text])
    
    # Busca no FAISS
    D, I = st.session_state.index.search(query_embedding, k=5)  # Retorna os 5 mais relevantes
    
    # Filtrar e preparar os resultados
    min_score = 0.5  # Define um score mínimo
    results = [
        {
            "score": float(D[0][i]),
            "text": st.session_state.documents[I[0][i]],
            "source": st.session_state.doc_sources[I[0][i]]
        }
        for i in range(len(I[0])) if D[0][i] > min_score
    ]
    
    # Reordenar por similaridade textual
    return rank_by_similarity(query_text, results)

# Interface do Streamlit
st.text("AS05 - Caio Henrique Alvarenga Gonçalves")
st.title("Assistente Conversacional com PDFs")

# Seção para upload de PDFs
st.header("Envio de PDFs")
uploaded_files = st.file_uploader(
    "Faça upload dos arquivos PDF que deseja processar", 
    type=["pdf"], 
    accept_multiple_files=True
)

# Processa PDFs apenas se eles não foram processados ainda
if uploaded_files and not st.session_state.processed:
    with st.spinner("Processando arquivos..."):
        process_pdfs(uploaded_files)
    st.success(f"{len(uploaded_files)} arquivo(s) processado(s) com sucesso!")

if st.button("Recomeçar"):
    streamlit_js_eval(js_expressions="parent.window.location.reload()")

# Seção para consultas
st.header("Perguntas")
query_text = st.text_input("Digite sua pergunta e o assistente tentará responder com sentenças dos pdfs fornecidos:")

if query_text:
    with st.spinner("Buscando resposta..."):
        results = search_query(query_text)
        if results:
            st.subheader("Resultados")
            for result in results:
                st.write(f"**Fonte:** {result['source']}")
                st.write(f"**Trecho:** {result['text']}")
                st.write(f"**Relevância:** {result['score']:.2f}")
                st.write("---")
        else:
            st.warning("Nenhuma resposta encontrada para sua pergunta.")
