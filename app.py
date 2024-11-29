from flask import Flask, request, jsonify, send_from_directory
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import os
from difflib import SequenceMatcher

app = Flask(__name__)

# Modelo de embeddings
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

# Banco de dados FAISS
dimension = 768  # Dimensão do modelo 'all-MiniLM-L6-v2'
index = faiss.IndexFlatL2(dimension)

# Armazenar mapeamento de índices para textos e PDFs
documents = []
doc_sources = []  # Armazena o nome do PDF para cada texto indexado

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

@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')


@app.route('/upload', methods=['POST'])
def upload_pdfs():
    # Verifica se múltiplos arquivos foram enviados
    if 'files' not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado."}), 400
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "Nenhum arquivo enviado."}), 400

    for pdf_file in files:
        try:
            text = extract_text_from_pdf(pdf_file)
            
            # Gera embeddings do texto
            sentences = text.split(".")  # Divide em sentenças (ajuste se necessário)
            embeddings = model.encode(sentences)
            
            # Indexa embeddings no FAISS
            index.add(embeddings)
            documents.extend(sentences)
            
            # Adiciona o nome do PDF à lista de fontes
            doc_sources.extend([pdf_file.filename] * len(sentences))
        
        except Exception as e:
            return jsonify({"error": f"Erro ao processar o arquivo {pdf_file.filename}: {str(e)}"}), 500

    return jsonify({"message": f"{len(files)} PDFs processados e indexados com sucesso."}), 200

@app.route('/query', methods=['POST'])
def query():
    # Recebe a pergunta do usuário
    if not documents:
        return jsonify({"error": "Forneça pelo menos 1 PDF antes de fazer perguntas"}), 400
    data = request.json
    query_text = data.get('query', '')
    if not query_text:
        return jsonify({"error": "Nenhuma consulta fornecida."}), 400
    
    # Gera embedding da consulta
    query_embedding = model.encode([query_text])
    
    # Busca no FAISS
    D, I = index.search(query_embedding, k=5)  # Retorna os 5 mais relevantes

    # Filtrar e preparar resultados
    min_score = 0.5  # Define o score mínimo para aceitar um resultado
    results = [
        {
            "score": float(D[0][i]),
            "text": documents[I[0][i]],
            "source": doc_sources[I[0][i]]
        }
        for i in range(len(I[0])) if D[0][i] > min_score
    ]
    
    # Reordenar resultados por similaridade textual
    results = rank_by_similarity(query_text, results)
    
    # Retorna os resultados ao cliente
    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(debug=True)