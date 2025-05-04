import fitz  # para ler PDF
import numpy as np
import faiss  # para busca vetorial
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ETAPA 1: Ler o conteúdo do PDF
def carregar_pdf(caminho_arquivo):
    doc = fitz.open(caminho_arquivo)
    texto = ""
    for pagina in doc:
        texto += pagina.get_text()
    return texto

# ETAPA 2: Dividir o texto em partes menores
def dividir_texto_em_blocos(texto, tamanho_maximo=200):  # agora usa no máx. 200 palavras
    palavras = texto.split()
    blocos = []
    bloco = []

    for palavra in palavras:
        bloco.append(palavra)
        if len(bloco) >= tamanho_maximo:
            blocos.append(' '.join(bloco))
            bloco = []

    if bloco:
        blocos.append(' '.join(bloco))

    return blocos

# ETAPA 3: Criar vetores (embeddings) com IA
def gerar_embeddings(lista_textos, modelo):
    return modelo.encode(lista_textos)

# ETAPA 4: Criar o índice de busca
def criar_indice_faiss(embeddings):
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# Buscar os blocos mais parecidos com a pergunta
def buscar_resposta(pergunta, modelo, index, blocos):
    vetor_pergunta = modelo.encode([pergunta]).astype('float32')
    distancias, indices = index.search(vetor_pergunta, 1)  # apenas o bloco mais relevante
    return blocos[indices[0][0]]

# ETAPA 5: Gerar a resposta com IA (usando só 1 bloco)
def responder_com_qa(pergunta, contexto, qa_pipeline):
    resposta = qa_pipeline(question=pergunta, context=contexto)
    
    # Se a confiança for baixa, não responde
    if resposta['score'] < 0.3:
        return " Desculpe, não consegui encontrar uma boa resposta."
    else:
        return resposta['answer']

# PROGRAMA PRINCIPAL
if __name__ == "__main__":
    caminho_pdf = "Artigo1.pdf"  

    print(" Lendo o PDF...")
    texto = carregar_pdf(caminho_pdf)

    print(" Dividindo em blocos...")
    blocos = dividir_texto_em_blocos(texto)

    print(" Gerando embeddings...")
    modelo = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = gerar_embeddings(blocos, modelo)

    print(" Criando índice de busca...")
    index = criar_indice_faiss(embeddings)

    print(" Carregando modelo de perguntas e respostas...")
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

    # Chat interativo
    while True:
        pergunta = input("\n Digite sua pergunta (ou 'sair'): ")
        if pergunta.lower() == "sair":
            print(" Encerrando...")
            break

        print(" Procurando resposta...")
        bloco_relevante = buscar_resposta(pergunta, modelo, index, blocos)
        resposta = responder_com_qa(pergunta, bloco_relevante, qa_pipeline)

        print(f"\n Resposta: {resposta}")
