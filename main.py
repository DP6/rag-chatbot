from model import RAG

rag = RAG(
    docs_dir='docs', # Nome do diretório onde estão os documentos
    n_retrievals=1, # Número de documentos retornados pela busca (int)  :   default=4
    chat_max_tokens=3097, # Número máximo de tokens que podem ser usados na memória do chat (int)  :   default=3097
    creativeness=1.2, # Quão criativa será a resposta (float 0-2)  :   default=0.7
)

print("\nDigite 'sair' para sair do programa.")
while True:
    question = str(input("Pergunta: "))
    if question == "sair":
        break
    answer = rag.ask(question)
    print('Resposta:', answer)