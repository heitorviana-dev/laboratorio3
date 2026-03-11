"""
O Decoder gera tokens um a um. A cada passo:
  1. O estado atual (tokens gerados até agora) passa pelo Decoder (mock).
  2. O vetor final (d_model) é projetado para o vocabulário (W_out).
  3. Softmax emite uma distribuição de probabilidades.
  4. argmax seleciona o próximo token.
  5. O token é adicionado à sequência — e o loop recomeça.
  6. Ao gerar <EOS>, o loop para e a frase final é impressa.
"""

import numpy as np

## Hiperparâmetros
D_MODEL    = 512
VOCAB_SIZE = 10_000
MAX_STEPS  = 20      # limite de segurança para evitar loop infinito

np.random.seed(7)

# Vocabulário fictício: índices 0-9997 são palavras; 9998 = <EOS>
EOS_TOKEN  = "<EOS>"
EOS_IDX    = VOCAB_SIZE - 2      # índice 9998

# Monta um vocabulário fictício de palavras genéricas
vocab = {i: f"word_{i}" for i in range(VOCAB_SIZE)}
vocab[EOS_IDX] = EOS_TOKEN

# Projeção linear final: d_model → vocab_size  (simulando a cabeça de linguagem)
W_out = np.random.randn(D_MODEL, VOCAB_SIZE) * 0.01

## Softmax

def softmax(x: np.ndarray) -> np.ndarray:
    x_shifted = x - np.max(x)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)

## Mock do Decoder

def generate_next_token(current_sequence: list,
                        encoder_out: np.ndarray) -> np.ndarray:
    """
    Mock do Decoder: simula a passagem pelo bloco completo e
    retorna um vetor de probabilidades de tamanho VOCAB_SIZE.

    Parâmetros
    current_sequence : list[str]   tokens gerados até agora
    encoder_out      : np.ndarray  (1, seq_src, d_model)

    Retorna
    probs : np.ndarray  shape (VOCAB_SIZE,)
    """
    seq_len = len(current_sequence)

    # Simula o estado do Decoder como vetor aleatório baseado no contexto
    decoder_hidden = np.random.randn(D_MODEL) * 0.1

    # Adiciona influência do encoder_out (média sobre a sequência fonte)
    encoder_context = encoder_out[0].mean(axis=0)  
    decoder_hidden += encoder_context * 0.05

    # Projeção linear → logits sobre o vocabulário
    logits = decoder_hidden @ W_out               

    # Força o <EOS> no passo 5 para demonstrar a parada do loop
    if seq_len >= 5:
        logits[EOS_IDX] = logits.max() + 10.0
        
    probs = softmax(logits)
    return probs


## Loop de inferência auto-regressivo

def autoregressive_loop(encoder_out: np.ndarray,
                        start_token: str = "<START>") -> list:
    """
    Gera tokens iterativamente até encontrar <EOS> ou atingir MAX_STEPS.

    Parâmetros
    encoder_out : np.ndarray  (1, seq_src, d_model)
    start_token : str

    Retorna
    sequence : list[str]  frase gerada (sem <START>)
    """
    sequence = [start_token]
    step     = 0

    print("=== Loop de Inferência Auto-Regressivo ===\n")
    print(f"  Início : {sequence}")

    while step < MAX_STEPS:
        # Passo 1 — gera distribuição de probabilidades
        probs = generate_next_token(sequence, encoder_out)

        # Passo 2 — argmax seleciona o próximo token
        next_idx   = int(np.argmax(probs))
        next_token = vocab[next_idx]

        # Passo 3 — adiciona à sequência de contexto
        sequence.append(next_token)
        step += 1

        print(f"  Passo {step:02d} | idx={next_idx:5d} | token='{next_token}' "
              f"| prob={probs[next_idx]:.6f}")

        # Passo 4 — para se gerou <EOS>
        if next_token == EOS_TOKEN:
            print(f"\n  <EOS> detectado. Geração encerrada.")
            break
    else:
        print(f"\n  Limite de {MAX_STEPS} passos atingido.")

    return sequence

## Execução

if __name__ == "__main__":
    BATCH       = 1
    SEQ_LEN_SRC = 10

    # Saída fictícia do Encoder (substitui o resultado real do Lab 2)
    encoder_output = np.random.randn(BATCH, SEQ_LEN_SRC, D_MODEL)
    print(f"encoder_output shape : {encoder_output.shape}\n")

    sequence = autoregressive_loop(encoder_output)

    # Remove o token <START> da exibição final
    frase_gerada = [t for t in sequence if t != "<START>"]
    print(f"\n=== Frase Final Gerada ===")
    print(f"  {' '.join(frase_gerada)}")
    print(f"  ({len(frase_gerada)} tokens, incluindo <EOS>)")
    print("\n[Tarefa 3 concluída ✓]")
