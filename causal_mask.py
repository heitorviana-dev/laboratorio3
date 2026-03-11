"""
Laboratório 3 - Implementando o Decoder
Disciplina: Tópicos em IA

Impede que a posição i atenda à posição i+1 durante o treinamento,
injetando -inf nos scores antes do Softmax.

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k)  +  M ) V
"""

import numpy as np

D_MODEL = 64
D_K     = 64

## Softmax
def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax numericamente estável no último eixo."""
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

## Criação da máscara causal
def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Retorna matriz [seq_len, seq_len]:
      - triangular inferior + diagonal  →  0
      - triangular superior             →  -inf

    Exemplo para seq_len=4:
      [[ 0, -inf, -inf, -inf],
       [ 0,    0, -inf, -inf],
       [ 0,    0,    0, -inf],
       [ 0,    0,    0,    0]]
    """
    ## np.triu retorna a parte superior; k=1 exclui a diagonal
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    mask = np.where(mask == 1, -np.inf, 0.0)
    return mask


if __name__ == "__main__":
    np.random.seed(42)

    SEQ_LEN = 5
    BATCH   = 1

    ## Matrizes fictícias Q e K  →  (batch, seq, d_k)
    Q = np.random.randn(BATCH, SEQ_LEN, D_K)
    K = np.random.randn(BATCH, SEQ_LEN, D_K)

    ## Scores sem máscara
    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(D_K)

    ## Máscara causal
    M = create_causal_mask(SEQ_LEN)
    print("=== Máscara Causal M ===")
    print(M)

    ## Scores mascarados  →  posições futuras recebem -inf
    scores_masked = scores + M

    ## Softmax: -inf  →  0.0
    weights = softmax(scores_masked)

    print("\n=== Pesos de Atenção após Softmax (batch 0) ===")
    np.set_printoptions(precision=4, suppress=True)
    print(weights[0])

    ## Verificação: toda posição acima da diagonal deve ser 0.0
    upper = weights[0][np.triu(np.ones((SEQ_LEN, SEQ_LEN), dtype=bool), k=1)]
    tudo_zero = np.allclose(upper, 0.0)
    print(f"\nProbabilidades futuras são todas 0.0? {'✓ SIM' if tudo_zero else '✗ NÃO'}")
    print("\n[Tarefa 1 concluída ✓]")
