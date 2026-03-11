"""
No Cross-Attention, Q vem do Decoder e K/V vêm do Encoder.
Não há máscara causal — o Decoder pode consultar toda a
sequência de entrada do Encoder livremente.

    Q  ←  decoder_state  @ W_Q
    K  ←  encoder_out    @ W_K
    V  ←  encoder_out    @ W_V

    CrossAttention = softmax( Q K^T / sqrt(d_k) ) V
    
Questão 3 da prova :)
"""

import numpy as np

## Hiperparâmetros
D_MODEL = 512
D_K     = 64
D_V     = 64


## Softmax
def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax numericamente estável no último eixo."""
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

## Cross Attention

## Matrizes de projeção globais (pesos aleatórios, simulando parâmetros treináveis)
np.random.seed(42)
W_Q = np.random.randn(D_MODEL, D_K) * 0.01   # projeção do Decoder  → Q
W_K = np.random.randn(D_MODEL, D_K) * 0.01   # projeção do Encoder  → K
W_V = np.random.randn(D_MODEL, D_V) * 0.01   # projeção do Encoder  → V


def cross_attention(encoder_out: np.ndarray,
                    decoder_state: np.ndarray) -> np.ndarray:
    """
    Atenção cruzada Encoder → Decoder.

    Parâmetros
    encoder_out   : (batch, seq_len_src, d_model)  — saída do Encoder
    decoder_state : (batch, seq_len_tgt, d_model)  — estado atual do Decoder

    Retorna
    context : (batch, seq_len_tgt, d_v)
        Cada posição do Decoder recebe um vetor de contexto que é uma
        combinação ponderada de TODOS os tokens do Encoder.
    """
    ## Q vem do Decoder — "o que estou buscando?"
    Q = decoder_state @ W_Q          # (batch, seq_tgt, d_k)

    ## K e V vêm do Encoder — "o que está disponível na memória?"
    K = encoder_out @ W_K            # (batch, seq_src, d_k)
    V = encoder_out @ W_V            # (batch, seq_src, d_v)

    ## Scores:  (batch, seq_tgt, seq_src)
    ## Cada posição do Decoder pontua TODAS as posições do Encoder
    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(D_K)

    ## Sem máscara causal — o Decoder consulta a fonte inteira
    weights = softmax(scores)        # (batch, seq_tgt, seq_src)

    ## Contexto: soma ponderada dos valores do Encoder
    context = weights @ V            # (batch, seq_tgt, d_v)
    return context, weights


## Smoke test

if __name__ == "__main__":
    BATCH         = 1
    SEQ_LEN_SRC   = 10   # frase em francês (Encoder)
    SEQ_LEN_TGT   = 4    # tokens já gerados em inglês (Decoder)

    ## Tensores fictícios conforme o pedido no enunciado
    encoder_output = np.random.randn(BATCH, SEQ_LEN_SRC, D_MODEL)
    decoder_state  = np.random.randn(BATCH, SEQ_LEN_TGT, D_MODEL)

    print(f"encoder_output shape : {encoder_output.shape}")
    print(f"decoder_state  shape : {decoder_state.shape}")

    context, weights = cross_attention(encoder_output, decoder_state)

    print(f"\nCross-Attention output shape : {context.shape}")
    print(f"  (esperado: ({BATCH}, {SEQ_LEN_TGT}, {D_V}))")

    print(f"\nPesos de atenção shape       : {weights.shape}")
    print(f"  (esperado: ({BATCH}, {SEQ_LEN_TGT}, {SEQ_LEN_SRC}))")

    ## Cada linha de weights deve somar 1.0  (distribuição de probabilidade)
    somas = weights[0].sum(axis=-1)
    print(f"\nSoma dos pesos por token do Decoder (deve ser ~1.0):")
    np.set_printoptions(precision=6, suppress=True)
    print(f"  {somas}")

    ok = np.allclose(somas, 1.0)
    print(f"  Todas somam 1.0? {'✓ SIM' if ok else '✗ NÃO'}")
    print("\n[Tarefa 2 concluída ✓]")
