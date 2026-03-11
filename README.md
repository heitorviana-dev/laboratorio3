# Transformer Decoder — From Scratch

**Disciplina:** Tópicos em Inteligência Artificial  
**Professor:** Prof. Dimmy Magalhães  
**Instituição:** iCEV - Instituto de Ensino Superior

Implementação dos blocos matemáticos centrais do Decoder do Transformer,
baseado em *"Attention Is All You Need"* (Vaswani et al., 2017),
usando apenas `Python 3`, `numpy` e `pandas`.

---

## Estrutura do Projeto

```
transformer-decoder/
├── causal_mask.py          # Tarefa 1 — Máscara Causal (Look-Ahead Mask)
├── cross_attention.py      # Tarefa 2 — Cross-Attention Encoder-Decoder
├── autoregressive_loop.py  # Tarefa 3 — Loop de Inferência Auto-Regressivo
└── README.md
```

---

## Como Rodar

### Pré-requisitos

```bash
pip install numpy
```

### Executar cada tarefa individualmente

```bash
# Tarefa 1 — Máscara Causal
python causal_mask.py

# Tarefa 2 — Cross-Attention
python cross_attention.py

# Tarefa 3 — Loop Auto-Regressivo
python autoregressive_loop.py
```

---

## Arquitetura Implementada

```
Encoder Output  ──────────────────────────┐
                                           │  K, V
Tokens gerados  →  Masked Self-Attention  →  Cross-Attention  →  FFN
    (Q)              (+ Causal Mask)            (Q do Decoder)
                                                      │
                                              Projeção Linear
                                              (d_model → vocab)
                                                      │
                                                   Softmax
                                                      │
                                                   argmax → próximo token
```

---

## Componentes Matemáticos

### Tarefa 1 — Máscara Causal

```
M[i][j] = 0     se j <= i   (pode ver)
M[i][j] = -inf  se j >  i   (bloqueado)

Attention(Q, K, V) = softmax( Q K^T / √d_k  +  M ) V
```
O `-inf` vira `0.0` após o Softmax, zerando qualquer atenção a posições futuras.

### Tarefa 2 — Cross-Attention

```
Q  ←  decoder_state @ W_Q     (o que o Decoder busca)
K  ←  encoder_out   @ W_K     (chaves da memória do Encoder)
V  ←  encoder_out   @ W_V     (valores da memória do Encoder)

CrossAttention = softmax( Q K^T / √d_k ) V    (sem máscara)
```
Sem máscara causal — o Decoder consulta toda a sequência fonte livremente.

### Tarefa 3 — Loop Auto-Regressivo

```python
sequence = ["<START>"]
while True:
    probs      = generate_next_token(sequence, encoder_out)
    next_token = vocab[argmax(probs)]
    sequence.append(next_token)
    if next_token == "<EOS>":
        break
```

---

## Nota de Integridade Acadêmica

Claude (Anthropic) foi consultado como ferramenta auxiliar para revisão de
estrutura de código e sintaxe NumPy, conforme permitido pelo Contrato Pedagógico.
A lógica, implementação matemática e decisões de arquitetura são de autoria do aluno.
