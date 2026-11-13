# 📚 rag-pt-eval

RAG em **português** com foco no que normalmente falta: **medir se a recuperação funciona**.
Em vez de "parece bom", o projeto reporta **Hit@k** e **MRR** num conjunto de perguntas com gabarito e compara configurações de chunking.

**Domínio:** [ex.: documentos públicos da construção civil — normas, relatórios de entidades do setor, etc. Escolha um domínio e seja específico.]

## Resultados

Avaliação em [N] perguntas escritas por mim sobre [N] documentos ([links ou descrição]):

| chunk | overlap | n | hit@5 | MRR |
|---|---|---|---|---|
| 400 | 80 | [n] | [x] | [x] |
| 800 | 150 | [n] | [x] | [x] |
| 1200 | 200 | [n] | [x] | [x] |

**Conclusões** *(escreva o que realmente observou)*:
1. [ex.: chunks menores melhoraram Hit@5, mas pioraram respostas que dependem de contexto longo]
2. [ex.: perguntas com siglas falharam; abordagem híbrida (BM25 + embeddings) seria o próximo passo]

## Como rodar

```bash
git clone https://github.com/lucasguimaraes-web/rag-pt-eval && cd rag-pt-eval
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) coloque documentos PÚBLICOS (.pdf/.txt/.md) em data/docs/
# 2) copie data/eval_questions.example.jsonl para data/eval_questions.jsonl e escreva ~30 perguntas reais
python src/eval.py --questions data/eval_questions.jsonl --sweep

# 3) para gerar respostas (precisa de chave de API):
export ANTHROPIC_API_KEY=...
python src/rag.py "sua pergunta aqui"
```

## Como funciona
`documentos → chunks → embeddings multilíngues → busca por similaridade de cosseno → LLM responde só com o contexto e cita as fontes`.
Sem banco vetorial de propósito: o código fica pequeno e legível (veja `src/rag.py`).

## Limitações
- A avaliação mede **recuperação**, não a qualidade final da resposta do LLM.
- Conjunto de avaliação pequeno e escrito por uma pessoa só.
- [adicione as suas]

## Próximos passos
- Busca híbrida (BM25 + embeddings) e reranking.
- Avaliar fidelidade da resposta (a resposta está sustentada pelos trechos?).
- Interface simples para perguntas.

Projeto pessoal, feito por **Lucas** — estudante de Engenharia de Software no Inatel e pesquisador de IC no **WAI Lab**.
