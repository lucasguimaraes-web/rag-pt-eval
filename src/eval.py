"""Avalia a RECUPERAÇÃO (Hit@k e MRR) e compara configurações de chunking.

Formato do arquivo de avaliação (JSONL), uma pergunta por linha:
    {"question": "...", "source": "nome_do_arquivo.pdf", "must_contain": "trecho esperado (opcional)"}

Uso:
    python src/eval.py --questions data/eval_questions.jsonl --k 5
    python src/eval.py --questions data/eval_questions.jsonl --sweep   # compara tamanhos de chunk
"""
import argparse
import json

from rag import Index


def evaluate(index: Index, rows: list[dict], k: int) -> dict:
    hits, rr = 0, 0.0
    for r in rows:
        results = index.search(r["question"], k)
        rank = None
        for i, (chunk, _) in enumerate(results, start=1):
            ok_source = chunk.source == r["source"]
            ok_text = r.get("must_contain", "").lower() in chunk.text.lower()
            if ok_source and ok_text:
                rank = i
                break
        if rank:
            hits += 1
            rr += 1.0 / rank
    n = len(rows)
    return {"n": n, f"hit@{k}": hits / n, "mrr": rr / n}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", required=True)
    p.add_argument("--docs", default="data/docs")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--sweep", action="store_true", help="compara tamanhos de chunk")
    args = p.parse_args()

    rows = [json.loads(line) for line in open(args.questions, encoding="utf-8") if line.strip()]
    configs = [(800, 150)] if not args.sweep else [(400, 80), (800, 150), (1200, 200)]

    print("| chunk | overlap | n | hit@%d | MRR |\n|---|---|---|---|---|" % args.k)
    for size, overlap in configs:
        res = evaluate(Index(args.docs, size, overlap), rows, args.k)
        print(f"| {size} | {overlap} | {res['n']} | {res[f'hit@{args.k}']:.3f} | {res['mrr']:.3f} |")


if __name__ == "__main__":
    main()
