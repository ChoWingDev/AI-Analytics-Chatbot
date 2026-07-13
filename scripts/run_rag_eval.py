"""
scripts/run_rag_eval.py
-----------------------
RAG evaluation with Ragas.

The SQL half is graded by comparing result sets against ground-truth queries
(scripts/run_eval.py). That does not work for RAG: there is no single correct
string, so you cannot diff the answer. Instead we grade three things an LLM
judge CAN assess:

  faithfulness        Is every claim in the answer supported by the retrieved
                      context? This is the hallucination metric -- a confident
                      answer that the documents do not support scores 0.
  answer_relevancy    Does the answer actually address the question asked, or
                      does it wander?
  context_precision   Did the retriever surface USEFUL chunks, or did it bury
                      the relevant one under noise? This grades retrieval, not
                      generation -- so when a score is bad, you know which half
                      to fix.

Everything runs on the same HuggingFace token as the rest of the system: Ragas
talks to HF through its OpenAI-compatible endpoint, and embeddings run locally.

Usage:
    python scripts/run_rag_eval.py
    python scripts/run_rag_eval.py --limit 2
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from openai import AsyncOpenAI  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings  # noqa: E402
from ragas.metrics.collections import (  # noqa: E402
    Faithfulness,
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
)

from src.rag.parsing import load_parsed_documents, load_and_parse_pdfs, save_parsed_documents  # noqa: E402
from src.rag.vectorstore import create_vectorstore, load_vectorstore, rebuild_child_docs  # noqa: E402
from src.rag.retrieval import create_retriever  # noqa: E402
from src.rag.chain import build_rag_chain  # noqa: E402
from src.rag.config import CHROMA_DIR, LLM_MODEL_ID, EMBEDDING_MODEL  # noqa: E402

HF_OPENAI_BASE_URL = "https://router.huggingface.co/v1"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# RRF fuses two retrievers, so it can hand back ~10 chunks. Feeding all of them to
# a 7B judge that must emit a strict JSON schema makes it fail to parse. Judge the
# top-k the LLM actually leans on; this is what the answer was generated from.
JUDGE_TOP_K = 5

# Questions answerable from the loaded corpus (Aritzia / Zara annual reports and
# the European ecommerce benchmark report).
EVAL_QUESTIONS = [
    "What was Aritzia's eCommerce net revenue in fiscal 2025 and how did it grow?",
    "What percentage of online shopping carts are abandoned, and why?",
    "What technology or ecommerce platform investments did Aritzia make?",
    "What cybersecurity or data privacy risks do these companies disclose?",
    "What are the main ecommerce trends reported for the European market?",
]


def build_rag():
    docs = load_parsed_documents()
    if docs is None:
        docs = load_and_parse_pdfs()
        if docs:
            save_parsed_documents(docs)
    if not docs:
        sys.exit("[error] No report documents found. Add PDFs to data/reports/.")

    if os.path.exists(CHROMA_DIR):
        vectorstore = load_vectorstore()
        child_docs = rebuild_child_docs(docs)
    else:
        vectorstore, child_docs = create_vectorstore(docs)
    return vectorstore, child_docs


JUDGE_ATTEMPTS = 3


async def _score_one(metric, name, sample):
    """
    Score a single metric, retrying transient judge failures.

    Two things bite here:
      * Metrics take DIFFERENT arguments. answer_relevancy compares the answer to
        the question only (it never sees the contexts), so passing
        retrieved_contexts to it is a TypeError.
      * The judge produces structured output via `instructor`. With a lot of
        context text, a 7B model sometimes fails to emit a valid schema and
        instructor gives up ("<failed_attempts>"). That is judge flakiness, not a
        RAG failure, so retry it rather than recording a false zero.
    """
    kwargs = {"user_input": sample["question"], "response": sample["answer"]}
    if name != "answer_relevancy":
        kwargs["retrieved_contexts"] = sample["contexts"]

    last = None
    for attempt in range(JUDGE_ATTEMPTS):
        try:
            result = await metric.ascore(**kwargs)
            return float(result.value), ""
        except TypeError:
            raise  # a real coding error — don't mask it behind retries
        except Exception as e:
            last = e
            await asyncio.sleep(2 * (attempt + 1))
    return None, str(last)[:80]


async def score_all(samples, llm, embeddings):
    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecisionWithoutReference(llm=llm),
    }

    rows = []
    for i, s in enumerate(samples, 1):
        print(f"  scoring [{i}/{len(samples)}] {s['question'][:60]}...")
        row = {"question": s["question"], "answer": s["answer"],
               "n_contexts": len(s["contexts"])}
        for name, metric in metrics.items():
            score, err = await _score_one(metric, name, s)
            row[name] = score
            if score is None:
                print(f"    [warn] {name} unscored after {JUDGE_ATTEMPTS} attempts: {err}")
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline with Ragas.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions.")
    args = parser.parse_args()

    questions = EVAL_QUESTIONS[: args.limit] if args.limit else EVAL_QUESTIONS

    if not os.getenv("HF_TOKEN"):
        sys.exit("[error] HF_TOKEN is not set.")

    print("Building RAG pipeline...")
    vectorstore, child_docs = build_rag()

    # Generate an answer + capture the exact contexts the retriever surfaced.
    print(f"\nAnswering {len(questions)} questions...")
    samples = []
    for i, q in enumerate(questions, 1):
        print(f"  [{i}/{len(questions)}] {q[:60]}...")
        retriever = create_retriever(vectorstore, child_docs)
        docs = retriever.invoke(q)
        answer = build_rag_chain(retriever).invoke(q)
        samples.append({
            "question": q,
            "answer": answer,
            "contexts": [d.page_content for d in docs[:JUDGE_TOP_K]],
        })

    # Ragas judge: HF through its OpenAI-compatible endpoint; embeddings local.
    print("\nScoring with Ragas...")
    client = AsyncOpenAI(base_url=HF_OPENAI_BASE_URL, api_key=os.getenv("HF_TOKEN"))
    llm = llm_factory(LLM_MODEL_ID, provider="openai", client=client)
    embeddings = RagasHFEmbeddings(model=EMBEDDING_MODEL)

    rows = asyncio.run(score_all(samples, llm, embeddings))
    df = pd.DataFrame(rows)

    metrics = ["faithfulness", "answer_relevancy", "context_precision"]
    print("\n" + "=" * 62)
    print("RAG EVALUATION (Ragas)")
    print("=" * 62)
    print(f"Questions: {len(df)}")
    for m in metrics:
        if m in df and df[m].notna().any():
            print(f"{m:<20} {df[m].mean():.3f}   (scored {int(df[m].notna().sum())}/{len(df)})")

    print("\nPer question:")
    for _, r in df.iterrows():
        scores = "  ".join(
            f"{m.split('_')[0]}={r[m]:.2f}" if pd.notna(r.get(m)) else f"{m.split('_')[0]}=n/a"
            for m in metrics
        )
        print(f"  {scores}   {r['question'][:52]}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    csv_path = OUTPUT_DIR / "rag_evaluation_result.csv"
    json_path = OUTPUT_DIR / "rag_evaluation_summary.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({
        "questions": len(df),
        **{m: (float(df[m].mean()) if m in df and df[m].notna().any() else None) for m in metrics},
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
