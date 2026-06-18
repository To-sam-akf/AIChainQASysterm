"""Evaluation platform primitives for AIQASYS."""

from aika.eval.dataset import DEFAULT_QA_BENCHMARK, EvalCase, load_eval_cases
from aika.eval.feedback import FeedbackStore
from aika.eval.rag_dataset import (
    DEFAULT_RAG_RETRIEVAL_BENCHMARK,
    RagRetrievalCase,
    load_rag_retrieval_cases,
)
from aika.eval.rag_runner import run_rag_retrieval_benchmark
from aika.eval.runner import run_qa_benchmark
from aika.eval.store import EvalRunStore

__all__ = [
    "DEFAULT_QA_BENCHMARK",
    "DEFAULT_RAG_RETRIEVAL_BENCHMARK",
    "EvalCase",
    "EvalRunStore",
    "FeedbackStore",
    "RagRetrievalCase",
    "load_eval_cases",
    "load_rag_retrieval_cases",
    "run_qa_benchmark",
    "run_rag_retrieval_benchmark",
]
