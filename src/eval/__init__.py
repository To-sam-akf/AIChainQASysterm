"""Evaluation platform primitives for AIQASYS."""

from src.eval.dataset import DEFAULT_QA_BENCHMARK, EvalCase, load_eval_cases
from src.eval.feedback import FeedbackStore
from src.eval.rag_dataset import (
    DEFAULT_RAG_RETRIEVAL_BENCHMARK,
    RagRetrievalCase,
    load_rag_retrieval_cases,
)
from src.eval.rag_runner import run_rag_retrieval_benchmark
from src.eval.runner import run_qa_benchmark
from src.eval.store import EvalRunStore

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
