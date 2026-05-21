"""
Run ARGOS evaluation on ReDial test dataset.

Generates TSV output files in output/agros/REDIAL/ with the same format as
output/REDIAL/ (legacy GRACE).

Columns: id | recommend_item | summarized_conversation | recommend_movie_list | movie_candidate_list | recall

Pre-requisites (run once before this script):
  1. Neo4j running and populated with ReDial KG:
       cd backend && python scripts/graph_builder.py
  2. Vector embeddings built:
       cd backend && python scripts/generate_embedding.py --dataset redial
  3. .env configured with GEMINI_API_KEY, OPENAI_API_KEY, AWS credentials

Usage (from project root):
  cd backend
  python scripts/run_argos_redial_eval.py
  python scripts/run_argos_redial_eval.py --sample 925
  python scripts/run_argos_redial_eval.py --sample 925 --concurrency 3 --model decoupled
  python scripts/run_argos_redial_eval.py --top-ks 1 10 50

Options:
  --sample N        Number of test dialogs to evaluate (default: all 4623)
  --concurrency C   Max parallel API calls (default: 5)
  --model MODEL     Reranker: 'decoupled', 'cohere', or 'llm' (default: decoupled)
  --n-retrieve N    Retrieval candidates per stream (default: 20)
  --top-ks K...     Space-separated K values for Recall@K (default: 1 5 10 50)
  --seed S          Random seed when sampling (default: 42)
  --resume          Skip dialogs whose id already appears in existing output files
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Add backend/ root to Python path (script lives in backend/scripts/)
# ---------------------------------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from domain.agent.critic import CriticAgent
from domain.agent.relaxation import RelaxationAgent
from domain.generation.service import GenerationService
from domain.reranking.service import RerankingService
from domain.retrieval.service import RetrievalService
from shared.utils.logger import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = BACKEND_ROOT.parent
REDIAL_TEST_DIALOGS = PROJECT_ROOT / "dataset/REDIAL/processed/dialog_data/test_data.json"
OUTPUT_DIR = PROJECT_ROOT / "output/agros/REDIAL"
MODEL_LABEL = "gemini-2.0-flash"

_STOP_WORDS = {
    "Initiator", "Respondent",
    "RECOMMENDER", "SEEKER", "Hi", "There", "What", "types", "movies",
    "like", "watch", "Yes", "No", "Thanks", "Thank", "you", "It", "The",
    "For", "How", "Do", "Does", "Did", "Are", "Is", "Was", "Have", "Has",
    "Can", "Could", "Would", "Should", "Will", "Okay", "Sure", "Great",
    "Good", "Nice", "Cool",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_test_dialogs(path: Path, sample_size: Optional[int], seed: int) -> List[Dict]:
    import random
    with open(path, "r", encoding="utf-8") as f:
        all_dialogs = json.load(f)
    if sample_size is None or sample_size >= len(all_dialogs):
        return all_dialogs
    random.seed(seed)
    return random.sample(all_dialogs, sample_size)


def dialog_to_context(dialog_turns: List[str]) -> str:
    return "\n".join(dialog_turns)


def extract_liked_movies(dialog_turns: List[str]) -> List[str]:
    text = " ".join(dialog_turns)
    mentions = re.findall(r"[A-Z][a-zA-Z\s]+(?:\([0-9]{4}\))?", text)
    return [
        m.strip()
        for m in mentions
        if m.strip() not in _STOP_WORDS and len(m.strip()) > 3
    ][:5]


def standardize_title(title: str) -> str:
    t = re.sub(r"\s*\(\d{4}\)", "", title)
    t = re.sub(r"[^\w\s]", "", t)
    return t.strip().lower()


def compute_recall(recommendations: List[str], ground_truth: List[str], k: int) -> float:
    top_k_norm = {standardize_title(r) for r in recommendations[:k]}
    gt_norm = [standardize_title(g) for g in ground_truth]
    if not gt_norm:
        return 0.0
    return sum(1 for g in gt_norm if g in top_k_norm) / len(gt_norm)


def load_done_ids(output_paths: Dict[int, Path]) -> set:
    done_sets = []
    for path in output_paths.values():
        if path.exists():
            try:
                df = pd.read_csv(path, sep="\t", usecols=["id"])
                done_sets.append(set(df["id"].astype(str).tolist()))
            except Exception:
                done_sets.append(set())
        else:
            return set()
    if not done_sets:
        return set()
    return done_sets[0].intersection(*done_sets[1:])


def init_output_files(output_dir: Path, sample_size: int, top_ks: List[int]) -> Dict[int, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[int, Path] = {}
    header = ["id", "recommend_item", "summarized_conversation",
              "recommend_movie_list", "movie_candidate_list", "recall"]
    for k in top_ks:
        path = output_dir / f"{MODEL_LABEL}_recall@{k}_{sample_size}sample.tsv"
        if not path.exists():
            pd.DataFrame(columns=header).to_csv(path, sep="\t", index=False)
        paths[k] = path
    return paths


def append_row(path: Path, row: Dict) -> None:
    pd.DataFrame([row]).to_csv(path, sep="\t", index=False, header=False, mode="a")


# ---------------------------------------------------------------------------
# Core ARGOS pipeline
# ---------------------------------------------------------------------------

async def process_dialog(
    conv: Dict,
    idx: int,
    generation_svc: GenerationService,
    retrieval_svc: RetrievalService,
    reranking_svc: RerankingService,
    critic_agent: CriticAgent,
    relaxation_agent: RelaxationAgent,
    n_retrieve: int,
    model: str,
    max_top_k: int,
) -> Optional[Dict]:
    dialog_turns: List[str] = conv.get("dialog", [])
    raw_target = conv.get("target", "")
    target: str = raw_target if isinstance(raw_target, str) else raw_target[0]
    context: str = dialog_to_context(dialog_turns)
    liked_movies: List[str] = extract_liked_movies(dialog_turns)

    try:
        user_pref_obj = await generation_svc.summarize_conversation(context)
        user_preferences: str = user_pref_obj.user_preferences

        retrieval_result = await retrieval_svc.retrieve_movies(
            user_preferences=user_preferences,
            liked_movies=liked_movies,
            n=n_retrieve,
            dynamic_weights=user_pref_obj.dynamic_weights.model_dump(),
            semantic_queries=user_pref_obj.semantic_queries,
            hard_constraints=user_pref_obj.hard_constraints,
            genres=user_pref_obj.genres,
        )
        candidates: List[Dict] = retrieval_result.get("combined", [])

        MAX_RELAX = 1
        current_candidates = candidates
        current_pref_obj = user_pref_obj
        filtered = candidates

        for attempt in range(MAX_RELAX + 1):
            critic_result = await critic_agent.filter_candidates(
                user_preferences=current_pref_obj.user_preferences,
                candidates=current_candidates,
                hard_constraints=current_pref_obj.hard_constraints,
            )
            filtered = critic_result.get("movies", current_candidates) or current_candidates

            if not critic_result.get("requires_relaxation", False) or attempt >= MAX_RELAX:
                break

            relaxed = await relaxation_agent.run(current_pref_obj, critic_result.get("reasoning", ""))
            re_result = await retrieval_svc.retrieve_movies(
                user_preferences=relaxed.user_preferences,
                liked_movies=liked_movies,
                n=n_retrieve,
                dynamic_weights=relaxed.dynamic_weights.model_dump(),
                hard_constraints=relaxed.hard_constraints,
                semantic_queries=relaxed.semantic_queries,
                genres=relaxed.genres,
            )
            current_candidates = re_result.get("combined", []) or current_candidates
            current_pref_obj = relaxed

        reranking_result = await reranking_svc.rerank_movies(
            user_preferences=current_pref_obj.user_preferences,
            candidates=filtered,
            conversation=context,
            top_k=max_top_k,
            model=model,
        )
        reranked: List[Dict] = reranking_result.get("movies", [])

        return {
            "idx": idx,
            "conv_id": conv.get("conv_id", str(idx)),
            "target": target,
            "user_preferences": user_preferences,
            "rec_titles": [m["title"] for m in reranked],
            "candidate_titles": [m["title"] for m in candidates],
        }

    except Exception as exc:
        logger.error(f"[{idx}] Failed: {exc}", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(
    sample_size: Optional[int],
    concurrency: int,
    model: str,
    n_retrieve: int,
    top_ks: List[int],
    seed: int,
    resume: bool,
) -> None:
    dialogs = load_test_dialogs(REDIAL_TEST_DIALOGS, sample_size, seed)
    actual_n = len(dialogs)
    print(f"Loaded {actual_n} ReDial test dialogs from {REDIAL_TEST_DIALOGS.name}")

    output_paths = init_output_files(OUTPUT_DIR, actual_n, top_ks)

    done_ids: set = set()
    if resume:
        done_ids = load_done_ids(output_paths)
        print(f"Resume mode: {len(done_ids)} dialogs already completed, skipping.")

    todo = [
        (i, conv) for i, conv in enumerate(dialogs)
        if str(i) not in done_ids
    ]
    print(f"To process: {len(todo)} dialogs  |  concurrency={concurrency}  |  model={model}")

    generation_svc = GenerationService()
    retrieval_svc = RetrievalService()
    reranking_svc = RerankingService()
    critic_agent = CriticAgent()
    relaxation_agent = RelaxationAgent()

    max_k = max(top_ks)
    semaphore = asyncio.Semaphore(concurrency)

    async def sem_task(idx: int, conv: Dict) -> Optional[Dict]:
        async with semaphore:
            return await process_dialog(
                conv, idx,
                generation_svc, retrieval_svc, reranking_svc,
                critic_agent, relaxation_agent,
                n_retrieve, model, max_k,
            )

    pending_results: List[Optional[Dict]] = []
    tasks = [sem_task(idx, conv) for idx, conv in todo]
    with tqdm(total=len(tasks), desc="ARGOS evaluation") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            pending_results.append(result)
            pbar.update(1)

    pending_results.sort(key=lambda r: r["idx"] if r else -1)

    success = 0
    for result in pending_results:
        if result is None:
            continue
        ground_truth = [result["target"]] if result["target"] else []
        for k in top_ks:
            recall = compute_recall(result["rec_titles"], ground_truth, k)
            append_row(output_paths[k], {
                "id": result["idx"],
                "recommend_item": "|".join(ground_truth),
                "summarized_conversation": result["user_preferences"],
                "recommend_movie_list": "|".join(result["rec_titles"][:k]),
                "movie_candidate_list": "|".join(result["candidate_titles"]),
                "recall": recall,
            })
        success += 1

    print(f"\n{'='*60}")
    print(f"Completed: {success}/{len(todo)} dialogs")
    print(f"{'='*60}")
    for k in top_ks:
        df = pd.read_csv(output_paths[k], sep="\t")
        avg = df["recall"].mean() if len(df) > 0 else 0.0
        print(f"  Recall@{k:<3d}: {avg:.4f}  ({len(df)} rows)  → {output_paths[k].name}")
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ARGOS multi-agent evaluation on ReDial test dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sample", type=int, default=None,
                        help="Dialogs to evaluate (default: all 4623)")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Max parallel API calls (default: 5)")
    parser.add_argument("--model", type=str, default="decoupled", choices=["cohere", "llm", "decoupled"],
                        help="Reranker model: 'decoupled', 'cohere', or 'llm' (default: decoupled)")
    parser.add_argument("--n-retrieve", type=int, default=20,
                        help="Retrieval candidates per stream (default: 20)")
    parser.add_argument("--top-ks", type=int, nargs="+", default=[1, 5, 10, 50],
                        help="K values for Recall@K (default: 1 5 10 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip dialogs already present in existing output files")
    args = parser.parse_args()

    asyncio.run(main(
        sample_size=args.sample,
        concurrency=args.concurrency,
        model=args.model,
        n_retrieve=args.n_retrieve,
        top_ks=sorted(set(args.top_ks)),
        seed=args.seed,
        resume=args.resume,
    ))
