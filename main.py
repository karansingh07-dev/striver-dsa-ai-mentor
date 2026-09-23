import argparse
import sys
from pathlib import Path

from pipeline import LecturePipeline
from chunker import TranscriptChunker
from embedder import EmbeddingGenerator
from vector_store import QdrantVectorStore
from rag_engine import RAGPipeline
from retrieval.hybrid_search import HybridRetriever
from retrieval.pipeline import RetrievalPipeline
from evaluation.runner import run_evaluation
from config import DEFAULT_WHISPER_MODEL, DEFAULT_TOP_K, YTRAG_MIN_SCORE


def print_single_result(result):
    """Display a single lecture transcription summary."""

    print("\n" + "=" * 50)
    print(" TRANSCRIPTION SUMMARY")
    print("=" * 50)

    print(f"Title:          {result['title']}")
    print(f"Video ID:       {result['video_id']}")
    print(
        f"Language:       {result['language']} "
        f"(confidence: {result['language_probability'] * 100:.1f}%)"
    )
    print(f"Duration:       {result['duration']} seconds")
    print(f"Total Segments: {result['total_segments']}")
    print(
        f"Time Taken:     "
        f"{result['transcription_time_seconds']} seconds"
    )

    print("\n--- First 3 Timestamped Segments ---")

    for seg in result["segments"][:3]:
        print(
            f"[{seg['start_time']} -> {seg['end_time']}] "
            f"{seg['text']}"
        )

        print(
            f"   Clickable Link: "
            f"{seg['youtube_url']}"
        )

    transcript_path = (
        Path("data/transcripts")
        / f"{result['video_id']}.json"
    )

    print(
        f"\nFull JSON Transcript stored at: "
        f"{transcript_path.resolve()}"
    )


def print_search_results(query: str, results: list):
    """Display semantic search results formatted cleanly."""
    print("\n" + "=" * 50)
    print("SEMANTIC SEARCH")
    print("=" * 50)
    print(f"\nQuery:\n{query}\n")

    if not results:
        print("No matching video chunks found.")
        print("=" * 50 + "\n")
        return

    for idx, res in enumerate(results, start=1):
        print(f"Result {idx}")
        print("-" * 42)
        print(f"Score:     {res['score']}")
        print(f"Title:     {res['title']}")
        print(f"Timestamp: {res['start_time']}")
        print(f"Start:     {res['start_sec']} sec")
        print("\nText:")
        print(res['text'])
        print("\nWatch:")
        print(res['youtube_url'])
        print()

    print("=" * 50 + "\n")


def print_search_report(result: dict) -> None:
    """Prints the two-stage retrieval report plus the final chunks in detail."""
    RetrievalPipeline.print_report(result)

    final = result.get("final", [])
    if final:
        print("\nFINAL CHUNKS (full text):")
        print_search_results(result.get("query", ""), final)
    else:
        print_search_results(result.get("query", ""), [])


def print_rag_answer(result: dict) -> None:
    """Pretty-prints a grounded RAG response with answer and sources."""
    print("\n" + "=" * 60)
    print("RAG ANSWER")
    print("=" * 60)

    grounded_label = "YES (grounded)" if result["grounded"] else "NO (insufficient context)"
    print(f"Grounded: {grounded_label}\n")

    print("ANSWER:")
    print("-" * 60)
    print(result["answer"])
    print("-" * 60)

    sources = result.get("sources", [])
    if sources:
        print(f"\nSOURCES ({len(sources)}):")
        for i, src in enumerate(sources, start=1):
            # Format seconds as MM:SS for display
            m, s = divmod(int(src["start_sec"]), 60)
            ts = f"{m:02d}:{s:02d}"
            print(f"\n  {i}. {src['title']}")
            print(f"     Timestamp : {ts}")
            print(f"     Watch at  : {src['youtube_url']}")
    else:
        print("\nSOURCES: None")

    print("=" * 60 + "\n")


def main():

    # Windows console safety: LLM answers / transcripts often contain characters
    # outside the legacy cp1252 code page (Hindi text, arrows, emoji). Emit
    # UTF-8 (lossy-safe) so printing never crashes.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description=(
            "Striver DSA AI Mentor - "
            "Phase 7: Production Playlist Ingestion + RAG & Evaluation"
        )
    )

    # Single video
    parser.add_argument(
        "--url",
        type=str,
        help="YouTube lecture URL or Video ID"
    )

    # Playlist
    parser.add_argument(
        "--playlist",
        type=str,
        help="YouTube playlist URL"
    )

    # Chunk mode
    parser.add_argument(
        "--chunk",
        action="store_true",
        help="Chunk all cached transcripts in data/transcripts/"
    )

    # Embed mode
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Generate vector embeddings for all chunk files in data/chunks/"
    )

    # Embed Info mode
    parser.add_argument(
        "--embed-info",
        action="store_true",
        help="Display embedding model information and vector dimension"
    )

    # Index mode (Qdrant)
    parser.add_argument(
        "--index",
        action="store_true",
        help="Upload embeddings in data/embeddings/ to Qdrant vector database"
    )

    # Re-index mode (Qdrant reset + upload)
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Recreate Qdrant collection and re-index cached embeddings"
    )

    # Index Info mode (Qdrant)
    parser.add_argument(
        "--index-info",
        action="store_true",
        help="Display Qdrant collection status, vector count, and configuration"
    )

    # Semantic Search mode
    parser.add_argument(
        "--search",
        type=str,
        help="Perform two-stage semantic search (candidates -> score gate -> dedup -> top-k)"
    )

    # Top K results limit for search
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of final search results to return (default: {DEFAULT_TOP_K})"
    )

    # RAG Ask mode (Phase 5)
    parser.add_argument(
        "--ask",
        type=str,
        default=None,
        help="Ask a question answered via grounded RAG (e.g. 'lower bound kya hota hai?')"
    )

    # Retrieval-quality evaluation mode (Phase 6)
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help=(
            "Run keyword-based retrieval-quality evaluation against "
            "evaluation/questions.json (no LLM judge, no LLM calls)"
        )
    )

    # Debug flag for RAG (shows retrieved chunks, scores, and context)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug output for RAG: retrieved chunks, scores, and context sent to LLM"
    )

    # Phase 8: Hybrid retrieval debug
    parser.add_argument(
        "--debug-search",
        type=str,
        default=None,
        help="Run hybrid retrieval (semantic + BM25 + reranker) and print the full pipeline"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of playlist videos to process"
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip the first N playlist entries before applying --limit"
    )

    # Whisper model
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_WHISPER_MODEL,
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: base)"
    )

    # Force flag
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if cached."
    )

    # -------------------------------------------------------------
    # Phase 7: Production playlist ingestion
    # -------------------------------------------------------------
    parser.add_argument(
        "--ingest-playlist",
        type=str,
        default=None,
        help=(
            "Run the END-TO-END playlist pipeline (discovery -> transcription "
            "-> chunking -> embedding -> Qdrant) with resume-by-default."
        )
    )

    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "Read data/ingestion_report.json and retry ONLY the videos that "
            "failed during a previous ingestion run."
        )
    )

    parser.add_argument(
        "--dataset-info",
        action="store_true",
        help=(
            "Show dataset status: playlist, validated artifact counts, "
            "Qdrant vectors, failed videos, and storage sizes."
        )
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume mode (DEFAULT for --ingest-playlist): skip stages that are "
            "already completed and validated. This flag exists for clarity and "
            "does not change behavior."
        )
    )

    # Stage-specific force flags for --ingest-playlist.
    # They never delete anything: forced stages simply re-run and overwrite
    # their cache; Qdrant upserts are idempotent (deterministic point IDs).
    parser.add_argument(
        "--force-transcript",
        action="store_true",
        help="--ingest-playlist: force re-transcription (and downstream stages)."
    )
    parser.add_argument(
        "--force-chunk",
        action="store_true",
        help="--ingest-playlist: force re-chunking (and downstream stages)."
    )
    parser.add_argument(
        "--force-embed",
        action="store_true",
        help="--ingest-playlist: force re-embedding (and downstream stages)."
    )
    parser.add_argument(
        "--force-index",
        action="store_true",
        help="--ingest-playlist: force re-indexing (idempotent Qdrant upsert)."
    )

    # -------------------------------------------------------------
    # Phase 7.1: YouTube cookie authentication for yt-dlp downloads.
    # Also configurable in .env via YTDLP_COOKIES_FROM_BROWSER and
    # YTDLP_COOKIES_FILE; these CLI flags take precedence.
    # -------------------------------------------------------------
    parser.add_argument(
        "--yt-cookies-browser",
        type=str,
        default=None,
        help=(
            "Read YouTube cookies from a browser for downloads to avoid "
            "bot-checks, e.g. chrome (=> yt-dlp --cookies-from-browser chrome). "
            "Overrides YTDLP_COOKIES_FROM_BROWSER in .env."
        ),
    )
    parser.add_argument(
        "--yt-cookies-file",
        type=str,
        default=None,
        help=(
            "Path to a Netscape-format cookies.txt for downloads "
            "(=> yt-dlp --cookies <file>). Overrides YTDLP_COOKIES_FILE in .env."
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------
    # Chunking Mode
    # -----------------------------------------
    if args.chunk:
        chunker = TranscriptChunker()
        chunker.process_all_transcripts(force=args.force)
        return
    # -----------------------------------------
    # Embed Mode
    # -----------------------------------------
    if args.embed:
        embedder = EmbeddingGenerator()
        embedder.process_all_chunks(force=args.force)
        return

    # -----------------------------------------
    # Embed Info Mode
    # -----------------------------------------
    if args.embed_info:
        embedder = EmbeddingGenerator()
        embedder.print_info()
        return

    # -----------------------------------------
    # Qdrant Index Mode
    # -----------------------------------------
    if args.index:
        vector_store = QdrantVectorStore()
        vector_store.index_all_embeddings(reindex=False)
        return

    # -----------------------------------------
    # Qdrant Re-index Mode
    # -----------------------------------------
    if args.reindex:
        vector_store = QdrantVectorStore()
        vector_store.index_all_embeddings(reindex=True)
        return

    # -----------------------------------------
    # Qdrant Index Info Mode
    # -----------------------------------------
    if args.index_info:
        vector_store = QdrantVectorStore()
        vector_store.print_info()
        return

    # -----------------------------------------
    # RAG Ask Mode (Phase 5)
    # -----------------------------------------
    if args.ask:
        print(f"\nQUESTION:\n{args.ask}\n")
        rag = RAGPipeline()
        result = rag.answer_question(
            query=args.ask,
            top_k=args.top_k,
            min_score=YTRAG_MIN_SCORE,
            debug=args.debug,
        )
        print_rag_answer(result)
        return

    # -----------------------------------------
    # Qdrant Search Mode (Phase 6: two-stage retrieval)
    # -----------------------------------------
    if args.search:
        pipeline = RetrievalPipeline()
        result = pipeline.retrieve(
            args.search,
            top_k=args.top_k,
            min_score=YTRAG_MIN_SCORE,
        )
        print_search_report(result)
        return

    # -----------------------------------------
    # Phase 8: Hybrid search debug mode
    # -----------------------------------------
    if args.debug_search:
        retriever = HybridRetriever()
        result = retriever.retrieve(args.debug_search)
        HybridRetriever.print_report(result)
        return

    # -----------------------------------------
    # Retrieval-Quality Evaluation Mode (Phase 6)
    # -----------------------------------------
    if args.evaluate:
        run_evaluation(
            limit=args.limit,
            debug=args.debug,
            top_k=args.top_k,
            min_score=YTRAG_MIN_SCORE,
        )
        return

    # -----------------------------------------
    # Phase 7: End-to-end Playlist Ingestion
    # -----------------------------------------
    if args.ingest_playlist:
        from ingestion import PlaylistIngestor

        if args.retry_failed:
            print(
                "Error: use either --ingest-playlist OR --retry-failed, "
                "not both."
            )
            sys.exit(1)

        force_flags = {}
        if args.force or args.force_transcript:
            force_flags["transcription"] = True
        if args.force or args.force_chunk:
            force_flags["chunking"] = True
        if args.force or args.force_embed:
            force_flags["embedding"] = True
        if args.force or args.force_index:
            force_flags["indexing"] = True

        ingestor = PlaylistIngestor(
            playlist_url=args.ingest_playlist,
            limit=args.limit,
            start_index=args.start_index,
            force_flags=force_flags,
            whisper_model=args.model,
            cookies_from_browser=args.yt_cookies_browser,
            cookies_file=args.yt_cookies_file,
        )
        ingestor.run(mode="full")
        return

    # -----------------------------------------
    # Phase 7: Retry failed videos
    # -----------------------------------------
    if args.retry_failed:
        from ingestion import PlaylistIngestor

        ingestor = PlaylistIngestor(
            limit=args.limit,
            force_flags={},
            whisper_model=args.model,
            cookies_from_browser=args.yt_cookies_browser,
            cookies_file=args.yt_cookies_file,
        )
        ingestor.retry_failed()
        return

    # -----------------------------------------
    # Phase 7: Dataset info
    # -----------------------------------------
    if args.dataset_info:
        from ingestion import dataset_info

        dataset_info()
        return

    # -----------------------------------------
    # Validate input
    # -----------------------------------------

    if args.url and args.playlist:
        print(
            "Error: Use either --url OR --playlist, "
            "not both."
        )
        sys.exit(1)

    # -----------------------------------------
    # Interactive mode
    # -----------------------------------------

    if not (
        args.url
        or args.playlist
        or args.chunk
        or args.embed
        or args.embed_info
        or args.index
        or args.reindex
        or args.index_info
        or args.search
        or args.ask
        or args.evaluate
        or args.ingest_playlist
        or args.retry_failed
        or args.dataset_info
        or args.debug_search
    ):

        print("\n--- Striver DSA AI Mentor ---")
        print("Phase 7: Production Playlist Ingestion + RAG & Evaluation\n")

        print("1. Process single video")
        print("2. Process playlist")
        print("3. Chunk cached transcripts")
        print("4. Generate embeddings")
        print("5. Show embedding info")
        print("6. Index embeddings to Qdrant")
        print("7. Re-index embeddings (reset Qdrant collection)")
        print("8. Show Qdrant DB info")
        print("9. Semantic search query (two-stage retrieval)")
        print("10. Ask a question (RAG)")
        print("11. Evaluate retrieval quality (keyword-based, no LLM judge)")
        print("12. Ingest playlist end-to-end (Phase 7, auto-resume)")
        print("13. Retry failed videos")
        print("14. Show dataset info")
        print("15. Debug hybrid search (Phase 8)")

        choice = input(
            "\nChoose option (1-15): "
        ).strip()

        if choice == "1":
            args.url = input(
                "Enter YouTube Video URL: "
            ).strip()

        elif choice == "2":
            args.playlist = input(
                "Enter YouTube Playlist URL: "
            ).strip()

            limit_input = input(
                "Number of videos to test "
                "(press Enter for all): "
            ).strip()

            if limit_input:
                try:
                    args.limit = int(limit_input)
                except ValueError:
                    print(
                        "Invalid limit. "
                        "Processing all videos."
                    )
                    args.limit = None

        elif choice == "3":
            chunker = TranscriptChunker()
            chunker.process_all_transcripts(force=args.force)
            return

        elif choice == "4":
            embedder = EmbeddingGenerator()
            embedder.process_all_chunks(force=args.force)
            return

        elif choice == "5":
            embedder = EmbeddingGenerator()
            embedder.print_info()
            return

        elif choice == "6":
            vector_store = QdrantVectorStore()
            vector_store.index_all_embeddings(reindex=False)
            return

        elif choice == "7":
            vector_store = QdrantVectorStore()
            vector_store.index_all_embeddings(reindex=True)
            return

        elif choice == "8":
            vector_store = QdrantVectorStore()
            vector_store.print_info()
            return

        elif choice == "9":
            query = input("Enter search query: ").strip()
            pipeline = RetrievalPipeline()
            result = pipeline.retrieve(
                query,
                top_k=DEFAULT_TOP_K,
                min_score=YTRAG_MIN_SCORE,
            )
            print_search_report(result)
            return

        elif choice == "10":
            query = input("Enter your question: ").strip()
            print(f"\nQUESTION:\n{query}\n")
            rag = RAGPipeline()
            result = rag.answer_question(
                query=query,
                top_k=DEFAULT_TOP_K,
                min_score=YTRAG_MIN_SCORE,
                debug=False,
            )
            print_rag_answer(result)
            return

        elif choice == "11":
            run_evaluation(limit=args.limit, debug=args.debug)
            return

        elif choice == "12":
            from ingestion import PlaylistIngestor

            playlist_url = input(
                "Enter YouTube Playlist URL: "
            ).strip()
            limit_input = input(
                "Videos to process (Enter = all): "
            ).strip()
            limit = int(limit_input) if limit_input.isdigit() else None
            ingestor = PlaylistIngestor(
                playlist_url=playlist_url,
                limit=limit,
                force_flags={},
                whisper_model=args.model,
                cookies_from_browser=args.yt_cookies_browser,
                cookies_file=args.yt_cookies_file,
            )
            ingestor.run(mode="full")
            return

        elif choice == "13":
            from ingestion import PlaylistIngestor

            ingestor = PlaylistIngestor(
                force_flags={},
                whisper_model=args.model,
                cookies_from_browser=args.yt_cookies_browser,
                cookies_file=args.yt_cookies_file,
            )
            ingestor.retry_failed()
            return

        elif choice == "14":
            from ingestion import dataset_info

            dataset_info()
            return

        elif choice == "15":
            query = input("Enter debug search query: ").strip()
            retriever = HybridRetriever()
            result = retriever.retrieve(query)
            HybridRetriever.print_report(result)
            return

        else:
            print("Invalid choice.")
            sys.exit(1)

    # -----------------------------------------
    # Create pipeline
    # -----------------------------------------

    try:

        pipeline = LecturePipeline(
            whisper_model=args.model
        )

        # -------------------------------------
        # Single video mode
        # -------------------------------------

        if args.url:

            result = pipeline.process_url(
                args.url,
                force_retranscribe=args.force
            )

            print_single_result(result)

            print(
                "\nPhase 1 process completed successfully!"
            )

        # -------------------------------------
        # Playlist mode
        # -------------------------------------

        elif args.playlist:

            result = pipeline.process_playlist(
                args.playlist,
                limit=args.limit,
                force_retranscribe=args.force
            )

            print("\n==========================================")
            print(" PHASE 2 COMPLETED")
            print("==========================================")

            print(
                f"Processed: "
                f"{result['successful'].__len__()} videos"
            )

            print(
                f"Failed:    "
                f"{result['failed'].__len__()} videos"
            )

    except KeyboardInterrupt:

        print(
            "\n\nProcess interrupted by user."
        )
        print(
            "Cached transcripts are safe."
        )
        sys.exit(1)

    except Exception as e:

        print(
            f"\n[ERROR] Pipeline failed: {e}",
            file=sys.stderr
        )

        sys.exit(1)


if __name__ == "__main__":
    main()