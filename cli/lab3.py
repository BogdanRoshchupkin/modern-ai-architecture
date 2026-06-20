from __future__ import annotations

from src.backend.flash_attention import build_benchmark_parser, run_benchmark_from_args


def main() -> None:
    parser = build_benchmark_parser()
    args = parser.parse_args()
    result = run_benchmark_from_args(args)
    print(f"backend: {args.backend}")
    print(f"torch attention median: {result.torch_ms:.3f} ms")
    print(f"flash attention median: {result.flash_ms:.3f} ms")
    print(f"speedup: {result.speedup:.2f}x")
    print(f"torch score matrix memory: {result.torch_score_memory_mb:.2f} MB")
    print(f"flash score block memory: {result.flash_score_memory_mb:.2f} MB")
    print(f"score-memory reduction: {result.memory_ratio:.2f}x")


if __name__ == "__main__":
    main()
