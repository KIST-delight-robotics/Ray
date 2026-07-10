"""Long-term memory benchmark harness (LoCoMo).

프로덕션 memory 모듈(MemoryWriter/MemoryRetriever)을 무수정으로 사용해
LoCoMo류 멀티세션 대화 벤치마크를 실행한다. 3단계 CLI:

    uv run python -m evaluation.memory_bench ingest --data ... --run-dir ...
    uv run python -m evaluation.memory_bench answer --run-dir ...
    uv run python -m evaluation.memory_bench score --run-dir ...

각 단계는 run 디렉토리에 산출물(DB 스냅샷 → answers.jsonl → scores.json)을
남기며 독립적으로 재실행할 수 있다. 설계 배경: docs/decisions-wip.md.
"""
