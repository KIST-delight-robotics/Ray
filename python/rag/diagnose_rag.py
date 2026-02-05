# -*- coding: utf-8 -*-
"""RAG 검색 진단 스크립트"""

import os
import sys
from pathlib import Path

# python 폴더를 sys.path에 추가 (rag 폴더에서 실행해도 상위 모듈 import 가능)
python_dir = Path(__file__).parent.parent
sys.path.insert(0, str(python_dir))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("오류: OPENAI_API_KEY 환경변수 필요")
    sys.exit(1)

from rag import init_db
from rag.retriever import get_collection_stats
from config import RAG_PERSIST_DIR

print("📊 RAG DB 진단 시작...\n")

# 초기화
init_db(str(RAG_PERSIST_DIR), OPENAI_API_KEY)

# 초기화 후 _vectorstore 가져오기
from rag import retriever
vectorstore = retriever._vectorstore

stats = get_collection_stats()
print(f"DB 통계: 총 {stats.get('count', 0)}개 문서\n")

# 모든 문서 제목 확인
print("=" * 50)
print("저장된 영화 목록:")
print("=" * 50)
try:
    # ChromaDB에서 모든 문서 가져오기
    collection = vectorstore._collection
    all_docs = collection.get()
    
    titles = set()
    for metadata in all_docs.get('metadatas', []):
        title = metadata.get('title', 'N/A')
        titles.add(title)
    
    for i, title in enumerate(sorted(titles), 1):
        print(f"  {i}. {title}")
    
    print(f"\n총 {len(titles)}개 고유 영화")
except Exception as e:
    print(f"오류: {e}")

# 검색 테스트
print("\n" + "=" * 50)
print("검색 테스트:")
print("=" * 50)

test_queries = [
    "action",
    "horror monster",
    "romantic love",
    "funny comedy",
    "sad emotional"
]

for query in test_queries:
    try:
        docs = vectorstore.similarity_search(query, k=3)
        titles = [d.metadata.get('title', 'N/A')[:30] for d in docs]
        print(f"\n'{query}':")
        for i, title in enumerate(titles, 1):
            print(f"  {i}. {title}")
    except Exception as e:
        print(f"  오류: {e}")

print("\n✅ 진단 완료")
