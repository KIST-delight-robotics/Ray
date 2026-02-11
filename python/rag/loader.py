# -*- coding: utf-8 -*-
"""
RAG Data Loader - 영화 데이터 로딩 유틸리티

JSON 파일에서 영화 데이터를 읽어 ChromaDB에 저장합니다.
LangChain의 RecursiveCharacterTextSplitter를 사용한 청킹을 지원합니다.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Optional

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# 기본 청킹 설정
DEFAULT_CHUNK_SIZE = 700       # 한국어 최적화 (영어 500자 대비)
DEFAULT_CHUNK_OVERLAP = 100    # 문장 끊김 방지


def load_movie_data(
    json_path: str | Path,
    persist_dir: str,
    openai_api_key: str,
    collection_name: str = "movie_archive",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> int:
    """
    JSON 파일에서 영화 데이터를 읽어 ChromaDB에 저장
    
    Args:
        json_path: 입력 JSON 파일 경로
        persist_dir: ChromaDB 저장 경로
        openai_api_key: OpenAI API 키
        collection_name: 컬렉션 이름
        chunk_size: 청크 크기 (문자 수)
        chunk_overlap: 청크 오버랩 (문자 수)
        
    Returns:
        저장된 문서 수
        
    입력 JSON 형식:
    [
        {
            "id": "unique_doc_id",
            "movie_id": "tmdb_12345",  // 또는 ["tmdb_123", "tmdb_456"]
            "title": "영화 제목",
            "director": "감독명",  // 선택
            "category": "basic_info" | "critique" | "interview",
            "content": "본문 텍스트"
        },
        ...
    ]
    """
    # JSON 파일 로드
    path = Path(json_path) if isinstance(json_path, str) else json_path
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    logger.info(f"JSON 로드 완료: {len(data)}개 항목")
    
    # 텍스트 분할기 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""]
    )
    
    # Document 객체 생성 및 청킹
    all_docs = []
    for item in data:
        doc_id = item.get("id", "")
        content = item.get("content", "")
        
        if not content:
            logger.warning(f"빈 content 건너뜀: {doc_id}")
            continue
        
        # 메타데이터 구성
        metadata = {
            "doc_id": doc_id,
            "movie_id": item.get("movie_id", ""),
            "title": item.get("title", ""),
            "category": item.get("category", "other"),
        }
        
        # 선택적 메타데이터
        if "director" in item:
            metadata["director"] = item["director"]
        if "author" in item:
            metadata["author"] = item["author"]
        if "source" in item:
            metadata["source"] = item["source"]
        
        # 청킹 (basic_info는 통째로, 나머지는 분할)
        if item.get("category") == "basic_info":
            # 기본 정보는 청킹하지 않음 (항상 전체 제공)
            all_docs.append(Document(page_content=content, metadata=metadata))
        else:
            # 평론/인터뷰는 청킹
            chunks = text_splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = i
                all_docs.append(Document(page_content=chunk, metadata=chunk_metadata))
    
    logger.info(f"청킹 완료: {len(all_docs)}개 문서 생성")
    
    # ChromaDB에 저장
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key
    )
    
    # 저장 디렉토리 생성
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_path)
    )
    
    logger.info(f"ChromaDB 저장 완료: {persist_dir}")
    
    return len(all_docs)


def clear_collection(persist_dir: str, collection_name: str = "movie_archive"):
    """컬렉션 초기화 (기존 데이터 삭제)"""
    import chromadb
    
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        client.delete_collection(collection_name)
        logger.info(f"컬렉션 삭제됨: {collection_name}")
    except Exception as e:
        logger.warning(f"컬렉션 삭제 실패: {e}")


# CLI 지원
if __name__ == "__main__":
    import os
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    parser = argparse.ArgumentParser(description="영화 데이터를 ChromaDB에 로드")
    parser.add_argument("--input", "-i", required=True, help="입력 JSON 파일 경로")
    parser.add_argument("--output", "-o", default="../data/chroma_db", help="ChromaDB 저장 경로")
    parser.add_argument("--clear", action="store_true", help="기존 컬렉션 삭제 후 저장")
    
    args = parser.parse_args()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        exit(1)
    
    if args.clear:
        clear_collection(args.output)
    
    count = load_movie_data(args.input, args.output, api_key)
    print(f"완료: {count}개 문서 저장됨")
