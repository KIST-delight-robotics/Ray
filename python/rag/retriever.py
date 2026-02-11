# -*- coding: utf-8 -*-
"""
RAG Retriever - 영화/음악 검색 모듈

ChromaDB + LangChain을 사용하여 영화 정보, 평론, 인터뷰 데이터를 검색합니다.
intent 기반 필터링 및 2차 보강(Enrichment) 로직을 포함합니다.
"""

import logging
from typing import Optional

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

# 모듈 레벨 상태
_vectorstore: Optional[Chroma] = None
_embeddings: Optional[OpenAIEmbeddings] = None


def init_db(persist_dir: str, openai_api_key: str, collection_name: str = "movie_archive"):
    """
    ChromaDB 초기화 (앱 시작 시 1회 호출)
    
    Args:
        persist_dir: ChromaDB 저장 경로
        openai_api_key: OpenAI API 키
        collection_name: 컬렉션 이름
    """
    global _vectorstore, _embeddings
    
    _embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key
    )
    
    _vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=_embeddings,
        persist_directory=persist_dir
    )
    
    logger.info(f"RAG DB 초기화 완료: {persist_dir}")


def search_archive(query: str, intent: str, top_k: int = 3) -> str:
    """
    intent별 영화 정보 검색
    
    Args:
        query: 검색 쿼리 (LLM이 변환한 검색어)
        intent: 검색 의도
            - "fact": 기본 정보만 검색 (감독, 출연진, 줄거리 등)
            - "vibe": 전체 검색 + enrichment (분위기 기반 추천)
            - "critique": 평론/인터뷰만 + enrichment (깊이 있는 해석)
        top_k: 반환할 문서 수
        
    Returns:
        포맷팅된 검색 결과 문자열
    """
    if _vectorstore is None:
        logger.warning("RAG DB가 초기화되지 않음")
        return "영화 정보를 검색할 수 없습니다."
    
    try:
        # intent에 따른 필터 설정
        if intent == "fact":
            filter_dict = {"category": "basic_info"}
            docs = _vectorstore.similarity_search(query, k=top_k, filter=filter_dict)
            return _format_results(docs)
        
        elif intent == "vibe":
            # 전체 검색 (필터 없음)
            docs = _vectorstore.similarity_search(query, k=top_k)
            return _enrich_and_format(docs)
        
        elif intent == "critique":
            # 평론/인터뷰만 검색
            filter_dict = {"category": {"$in": ["critique", "interview"]}}
            docs = _vectorstore.similarity_search(query, k=top_k, filter=filter_dict)
            return _enrich_and_format(docs)
        
        else:
            # 알 수 없는 intent → 전체 검색
            logger.warning(f"알 수 없는 intent: {intent}, 전체 검색 수행")
            docs = _vectorstore.similarity_search(query, k=top_k)
            return _format_results(docs)
            
    except Exception as e:
        logger.error(f"RAG 검색 오류: {e}")
        return "검색 중 오류가 발생했습니다."


def _enrich_and_format(docs: list) -> str:
    """
    검색된 심층 문서에 기본 정보를 보강하여 포맷팅
    
    심층 문서(평론/인터뷰)가 검색되면, 해당 영화의 기본 정보를 
    함께 조회하여 LLM이 환각 없이 답변할 수 있도록 합니다.
    """
    if not docs:
        return "관련 정보를 찾지 못했습니다."
    
    # 1. 검색된 문서에서 movie_id 추출
    movie_ids = set()
    for doc in docs:
        movie_id = doc.metadata.get("movie_id")
        if movie_id:
            # movie_id가 리스트일 수 있음 (복수 영화 연관 평론)
            if isinstance(movie_id, list):
                movie_ids.update(movie_id)
            else:
                movie_ids.add(movie_id)
    
    # 2. 기본 정보 조회 (보강)
    basic_info_docs = []
    found_movie_ids = set()
    if movie_ids and _vectorstore:
        for mid in movie_ids:
            try:
                results = _vectorstore.similarity_search(
                    "",  # 빈 쿼리 (메타데이터 필터만 사용)
                    k=1,
                    filter={"$and": [
                        {"movie_id": mid},
                        {"category": "basic_info"}
                    ]}
                )
                if results:
                    basic_info_docs.extend(results)
                    found_movie_ids.add(mid)
            except Exception as e:
                logger.debug(f"기본 정보 조회 실패 (movie_id={mid}): {e}")
    
    # 3. 결과 포맷팅
    result_parts = []
    
    # 기본 정보 섹션
    if basic_info_docs:
        result_parts.append("## 영화 기본 정보")
        seen_titles = set()
        for doc in basic_info_docs:
            title = doc.metadata.get("title", "제목 없음")
            if title not in seen_titles:
                seen_titles.add(title)
                result_parts.append(f"### {title}")
                result_parts.append(doc.page_content)
    
    # 기본 정보 없는 영화 표시
    missing_ids = movie_ids - found_movie_ids
    if missing_ids:
        result_parts.append(f"\n(기본정보 없음: {len(missing_ids)}편)")
    
    # 심층 정보 섹션
    result_parts.append("\n## 관련 평론/인터뷰")
    for doc in docs:
        category = doc.metadata.get("category", "기타")
        title = doc.metadata.get("title", "")
        if category != "basic_info":  # 기본 정보 중복 제외
            result_parts.append(f"[{title} - {category}]")
            result_parts.append(doc.page_content)
    
    return "\n".join(result_parts)


def _format_results(docs: list) -> str:
    """검색 결과 기본 포맷팅"""
    if not docs:
        return "관련 정보를 찾지 못했습니다."
    
    result_parts = []
    for doc in docs:
        title = doc.metadata.get("title", "제목 없음")
        category = doc.metadata.get("category", "")
        result_parts.append(f"[{title}] ({category})")
        result_parts.append(doc.page_content)
        result_parts.append("")  # 빈 줄로 구분
    
    return "\n".join(result_parts)


def get_collection_stats() -> dict:
    """컬렉션 통계 반환 (디버깅용)"""
    if _vectorstore is None:
        return {"error": "DB not initialized"}
    
    try:
        collection = _vectorstore._collection
        return {
            "count": collection.count(),
            "name": collection.name
        }
    except Exception as e:
        return {"error": str(e)}


def search_archive_debug(query: str, intent: str, top_k: int = 3) -> tuple:
    """
    디버그용 검색 함수 - 검색된 문서 리스트와 포맷된 결과를 함께 반환
    
    Returns:
        (docs_info: list[dict], formatted_result: str)
    """
    if _vectorstore is None:
        return [], "영화 정보를 검색할 수 없습니다."
    
    try:
        # intent에 따른 필터 설정
        if intent == "fact":
            filter_dict = {"category": "basic_info"}
            docs = _vectorstore.similarity_search(query, k=top_k, filter=filter_dict)
        elif intent == "vibe":
            docs = _vectorstore.similarity_search(query, k=top_k)
        elif intent == "critique":
            filter_dict = {"category": {"$in": ["critique", "interview"]}}
            docs = _vectorstore.similarity_search(query, k=top_k, filter=filter_dict)
        else:
            docs = _vectorstore.similarity_search(query, k=top_k)
        
        # 문서 정보 추출
        docs_info = []
        for i, doc in enumerate(docs):
            docs_info.append({
                "index": i + 1,
                "title": doc.metadata.get("title", "제목 없음"),
                "category": doc.metadata.get("category", ""),
                "movie_id": doc.metadata.get("movie_id", ""),
                "author": doc.metadata.get("author", ""),
                "chunk_index": doc.metadata.get("chunk_index"),
                "content_preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            })
        
        # 포맷된 결과
        if intent in ["vibe", "critique"]:
            formatted = _enrich_and_format(docs)
        else:
            formatted = _format_results(docs)
        
        return docs_info, formatted
        
    except Exception as e:
        logger.error(f"RAG 검색 오류: {e}")
        return [], f"검색 중 오류가 발생했습니다: {e}"

