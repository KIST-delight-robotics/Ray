"""
Cine21 크롤링 데이터 → RAG JSON 변환 스크립트

Cine21 기사와 TMDB 영화 정보를 RAG용 JSON으로 변환합니다.
- basic_info: TMDB 영화 기본 정보
- critique/review: Cine21 기사

사용법:
    python convert_cine21_to_rag.py              # 전체 변환
    python convert_cine21_to_rag.py --test       # 테스트 (100개만)
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional

# 경로 설정
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CRAWLED_DIR = DATA_DIR / "crawled"
OUTPUT_DIR = DATA_DIR / "rag_ready"

# 제거할 푸터 텍스트
FOOTER_TEXT = "\n\n글자크기 설정 시 다른 기사의 본문도 동일하게 적용됩니다."


def clean_content(content: str) -> str:
    """본문 정리: 푸터 제거 및 공백 정리"""
    if not content:
        return ""
    # 푸터 제거
    content = content.replace(FOOTER_TEXT, "")
    content = content.replace("글자크기 설정 시 다른 기사의 본문도 동일하게 적용됩니다.", "")
    return content.strip()


def load_tmdb_movies() -> dict:
    """TMDB 영화 정보 로드 (cine21_movie_id → movie_data 매핑)"""
    tmdb_file = CRAWLED_DIR / "movies_tmdb.json"
    if not tmdb_file.exists():
        print("⚠️ movies_tmdb.json 없음 - TMDB 정보 없이 진행")
        return {}
    
    with open(tmdb_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # cine21_movie_id로 인덱싱
    movies = {}
    for movie in data.get("movies", []):
        cine21_id = movie.get("cine21_movie_id")
        if cine21_id:
            movies[cine21_id] = movie
    
    print(f"✅ TMDB 영화 {len(movies)}개 로드")
    return movies


def convert_tmdb_to_basic_info(tmdb_movies: dict) -> list:
    """TMDB 영화 → basic_info 문서 변환"""
    documents = []
    
    for cine21_id, movie in tmdb_movies.items():
        if not movie.get("tmdb_found"):
            continue
        
        tmdb_id = movie.get("tmdb_id")
        title = movie.get("tmdb_title") or movie.get("cine21_title", "")
        original_title = movie.get("tmdb_original_title", "")
        
        # 콘텐츠 구성 - 제목을 맨 앞에!
        content_parts = []
        
        # 제목 (한글 + 원제)
        if title:
            if original_title and original_title != title:
                content_parts.append(f"제목: {title} ({original_title})")
            else:
                content_parts.append(f"제목: {title}")
        
        # 줄거리
        overview = movie.get("overview")
        if overview:
            content_parts.append(overview)
        
        # 기본 정보
        directors = movie.get("directors", [])
        if directors:
            content_parts.append(f"감독: {', '.join(directors)}")
        
        genres = movie.get("genres", [])
        if genres:
            content_parts.append(f"장르: {', '.join(genres)}")
        
        release_date = movie.get("release_date")
        if release_date:
            content_parts.append(f"개봉일: {release_date}")
        
        runtime = movie.get("runtime")
        if runtime:
            content_parts.append(f"상영시간: {runtime}분")
        
        vote_avg = movie.get("vote_average")
        if vote_avg:
            content_parts.append(f"평점: {vote_avg}/10")
        
        # 출연진
        cast = movie.get("cast", [])[:5]
        if cast:
            cast_names = [c.get("name", "") for c in cast]
            content_parts.append(f"출연: {', '.join(cast_names)}")
        
        if not content_parts:
            continue
        
        doc = {
            "id": f"tmdb_{tmdb_id}",
            "movie_id": f"tmdb_{tmdb_id}",
            "title": title,
            "category": "basic_info",
            "source": "tmdb",
            "content": " | ".join(content_parts)
        }
        
        if directors:
            doc["director"] = directors[0]
        
        documents.append(doc)
    
    return documents


def get_movie_id(article: dict, tmdb_movies: dict) -> Optional[str]:
    """기사의 관련 영화에서 movie_id 추출 (TMDB 우선, 없으면 cine21 ID)"""
    related = article.get("related_movies", [])
    if not related:
        return None
    
    # 첫 번째 관련 영화 사용
    cine21_movie_id = related[0].get("movie_id")
    if not cine21_movie_id:
        return None
    
    # TMDB에서 찾기
    tmdb_movie = tmdb_movies.get(cine21_movie_id)
    if tmdb_movie and tmdb_movie.get("tmdb_found"):
        return f"tmdb_{tmdb_movie['tmdb_id']}"
    
    # TMDB에 없으면 cine21 ID 사용
    return f"cine21_{cine21_movie_id}"


def convert_cine21_articles(tmdb_movies: dict, max_count: Optional[int] = None) -> tuple:
    """Cine21 기사 → critique/review 문서 변환"""
    documents = []
    content_lengths = []  # 본문 길이 통계용
    count = 0
    
    for json_file in CRAWLED_DIR.glob("cine21_*.json"):
        print(f"  📄 {json_file.name} 처리 중...")
        
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for article in data.get("articles", []):
            if max_count and count >= max_count:
                break
            
            mag_id = article.get("mag_id")
            title = article.get("title", "")
            content = clean_content(article.get("content", ""))
            
            if not content or len(content) < 50:
                continue
            
            content_lengths.append(len(content))
            
            # 카테고리 매핑
            category_name = article.get("category_name", "")
            if "비평" in category_name or "읽기" in category_name:
                category = "critique"
            else:
                category = "review"
            
            # movie_id 결정
            movie_id = get_movie_id(article, tmdb_movies)
            
            doc = {
                "id": f"cine21_{mag_id}",
                "movie_id": movie_id or "",
                "title": title,
                "category": category,
                "source": "cine21",
                "content": content
            }
            
            # 저자 정보
            author = article.get("author", "")
            if author:
                doc["author"] = author
            
            documents.append(doc)
            count += 1
        
        if max_count and count >= max_count:
            break
    
    return documents, content_lengths


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Cine21 → RAG JSON 변환")
    parser.add_argument("--test", action="store_true", help="테스트 모드 (100개만)")
    parser.add_argument("--articles-only", action="store_true", help="기사만 변환 (TMDB basic_info 제외)")
    args = parser.parse_args()
    
    print("\n🎬 Cine21 → RAG 변환")
    print(f"⏰ 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # TMDB 영화 로드
    tmdb_movies = load_tmdb_movies()
    
    all_documents = []
    
    # 1. TMDB basic_info 변환
    if not args.articles_only:
        print("\n📋 TMDB → basic_info 변환 중...")
        basic_docs = convert_tmdb_to_basic_info(tmdb_movies)
        all_documents.extend(basic_docs)
        print(f"  ✅ {len(basic_docs)}개 basic_info 문서 생성")
    
    # 2. Cine21 기사 변환
    print("\n📰 Cine21 → critique/review 변환 중...")
    max_count = 100 if args.test else None
    article_docs, content_lengths = convert_cine21_articles(tmdb_movies, max_count)
    all_documents.extend(article_docs)
    print(f"  ✅ {len(article_docs)}개 기사 문서 생성")
    
    # 본문 길이 통계
    if content_lengths:
        avg_len = sum(content_lengths) / len(content_lengths)
        min_len = min(content_lengths)
        max_len = max(content_lengths)
        print(f"\n📏 본문 길이 통계:")
        print(f"  평균: {avg_len:,.0f}자")
        print(f"  최소: {min_len:,}자 | 최대: {max_len:,}자")
    
    # 저장
    output_file = OUTPUT_DIR / "cine21_rag.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_documents, f, ensure_ascii=False, indent=2)
    
    # 통계
    categories = {}
    for doc in all_documents:
        cat = doc.get("category", "other")
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\n✅ 변환 완료!")
    print(f"  📊 총 {len(all_documents)}개 문서")
    for cat, cnt in categories.items():
        print(f"     - {cat}: {cnt}개")
    print(f"  📁 저장: {output_file}")
    print(f"⏰ 종료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
