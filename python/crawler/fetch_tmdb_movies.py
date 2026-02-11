"""
TMDB 영화 정보 수집 스크립트

Cine21에서 수집한 관련 영화들의 상세 정보를 TMDB에서 가져옵니다.

사용법:
    python fetch_tmdb_movies.py              # 전체 실행
    python fetch_tmdb_movies.py --test       # 테스트 (10개만)
    
환경변수:
    TMDB_API_KEY: TMDB API 키 (https://www.themoviedb.org/settings/api)
"""

import json
import os
import time
import argparse
import urllib3
from pathlib import Path
from datetime import datetime
from typing import Optional

import requests

urllib3.disable_warnings()

# 설정
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "crawled"
OUTPUT_FILE = DATA_DIR / "movies_tmdb_v2.json"
PROGRESS_FILE = DATA_DIR / "tmdb_progress_v2.json"

TMDB_BASE_URL = "https://api.themoviedb.org/3"
REQUEST_DELAY = 0.05  # TMDB는 초당 50회 허용

# TMDB API 키 (환경변수 또는 직접 입력)
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")


def get_api_key() -> str:
    """API 키 가져오기"""
    if TMDB_API_KEY:
        return TMDB_API_KEY
    
    # config.py에서 가져오기 시도
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "python"))
        from config import TMDB_API_KEY as CONFIG_KEY
        if CONFIG_KEY:
            return CONFIG_KEY
    except:
        pass
    
    print("❌ TMDB API 키가 필요합니다.")
    print("   1. 환경변수: set TMDB_API_KEY=your_key")
    print("   2. 또는 python/config.py에 TMDB_API_KEY 추가")
    print("   3. API 키 발급: https://www.themoviedb.org/settings/api")
    return ""


def collect_unique_movies() -> list:
    """모든 JSON 파일에서 고유한 영화 목록 수집"""
    movies = {}  # movie_id -> {title, year, cine21_movie_id}
    
    for json_file in DATA_DIR.glob("cine21_*.json"):
        print(f"  📄 {json_file.name} 읽는 중...")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for article in data.get("articles", []):
            for movie in article.get("related_movies", []):
                movie_id = movie.get("movie_id")
                if movie_id and movie_id not in movies:
                    movies[movie_id] = {
                        "cine21_movie_id": movie_id,
                        "title": movie.get("title"),
                        "year": movie.get("year")
                    }
    
    return list(movies.values())


def search_tmdb_movie(title: str, year: str, api_key: str, session: requests.Session) -> Optional[dict]:
    """TMDB에서 영화 검색 - 제목+연도 정확 매칭 우선"""
    if not title:
        return None
    
    def find_best_match(results: list, target_title: str, target_year: str) -> Optional[dict]:
        """검색 결과 중 최적 매칭 찾기"""
        if not results:
            return None
        
        target_title_norm = target_title.strip().lower()
        
        # 1순위: 제목 완전 일치 + 연도 일치
        if target_year and target_year.isdigit():
            for r in results:
                r_title = (r.get("title") or "").strip().lower()
                r_orig = (r.get("original_title") or "").strip().lower()
                r_year = (r.get("release_date") or "")[:4]
                
                if (r_title == target_title_norm or r_orig == target_title_norm) and r_year == target_year:
                    return r
        
        # 2순위: 제목 완전 일치 (연도 무시)
        for r in results:
            r_title = (r.get("title") or "").strip().lower()
            r_orig = (r.get("original_title") or "").strip().lower()
            
            if r_title == target_title_norm or r_orig == target_title_norm:
                return r
        
        # 3순위: 제목 포함 + 연도 일치
        if target_year and target_year.isdigit():
            for r in results:
                r_title = (r.get("title") or "").strip().lower()
                r_year = (r.get("release_date") or "")[:4]
                
                if target_title_norm in r_title and r_year == target_year:
                    return r
        
        # 4순위: 첫 번째 결과 (fallback)
        return results[0]
    
    params = {
        "api_key": api_key,
        "query": title,
        "language": "ko-KR",
        "include_adult": "false"
    }
    
    try:
        time.sleep(REQUEST_DELAY)
        r = session.get(f"{TMDB_BASE_URL}/search/movie", params=params, timeout=10, verify=False)
        
        if r.status_code != 200:
            return None
        
        data = r.json()
        results = data.get("results", [])
        
        # 최적 매칭 찾기
        best = find_best_match(results, title, year)
        if best:
            return best
        
        return None
        
    except Exception as e:
        return None


def get_movie_details(tmdb_id: int, api_key: str, session: requests.Session) -> Optional[dict]:
    """TMDB 영화 상세 정보 가져오기"""
    try:
        time.sleep(REQUEST_DELAY)
        params = {
            "api_key": api_key,
            "language": "ko-KR",
            "append_to_response": "credits,keywords"
        }
        r = session.get(f"{TMDB_BASE_URL}/movie/{tmdb_id}", params=params, timeout=10, verify=False)
        
        if r.status_code != 200:
            return None
        
        return r.json()
        
    except Exception as e:
        return None


def process_movie(movie_info: dict, api_key: str, session: requests.Session) -> Optional[dict]:
    """단일 영화 처리: 검색 → 상세 정보"""
    title = movie_info.get("title")
    year = movie_info.get("year")
    cine21_id = movie_info.get("cine21_movie_id")
    
    # TMDB 검색
    search_result = search_tmdb_movie(title, year, api_key, session)
    if not search_result:
        return {
            "cine21_movie_id": cine21_id,
            "cine21_title": title,
            "cine21_year": year,
            "tmdb_found": False
        }
    
    tmdb_id = search_result.get("id")
    
    # 상세 정보 가져오기
    details = get_movie_details(tmdb_id, api_key, session)
    if not details:
        details = search_result
    
    # 결과 정리
    credits = details.get("credits", {})
    cast = credits.get("cast", [])[:10]  # 상위 10명
    crew = credits.get("crew", [])
    
    directors = [c["name"] for c in crew if c.get("job") == "Director"]
    
    return {
        "cine21_movie_id": cine21_id,
        "cine21_title": title,
        "cine21_year": year,
        "tmdb_found": True,
        "tmdb_id": tmdb_id,
        "tmdb_title": details.get("title"),
        "tmdb_original_title": details.get("original_title"),
        "release_date": details.get("release_date"),
        "overview": details.get("overview"),
        "genres": [g["name"] for g in details.get("genres", [])],
        "runtime": details.get("runtime"),
        "vote_average": details.get("vote_average"),
        "vote_count": details.get("vote_count"),
        "poster_path": details.get("poster_path"),
        "backdrop_path": details.get("backdrop_path"),
        "directors": directors,
        "cast": [{"name": c["name"], "character": c.get("character")} for c in cast],
        "keywords": [k["name"] for k in details.get("keywords", {}).get("keywords", [])],
        "fetched_at": datetime.now().isoformat()
    }


def load_progress() -> dict:
    """진행 상태 로드"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"processed_ids": [], "movies": []}


def save_progress(progress: dict):
    """진행 상태 저장"""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def save_final_output(movies: list):
    """최종 결과 저장"""
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "total_count": len(movies),
            "found_count": sum(1 for m in movies if m.get("tmdb_found")),
            "not_found_count": sum(1 for m in movies if not m.get("tmdb_found")),
            "last_updated": datetime.now().isoformat(),
            "movies": movies
        }, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="TMDB 영화 정보 수집")
    parser.add_argument("--test", action="store_true", help="테스트 모드 (10개만)")
    parser.add_argument("--reset", action="store_true", help="진행 상태 초기화")
    args = parser.parse_args()
    
    print("\n🎬 TMDB 영화 정보 수집")
    print(f"⏰ 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # API 키 확인
    api_key = get_api_key()
    if not api_key:
        return
    
    print(f"✅ API 키 확인됨")
    
    # 진행 상태
    if args.reset and PROGRESS_FILE.exists():
        os.remove(PROGRESS_FILE)
        print("🗑️ 진행 상태 초기화됨")
    
    progress = load_progress()
    processed_ids = set(progress.get("processed_ids", []))
    movies = progress.get("movies", [])
    
    # 영화 목록 수집
    print("\n📋 영화 목록 수집 중...")
    all_movies = collect_unique_movies()
    print(f"  ✅ 총 {len(all_movies)}개 고유 영화")
    
    # 처리할 영화 필터링
    to_process = [m for m in all_movies if m["cine21_movie_id"] not in processed_ids]
    print(f"  📊 신규 처리 필요: {len(to_process)}개")
    
    if not to_process:
        print("  ℹ️ 처리할 영화가 없습니다.")
        save_final_output(movies)
        return
    
    if args.test:
        to_process = to_process[:10]
        print(f"  🧪 테스트 모드: 10개만 처리")
    
    # TMDB 조회
    print("\n📡 TMDB 정보 수집 중...")
    session = requests.Session()
    
    found = 0
    not_found = 0
    
    for idx, movie in enumerate(to_process):
        result = process_movie(movie, api_key, session)
        
        if result:
            movies.append(result)
            processed_ids.add(movie["cine21_movie_id"])
            
            if result.get("tmdb_found"):
                found += 1
            else:
                not_found += 1
        
        # 진행 상황 출력 및 저장
        if (idx + 1) % 10 == 0 or idx == len(to_process) - 1:
            print(f"  📰 진행: {idx + 1}/{len(to_process)} (발견: {found}, 미발견: {not_found})")
            progress["processed_ids"] = list(processed_ids)
            progress["movies"] = movies
            save_progress(progress)
    
    # 최종 저장
    save_final_output(movies)
    
    print(f"\n✅ 완료!")
    print(f"  📊 총 {len(movies)}개 영화")
    print(f"  ✅ TMDB 발견: {sum(1 for m in movies if m.get('tmdb_found'))}개")
    print(f"  ❌ 미발견: {sum(1 for m in movies if not m.get('tmdb_found'))}개")
    print(f"  📁 저장: {OUTPUT_FILE}")
    print(f"⏰ 종료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
