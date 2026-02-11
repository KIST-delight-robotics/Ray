"""TMDB 검색 로직 테스트"""
import requests
import urllib3
import os
import sys

urllib3.disable_warnings()

# API 키
api_key = os.getenv('TMDB_API_KEY')
if not api_key:
    sys.path.insert(0, str(os.path.dirname(os.path.dirname(__file__))))
    from config import TMDB_API_KEY
    api_key = TMDB_API_KEY


def find_best_match(results, target_title, target_year):
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
                return r, "1순위: 제목+연도 완전 일치"
    
    # 2순위: 제목 완전 일치 (연도 무시)
    for r in results:
        r_title = (r.get("title") or "").strip().lower()
        r_orig = (r.get("original_title") or "").strip().lower()
        
        if r_title == target_title_norm or r_orig == target_title_norm:
            return r, "2순위: 제목 완전 일치"
    
    # 3순위: 제목 포함 + 연도 일치
    if target_year and target_year.isdigit():
        for r in results:
            r_title = (r.get("title") or "").strip().lower()
            r_year = (r.get("release_date") or "")[:4]
            
            if target_title_norm in r_title and r_year == target_year:
                return r, "3순위: 제목 포함 + 연도 일치"
    
    # 4순위: 첫 번째 결과
    return results[0], "4순위: 첫 번째 결과 (fallback)"


def test_search(title, year=None):
    """TMDB 검색 테스트"""
    print(f"\n{'='*50}")
    print(f"🔍 검색어: {title} (연도: {year or '없음'})")
    print("="*50)
    
    session = requests.Session()
    params = {
        "api_key": api_key,
        "query": title,
        "language": "ko-KR"
    }
    
    r = session.get("https://api.themoviedb.org/3/search/movie", params=params, verify=False)
    results = r.json().get("results", [])
    
    print(f"\n📋 검색 결과 ({len(results)}개 중 상위 5개):")
    for i, m in enumerate(results[:5]):
        movie_year = (m.get("release_date") or "")[:4]
        print(f"  {i+1}. {m.get('title')} ({movie_year}) - TMDB ID: {m.get('id')}")
    
    # 개선된 매칭
    if results:
        best, reason = find_best_match(results, title, year)
        movie_year = (best.get("release_date") or "")[:4]
        print(f"\n✅ 선택됨 ({reason}):")
        print(f"   제목: {best.get('title')} ({movie_year})")
        print(f"   TMDB ID: {best.get('id')}")
        print(f"   원제: {best.get('original_title')}")


if __name__ == "__main__":
    # 테스트 케이스들
    test_search("신세계", "2013")  # 박훈정 감독 신세계
    test_search("기생충", "2019")  # 봉준호 감독
    test_search("올드보이", "2003")  # 박찬욱 감독
