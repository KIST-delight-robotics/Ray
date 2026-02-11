"""
Cine21 웹 크롤러 (Selenium 버전)

영화읽기, 영화비평, 씨네21 리뷰 섹션의 기사를 크롤링합니다.

사용법:
    # 테스트 모드 (각 카테고리 2페이지씩)
    python cine21_crawler.py --test-mode
    
    # 전체 크롤링
    python cine21_crawler.py
    
    # 특정 카테고리만
    python cine21_crawler.py --category review
    
    # 진행 상태 초기화
    python cine21_crawler.py --reset
"""

import json
import os
import re
import sys
import ssl
import time
import argparse
import urllib3
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
except ImportError:
    print("필요한 패키지를 설치해주세요:")
    print("  pip install selenium beautifulsoup4 lxml requests")
    sys.exit(1)

# SSL 경고 비활성화
urllib3.disable_warnings()
# SSL 인증서 검증 비활성화 (기업 환경에서 필요할 수 있음)
ssl._create_default_https_context = ssl._create_unverified_context


# ============================================================================
# 설정
# ============================================================================

BASE_URL = "https://cine21.com"

CATEGORIES = {
    "reading": {
        "name": "영화읽기",
        "section": "005004001",
        "max_pages": 78,
        "category_type": "critique"
    },
    "critique": {
        "name": "영화비평",
        "section": "005004016",
        "max_pages": 76,
        "category_type": "critique"
    },
    "review": {
        "name": "씨네21 리뷰",
        "section": "002001001",
        "max_pages": 974,
        "category_type": "review"
    }
}

# 병렬 처리 설정
MAX_CONCURRENT_REQUESTS = 5  # 동시 요청 수
REQUEST_DELAY = 1.0  # 요청 간 딜레이 (초)
PAGE_LOAD_TIMEOUT = 20  # 페이지 로드 타임아웃 (초)

# 경로 설정
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "crawled"
PROGRESS_FILE = DATA_DIR / "progress.json"

# HTTP 헤더
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


# ============================================================================
# 데이터 클래스
# ============================================================================

@dataclass
class Article:
    mag_id: str
    title: str
    author: str
    author_id: str
    date: str
    section: str
    magazine_issue: str
    category: str
    category_name: str
    content: str
    related_articles: list
    related_movies: list  # 관련 영화 정보 [{movie_id, title, year}, ...]
    url: str
    crawled_at: str


# ============================================================================
# 진행 상태 관리
# ============================================================================

def load_progress() -> dict:
    """저장된 진행 상태 로드"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"categories": {}, "crawled_ids": []}


def save_progress(progress: dict):
    """진행 상태 저장"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def save_articles(articles: list, category_key: str):
    """수집한 기사 저장"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = DATA_DIR / f"cine21_{category_key}.json"
    
    existing = []
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            existing = data.get("articles", [])
    
    existing_ids = {a["mag_id"] for a in existing}
    new_articles = [a for a in articles if a["mag_id"] not in existing_ids]
    all_articles = existing + new_articles
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "category": CATEGORIES[category_key]["name"],
            "total_count": len(all_articles),
            "last_updated": datetime.now().isoformat(),
            "articles": all_articles
        }, f, ensure_ascii=False, indent=2)
    
    print(f"  💾 저장됨: {output_file.name} (신규 {len(new_articles)}개, 총 {len(all_articles)}개)")
    return len(new_articles)


# ============================================================================
# 크롤링 함수
# ============================================================================

def create_driver() -> webdriver.Chrome:
    """Selenium WebDriver 생성 - 로컬 Chrome/Edge 사용"""
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument(f"user-agent={HEADERS['User-Agent']}")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--ignore-ssl-errors")
    options.add_argument("--allow-insecure-localhost")
    
    # 시스템에 설치된 Chrome 사용
    try:
        driver = webdriver.Chrome(options=options)
        return driver
    except Exception as e:
        print(f"  ℹ️ Chrome 드라이버 자동 감지 실패: {e}")
        print("  ℹ️ Edge 브라우저로 시도합니다...")
        
        # Edge 시도
        from selenium.webdriver.edge.options import Options as EdgeOptions
        from selenium.webdriver.edge.service import Service as EdgeService
        
        edge_options = EdgeOptions()
        edge_options.add_argument("--headless=new")
        edge_options.add_argument("--no-sandbox")
        edge_options.add_argument("--disable-dev-shm-usage")
        edge_options.add_argument("--disable-gpu")
        edge_options.add_argument(f"user-agent={HEADERS['User-Agent']}")
        edge_options.add_argument("--ignore-certificate-errors")
        
        return webdriver.Edge(options=edge_options)


def get_article_ids_from_page(driver, section: str, page_num: int) -> list:
    """목록 페이지에서 기사 ID 추출"""
    url = f"{BASE_URL}/news/section/?section={section}&p={page_num}"
    
    try:
        driver.get(url)
        WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(1.5)
        
        soup = BeautifulSoup(driver.page_source, "lxml")
        
        article_ids = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            match = re.search(r"mag_id=(\d+)", href)
            if match:
                article_ids.append(match.group(1))
        
        return list(dict.fromkeys(article_ids))
        
    except Exception as e:
        print(f"  ⚠️ 페이지 {page_num} 로드 실패: {e}")
        return []


def get_article_detail(mag_id: str, category_info: dict, session: requests.Session) -> Optional[dict]:
    """기사 상세 페이지에서 정보 추출"""
    url = f"{BASE_URL}/news/view/?mag_id={mag_id}"
    
    try:
        r = session.get(url, headers=HEADERS, verify=False, timeout=15)
        if r.status_code != 200:
            return None
            
        soup = BeautifulSoup(r.text, "lxml")
        
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text().strip()
            if " - " in title:
                title = title.split(" - ")[0].strip()
        
        author = ""
        author_id = ""
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "/db/writer/info/" in href:
                author = link.get_text().strip()
                match = re.search(r"pre_code=(\w+)", href)
                if match:
                    author_id = match.group(1)
                break
        
        date = ""
        text = soup.get_text()
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
        if date_match:
            date = date_match.group(1)
        
        section = ""
        magazine_issue = ""
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "/news/section/" in href and "section=" in href:
                section = link.get_text().strip()
            if "/db/mag/content/" in href:
                magazine_issue = link.get_text().strip()
        
        paragraphs = soup.find_all("p")
        texts = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30]
        content_text = "\n\n".join(texts)
        
        related_articles = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            match = re.search(r"mag_id=(\d+)", href)
            if match and match.group(1) != mag_id:
                related_articles.append(match.group(1))
        related_articles = list(dict.fromkeys(related_articles))[:5]
        
        # 관련 영화 추출 (.list_with_upthumb_item 에서)
        related_movies = []
        for item in soup.select(".list_with_upthumb_item a"):
            href = item.get("href", "")
            if "/movie/info/?movie_id=" in href or "movie_id=" in href:
                movie_id_match = re.search(r"movie_id=(\d+)", href)
                movie_id = movie_id_match.group(1) if movie_id_match else None
                title_el = item.select_one(".title")
                movie_title = title_el.text.strip() if title_el else None
                year_el = item.select_one(".etc_info p")
                year = year_el.text.strip().replace("(", "").replace(")", "") if year_el else None
                
                if movie_id:
                    related_movies.append({
                        "movie_id": movie_id,
                        "title": movie_title,
                        "year": year
                    })
        
        article = Article(
            mag_id=mag_id,
            title=title,
            author=author,
            author_id=author_id,
            date=date,
            section=section,
            magazine_issue=magazine_issue,
            category=category_info["category_type"],
            category_name=category_info["name"],
            content=content_text,
            related_articles=related_articles,
            related_movies=related_movies,
            url=url,
            crawled_at=datetime.now().isoformat()
        )
        
        return asdict(article)
        
    except Exception as e:
        print(f"  ⚠️ 기사 {mag_id} 로드 실패: {e}")
        return None


def crawl_category(category_key: str, max_pages: Optional[int] = None):
    """카테고리 전체 크롤링"""
    category = CATEGORIES[category_key]
    section = category["section"]
    total_pages = max_pages or category["max_pages"]
    
    print(f"\n{'='*60}")
    print(f"📂 {category['name']} 크롤링 시작 ({total_pages}페이지)")
    print(f"{'='*60}")
    
    progress = load_progress()
    cat_progress = progress.get("categories", {}).get(category_key, {})
    start_page = cat_progress.get("last_page", 0) + 1
    crawled_ids = set(progress.get("crawled_ids", []))
    
    if start_page > 1:
        print(f"  📌 이전 진행 상태에서 재개: {start_page}페이지부터")
    
    print(f"\n📋 1단계: 기사 목록 수집 중...")
    driver = create_driver()
    new_ids = []
    
    try:
        for page_num in range(start_page, total_pages + 1):
            ids = get_article_ids_from_page(driver, section, page_num)
            new_found = [id for id in ids if id not in crawled_ids]
            new_ids.extend(new_found)
            
            print(f"  📄 페이지 {page_num}/{total_pages}: {len(ids)}개 발견 (신규: {len(new_found)}개)")
            
            crawled_ids.update(ids)
            if "categories" not in progress:
                progress["categories"] = {}
            progress["categories"][category_key] = {
                "last_page": page_num,
                "updated_at": datetime.now().isoformat()
            }
            progress["crawled_ids"] = list(crawled_ids)
            
            if page_num % 5 == 0:
                save_progress(progress)
            
            time.sleep(REQUEST_DELAY)
        
        save_progress(progress)
        
    finally:
        driver.quit()
    
    print(f"  ✅ 기사 ID 수집 완료: 총 {len(new_ids)}개 신규")
    
    if not new_ids:
        print("  ℹ️ 새로 크롤링할 기사가 없습니다.")
        return
    
    print(f"\n📝 2단계: 기사 상세 크롤링 중...")
    articles = []
    session = requests.Session()
    
    def fetch_article(args):
        idx, mag_id = args
        time.sleep(REQUEST_DELAY / MAX_CONCURRENT_REQUESTS)
        return get_article_detail(mag_id, category, session)
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        futures = {executor.submit(fetch_article, (idx, mag_id)): (idx, mag_id) 
                   for idx, mag_id in enumerate(new_ids)}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result:
                articles.append(result)
            
            if completed % 20 == 0 or completed == len(new_ids):
                print(f"  📰 진행: {completed}/{len(new_ids)} ({len(articles)}개 성공)")
    
    if articles:
        save_articles(articles, category_key)
        print(f"  ✅ {category['name']} 크롤링 완료: {len(articles)}개 기사")


def main(args):
    """메인 함수"""
    print("\n🎬 Cine21 크롤러 시작")
    print(f"⏰ 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.reset:
        if PROGRESS_FILE.exists():
            os.remove(PROGRESS_FILE)
        print("🗑️ 진행 상태 초기화됨")
    
    categories_to_crawl = []
    if args.category:
        if args.category in CATEGORIES:
            categories_to_crawl = [args.category]
        else:
            print(f"❌ 알 수 없는 카테고리: {args.category}")
            print(f"   사용 가능: {', '.join(CATEGORIES.keys())}")
            return
    else:
        categories_to_crawl = list(CATEGORIES.keys())
    
    max_pages = 2 if args.test_mode else None
    if args.test_mode:
        print("🧪 테스트 모드: 각 카테고리 2페이지씩만 크롤링")
    
    for category_key in categories_to_crawl:
        crawl_category(category_key, max_pages)
    
    print(f"\n✅ 크롤링 완료!")
    print(f"⏰ 종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 데이터 저장 위치: {DATA_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cine21 영화 평론/리뷰 크롤러")
    parser.add_argument("--test-mode", action="store_true", help="테스트 모드 (각 카테고리 2페이지)")
    parser.add_argument("--category", type=str, help="특정 카테고리만 (reading, critique, review)")
    parser.add_argument("--reset", action="store_true", help="진행 상태 초기화")
    
    args = parser.parse_args()
    main(args)
