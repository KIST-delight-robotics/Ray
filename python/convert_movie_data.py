# -*- coding: utf-8 -*-
"""
영화 데이터셋 → RAG JSON 변환 스크립트
IMDB CSV와 TMDB XLSX 파일을 RAG용 JSON으로 변환합니다.
"""

import pandas as pd
import json
import re
from pathlib import Path


def clean_text(text):
    """텍스트 정리"""
    if pd.isna(text):
        return ""
    return str(text).strip()


def convert_tmdb_xlsx(xlsx_path: str, output_path: str, max_count: int = 100):
    """
    Movies_datas.xlsx → RAG JSON 변환
    Overview(줄거리)가 있어서 basic_info로 적합
    """
    df = pd.read_excel(xlsx_path)
    
    documents = []
    count = 0
    
    for idx, row in df.iterrows():
        title = clean_text(row.get('Movie_name', ''))
        overview = clean_text(row.get('Overview', ''))
        
        # 줄거리가 없으면 건너뜀
        if not overview or len(overview) < 20:
            continue
        
        # 영어가 아닌 제목 필터링 (선택적)
        # if not re.match(r'^[A-Za-z0-9\s\-\':,\.!?]+$', title):
        #     continue
        
        director = clean_text(row.get('Director', ''))
        genre = clean_text(row.get('Genre', ''))
        rating = row.get('Rating', '')
        release_date = clean_text(row.get('Release_date', ''))
        runtime = clean_text(row.get('Run_time', ''))
        
        # 컨텐츠 구성
        content_parts = [overview]
        if director:
            content_parts.append(f"감독: {director}")
        if genre:
            content_parts.append(f"장르: {genre}")
        if rating:
            content_parts.append(f"평점: {rating}")
        if runtime:
            content_parts.append(f"상영시간: {runtime}")
        
        doc = {
            "id": f"tmdb_{idx}",
            "movie_id": f"tmdb_{idx}",
            "title": title,
            "director": director,
            "category": "basic_info",
            "content": " | ".join(content_parts)
        }
        
        documents.append(doc)
        count += 1
        
        if count >= max_count:
            break
    
    # JSON 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    print(f"✅ {count}개 영화 데이터 변환 완료: {output_path}")
    return count


def convert_imdb_csv(csv_path: str, output_path: str, max_count: int = 100):
    """
    IMDb_Dataset_Composite_Cleaned.csv → RAG JSON 변환
    평점/메타 정보 위주
    """
    df = pd.read_csv(csv_path)
    
    documents = []
    count = 0
    
    for idx, row in df.iterrows():
        title = clean_text(row.get('Title', ''))
        
        if not title:
            continue
        
        director = clean_text(row.get('Director', ''))
        cast = clean_text(row.get('Stars/Cast', row.get('Cast', '')))
        year = row.get('Year', '')
        rating = row.get('IMDb Rating', '')
        metascore = row.get('MetaScore', '')
        genres = clean_text(row.get('Genres', ''))
        duration = row.get('Duration (minutes)', '')
        
        # 컨텐츠 구성 (줄거리 없음 → 메타 정보로 구성)
        content_parts = [f"영화 제목: {title}"]
        if year:
            content_parts.append(f"개봉년도: {year}")
        if director:
            content_parts.append(f"감독: {director}")
        if cast:
            content_parts.append(f"출연: {cast}")
        if genres:
            content_parts.append(f"장르: {genres}")
        if rating:
            content_parts.append(f"IMDb 평점: {rating}")
        if metascore:
            content_parts.append(f"MetaScore: {metascore}")
        if duration:
            content_parts.append(f"상영시간: {duration}분")
        
        doc = {
            "id": f"imdb_{idx}",
            "movie_id": f"imdb_{idx}",
            "title": title,
            "director": director,
            "category": "basic_info",
            "content": " | ".join(content_parts)
        }
        
        documents.append(doc)
        count += 1
        
        if count >= max_count:
            break
    
    # JSON 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    print(f"✅ {count}개 영화 데이터 변환 완료: {output_path}")
    return count


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="영화 데이터셋 → RAG JSON 변환")
    parser.add_argument("--source", choices=["tmdb", "imdb", "both"], default="tmdb",
                       help="데이터 소스 선택")
    parser.add_argument("--max", type=int, default=100, help="최대 영화 수")
    parser.add_argument("--output", default="../data/movies_rag.json", help="출력 파일")
    
    args = parser.parse_args()
    
    data_dir = Path(__file__).parent.parent / "data"
    
    if args.source == "tmdb":
        convert_tmdb_xlsx(
            str(data_dir / "Movies_datas.xlsx"),
            args.output,
            max_count=args.max
        )
    elif args.source == "imdb":
        convert_imdb_csv(
            str(data_dir / "IMDb_Dataset_Composite_Cleaned.csv"),
            args.output,
            max_count=args.max
        )
    else:  # both
        # TMDB 먼저 (줄거리 있음)
        docs1 = convert_tmdb_xlsx(
            str(data_dir / "Movies_datas.xlsx"),
            str(data_dir / "tmdb_movies.json"),
            max_count=args.max // 2
        )
        # IMDB
        docs2 = convert_imdb_csv(
            str(data_dir / "IMDb_Dataset_Composite_Cleaned.csv"),
            str(data_dir / "imdb_movies.json"),
            max_count=args.max // 2
        )
        print(f"총 {docs1 + docs2}개 영화 변환 완료")
