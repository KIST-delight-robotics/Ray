"""
RAG 검색 테스트 UI
브라우저에서 RAG 검색 결과를 확인할 수 있는 간단한 웹 인터페이스
- RAG 검색 (임베딩 기반)
- 메타데이터 검색 (제목, movie_id 등)
- 통계 정보
"""

import os
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from flask import Flask, render_template_string, request, jsonify
from rag.retriever import init_db, get_collection_stats

app = Flask(__name__)

_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        from rag import retriever
        _vectorstore = retriever._vectorstore
    return _vectorstore

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>RAG 테스트 대시보드</title>
    <meta charset="utf-8">
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', sans-serif; 
            max-width: 1400px; 
            margin: 0 auto; 
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }
        h1 { color: #00d9ff; margin-bottom: 5px; }
        h2 { color: #888; font-size: 14px; margin-top: 0; font-weight: normal; }
        
        .tabs {
            display: flex;
            gap: 5px;
            margin-bottom: 20px;
        }
        .tab {
            padding: 12px 24px;
            background: #16213e;
            border: none;
            border-radius: 8px 8px 0 0;
            color: #888;
            cursor: pointer;
            font-size: 14px;
        }
        .tab.active { background: #0066ff; color: white; }
        .tab:hover { background: #1a3a5c; }
        .tab.active:hover { background: #0066ff; }
        
        .panel { display: none; }
        .panel.active { display: block; }
        
        .search-box {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        input[type="text"] {
            flex: 1;
            padding: 12px 16px;
            font-size: 16px;
            border: 2px solid #333;
            border-radius: 8px;
            background: #16213e;
            color: #fff;
        }
        input:focus { outline: none; border-color: #00d9ff; }
        select, button {
            padding: 12px 20px;
            font-size: 14px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
        select { background: #16213e; color: #fff; border: 2px solid #333; }
        button { 
            background: linear-gradient(135deg, #00d9ff, #0066ff);
            color: white;
            font-weight: bold;
        }
        button:hover { opacity: 0.9; }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: #16213e;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }
        .stat-value { font-size: 32px; color: #00d9ff; font-weight: bold; }
        .stat-label { color: #888; margin-top: 5px; }
        
        .chart-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .chart-card {
            background: #16213e;
            padding: 20px;
            border-radius: 12px;
        }
        .chart-title { font-size: 16px; margin-bottom: 15px; color: #00d9ff; }
        .bar-chart { display: flex; flex-direction: column; gap: 8px; }
        .bar-row { display: flex; align-items: center; gap: 10px; }
        .bar-label { width: 120px; font-size: 13px; text-align: right; }
        .bar-bg { flex: 1; height: 24px; background: #0d1b2a; border-radius: 4px; overflow: hidden; }
        .bar-fill { height: 100%; background: linear-gradient(90deg, #00d9ff, #0066ff); border-radius: 4px; display: flex; align-items: center; justify-content: flex-end; padding-right: 8px; font-size: 12px; }
        
        .result {
            background: #16213e;
            border-radius: 12px;
            margin-bottom: 10px;
            border-left: 4px solid #00d9ff;
            overflow: hidden;
        }
        .result.basic_info { border-left-color: #00ff88; }
        .result.critique { border-left-color: #ff6b6b; }
        .result.review { border-left-color: #ffd93d; }
        .result-header {
            padding: 12px 16px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .result-header:hover { background: rgba(255,255,255,0.05); }
        .meta { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 5px; }
        .tag { background: #333; padding: 3px 8px; border-radius: 10px; font-size: 11px; }
        .tag.category { background: #0066ff; }
        .tag.matched { background: #ff6b6b; }
        .expand-icon { font-size: 18px; transition: transform 0.2s; }
        .result.expanded .expand-icon { transform: rotate(180deg); }
        .result-body { display: none; padding: 0 16px 16px; }
        .result.expanded .result-body { display: block; }
        .title { font-size: 15px; font-weight: bold; color: #00d9ff; }
        .preview { color: #888; font-size: 13px; margin-top: 4px; }
        .chunk {
            background: #0d1b2a;
            padding: 10px;
            border-radius: 6px;
            margin-bottom: 6px;
            font-size: 13px;
            line-height: 1.5;
            border: 2px solid transparent;
        }
        .chunk.matched { border-color: #ff6b6b; background: #1a0a0a; }
        .chunk-header { font-size: 10px; color: #666; margin-bottom: 4px; }
        .loading { text-align: center; padding: 40px; color: #888; }
        .no-results { text-align: center; padding: 40px; color: #666; }
    </style>
</head>
<body>
    <h1>🎬 RAG 테스트 대시보드</h1>
    <h2 id="db-info">DB 연결 중...</h2>
    
    <div class="tabs">
        <button class="tab active" onclick="switchTab('rag')">🔍 RAG 검색</button>
        <button class="tab" onclick="switchTab('meta')">📋 메타데이터 검색</button>
        <button class="tab" onclick="switchTab('stats')">📊 통계</button>
    </div>
    
    <!-- RAG 검색 패널 -->
    <div id="panel-rag" class="panel active">
        <div class="search-box">
            <input type="text" id="rag-query" placeholder="검색어 입력 (예: 봉준호, 우울한 영화, 인사이드 아웃)" />
            <select id="rag-intent">
                <option value="vibe">vibe (전체)</option>
                <option value="fact">fact (기본정보)</option>
                <option value="critique">critique (평론)</option>
            </select>
            <select id="rag-topk">
                <option value="3">3개</option>
                <option value="5">5개</option>
                <option value="10">10개</option>
            </select>
            <button onclick="searchRAG()">검색</button>
        </div>
        <div id="rag-results"></div>
    </div>
    
    <!-- 메타데이터 검색 패널 -->
    <div id="panel-meta" class="panel">
        <div class="search-box">
            <select id="meta-field">
                <option value="title">제목</option>
                <option value="movie_id">movie_id</option>
                <option value="author">저자</option>
                <option value="doc_id">doc_id</option>
            </select>
            <input type="text" id="meta-query" placeholder="검색할 값 입력 (예: 기생충, tmdb_496243)" />
            <select id="meta-topk">
                <option value="10">10개</option>
                <option value="20">20개</option>
                <option value="50">50개</option>
            </select>
            <button onclick="searchMeta()">검색</button>
        </div>
        <div id="meta-results"></div>
    </div>
    
    <!-- 통계 패널 -->
    <div id="panel-stats" class="panel">
        <div id="stats-content"><div class="loading">통계 로딩 중...</div></div>
    </div>

    <script>
        // 초기화
        fetch('/stats')
            .then(r => r.json())
            .then(data => {
                document.getElementById('db-info').innerHTML = 
                    `📂 컬렉션: ${data.name} | 청크 수: ${data.count?.toLocaleString() || 'N/A'}`;
            });

        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('panel-' + tab).classList.add('active');
            
            if (tab === 'stats') loadStats();
        }

        function toggleResult(el) {
            el.closest('.result').classList.toggle('expanded');
        }

        // RAG 검색
        function searchRAG() {
            const query = document.getElementById('rag-query').value;
            const intent = document.getElementById('rag-intent').value;
            const topk = document.getElementById('rag-topk').value;
            
            if (!query) return alert('검색어를 입력하세요');
            document.getElementById('rag-results').innerHTML = '<div class="loading">🔍 검색 중...</div>';
            
            fetch('/search/rag', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query, intent, top_k: parseInt(topk)})
            })
            .then(r => r.json())
            .then(data => renderResults('rag-results', data.results, true));
        }

        // 메타데이터 검색
        function searchMeta() {
            const field = document.getElementById('meta-field').value;
            const query = document.getElementById('meta-query').value;
            const topk = document.getElementById('meta-topk').value;
            
            if (!query) return alert('검색어를 입력하세요');
            document.getElementById('meta-results').innerHTML = '<div class="loading">🔍 검색 중...</div>';
            
            fetch('/search/meta', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({field, query, top_k: parseInt(topk)})
            })
            .then(r => r.json())
            .then(data => renderResults('meta-results', data.results, false));
        }

        function renderResults(containerId, results, showChunks) {
            let html = '';
            if (results && results.length > 0) {
                results.forEach(result => {
                    let chunksHtml = '';
                    if (showChunks && result.all_chunks) {
                        chunksHtml = result.all_chunks.map((chunk, idx) => `
                            <div class="chunk ${chunk.is_matched ? 'matched' : ''}">
                                <div class="chunk-header">청크 #${idx} ${chunk.is_matched ? '⭐ 매칭' : ''}</div>
                                ${chunk.content}
                            </div>
                        `).join('');
                    }
                    
                    html += `
                        <div class="result ${result.category}">
                            <div class="result-header" onclick="toggleResult(this)">
                                <div>
                                    <div class="meta">
                                        <span class="tag category">${result.category}</span>
                                        <span class="tag">${result.movie_id || 'N/A'}</span>
                                        ${result.author ? `<span class="tag">✏️ ${result.author}</span>` : ''}
                                        ${result.matched_chunk_index !== undefined ? `<span class="tag matched">#${result.matched_chunk_index}</span>` : ''}
                                    </div>
                                    <div class="title">${result.title}</div>
                                    <div class="preview">${result.preview || ''}...</div>
                                </div>
                                <span class="expand-icon">▼</span>
                            </div>
                            <div class="result-body">${chunksHtml || result.content || ''}</div>
                        </div>
                    `;
                });
            } else {
                html = '<div class="no-results">검색 결과가 없습니다.</div>';
            }
            document.getElementById(containerId).innerHTML = html;
        }

        // 통계 로드
        function loadStats() {
            document.getElementById('stats-content').innerHTML = '<div class="loading">📊 통계 로딩 중...</div>';
            fetch('/analytics')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('stats-content').innerHTML = `
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-value">${data.total_chunks?.toLocaleString()}</div>
                                <div class="stat-label">총 청크 수</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">${data.total_docs?.toLocaleString()}</div>
                                <div class="stat-label">총 문서 수</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">${data.unique_movies?.toLocaleString()}</div>
                                <div class="stat-label">고유 영화 수</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">${data.unique_authors?.toLocaleString()}</div>
                                <div class="stat-label">고유 저자 수</div>
                            </div>
                        </div>
                        <div class="chart-row">
                            <div class="chart-card">
                                <div class="chart-title">📁 카테고리별 분포</div>
                                <div style="font-size:12px; color:#888; margin-bottom:8px;">📄 문서 수</div>
                                <div class="bar-chart" id="category-doc-chart"></div>
                                <div style="font-size:12px; color:#888; margin:12px 0 8px;">📦 청크 수</div>
                                <div class="bar-chart" id="category-chunk-chart"></div>
                            </div>
                            <div class="chart-card">
                                <div class="chart-title">✏️ 상위 저자</div>
                                <div class="bar-chart" id="author-chart"></div>
                            </div>
                        </div>
                    `;
                    
                    // 카테고리 - 문서 수 차트
                    const docMax = Math.max(...Object.values(data.by_category));
                    let docHtml = '';
                    for (const [cat, count] of Object.entries(data.by_category)) {
                        const pct = (count / docMax * 100).toFixed(0);
                        docHtml += `<div class="bar-row">
                            <div class="bar-label">${cat}</div>
                            <div class="bar-bg"><div class="bar-fill" style="width:${pct}%">${count.toLocaleString()}</div></div>
                        </div>`;
                    }
                    document.getElementById('category-doc-chart').innerHTML = docHtml;
                    
                    // 카테고리 - 청크 수 차트
                    const chunkMax = Math.max(...Object.values(data.chunks_by_category || {}));
                    let chunkHtml = '';
                    for (const [cat, count] of Object.entries(data.chunks_by_category || {})) {
                        const pct = (count / chunkMax * 100).toFixed(0);
                        chunkHtml += `<div class="bar-row">
                            <div class="bar-label">${cat}</div>
                            <div class="bar-bg"><div class="bar-fill" style="width:${pct}%">${count.toLocaleString()}</div></div>
                        </div>`;
                    }
                    document.getElementById('category-chunk-chart').innerHTML = chunkHtml;
                    
                    // 저자 차트
                    const authMax = data.top_authors[0]?.[1] || 1;
                    let authHtml = '';
                    for (const [author, count] of data.top_authors.slice(0, 10)) {
                        const pct = (count / authMax * 100).toFixed(0);
                        authHtml += `<div class="bar-row">
                            <div class="bar-label">${author || '(없음)'}</div>
                            <div class="bar-bg"><div class="bar-fill" style="width:${pct}%">${count}</div></div>
                        </div>`;
                    }
                    document.getElementById('author-chart').innerHTML = authHtml;
                });
        }

        // Enter 키 검색
        document.getElementById('rag-query').addEventListener('keypress', e => { if (e.key === 'Enter') searchRAG(); });
        document.getElementById('meta-query').addEventListener('keypress', e => { if (e.key === 'Enter') searchMeta(); });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/stats')
def stats():
    return jsonify(get_collection_stats())

@app.route('/analytics')
def analytics():
    vs = get_vectorstore()
    if not vs:
        return jsonify({'error': 'DB not initialized'})
    
    try:
        collection = vs._collection
        all_data = collection.get(include=["metadatas"])
        metadatas = all_data.get("metadatas", [])
        
        # 문서 단위로 통계 계산 (doc_id 기준 중복 제거)
        doc_info = {}  # doc_id -> {category, author, movie_id}
        
        for meta in metadatas:
            doc_id = meta.get("doc_id", "")
            if doc_id and doc_id not in doc_info:
                doc_info[doc_id] = {
                    "category": meta.get("category", "other"),
                    "author": meta.get("author", ""),
                    "movie_id": meta.get("movie_id", "")
                }
        
        # 카테고리별 문서 수
        categories = Counter()
        chunks_by_category = Counter()  # 청크 기준
        authors = Counter()
        movie_ids = set()
        
        # 청크별 카운트
        for meta in metadatas:
            chunks_by_category[meta.get("category", "other")] += 1
        
        for doc_id, info in doc_info.items():
            categories[info["category"]] += 1
            if info["author"]:
                authors[info["author"]] += 1
            if info["movie_id"]:
                movie_ids.add(info["movie_id"])
        
        return jsonify({
            "total_chunks": len(metadatas),
            "total_docs": len(doc_info),
            "unique_movies": len(movie_ids),
            "unique_authors": len(authors),
            "by_category": dict(categories),
            "chunks_by_category": dict(chunks_by_category),
            "top_authors": authors.most_common(15)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/search/rag', methods=['POST'])
def search_rag():
    data = request.json
    query = data.get('query', '')
    intent = data.get('intent', 'vibe')
    top_k = data.get('top_k', 5)
    
    vs = get_vectorstore()
    if not vs:
        return jsonify({'error': 'DB not initialized', 'results': []})
    
    filter_dict = None
    if intent == "fact":
        filter_dict = {"category": "basic_info"}
    elif intent == "critique":
        filter_dict = {"category": {"$in": ["critique", "interview"]}}
    
    if filter_dict:
        docs = vs.similarity_search(query, k=top_k, filter=filter_dict)
    else:
        docs = vs.similarity_search(query, k=top_k)
    
    results = []
    seen_doc_ids = set()
    
    for doc in docs:
        doc_id = doc.metadata.get("doc_id", "")
        if not doc_id or doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        
        matched_chunk_index = doc.metadata.get("chunk_index", 0)
        
        try:
            all_chunks_docs = vs.similarity_search("", k=50, filter={"doc_id": doc_id})
            all_chunks_docs = sorted(all_chunks_docs, key=lambda x: x.metadata.get("chunk_index", 0))
        except:
            all_chunks_docs = [doc]
        
        all_chunks = []
        for chunk_doc in all_chunks_docs:
            chunk_idx = chunk_doc.metadata.get("chunk_index", 0)
            all_chunks.append({
                "index": chunk_idx,
                "content": chunk_doc.page_content,
                "is_matched": chunk_idx == matched_chunk_index
            })
        
        results.append({
            "doc_id": doc_id,
            "title": doc.metadata.get("title", "제목 없음"),
            "category": doc.metadata.get("category", ""),
            "movie_id": doc.metadata.get("movie_id", ""),
            "author": doc.metadata.get("author", ""),
            "matched_chunk_index": matched_chunk_index,
            "preview": doc.page_content[:150],
            "all_chunks": all_chunks
        })
    
    return jsonify({'results': results})

@app.route('/search/meta', methods=['POST'])
def search_meta():
    data = request.json
    field = data.get('field', 'title')
    query = data.get('query', '')
    top_k = data.get('top_k', 10)
    
    vs = get_vectorstore()
    if not vs:
        return jsonify({'error': 'DB not initialized', 'results': []})
    
    try:
        collection = vs._collection
        
        # 메타데이터 필터로 검색
        if field in ["movie_id", "doc_id", "author"]:
            # 정확히 일치
            filter_dict = {field: query}
        else:
            # 제목은 부분 일치가 어려우므로 전체 검색 후 필터
            filter_dict = None
        
        if filter_dict:
            all_data = collection.get(where=filter_dict, include=["metadatas", "documents"], limit=top_k)
        else:
            # 전체에서 제목 검색 (느릴 수 있음)
            all_data = collection.get(include=["metadatas", "documents"])
        
        results = []
        seen_docs = set()
        
        metadatas = all_data.get("metadatas", [])
        documents = all_data.get("documents", [])
        
        for i, meta in enumerate(metadatas):
            doc_id = meta.get("doc_id", "")
            
            # 제목 검색인 경우 필터링
            if field == "title" and query.lower() not in meta.get("title", "").lower():
                continue
            
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            
            if len(results) >= top_k:
                break
            
            content = documents[i] if i < len(documents) else ""
            
            results.append({
                "doc_id": doc_id,
                "title": meta.get("title", "제목 없음"),
                "category": meta.get("category", ""),
                "movie_id": meta.get("movie_id", ""),
                "author": meta.get("author", ""),
                "preview": content[:200] if content else "",
                "content": content
            })
        
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e), 'results': []})


if __name__ == '__main__':
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY 환경변수 필요")
        sys.exit(1)
    
    db_path = str(PROJECT_ROOT / "data" / "chroma_db")
    print(f"📂 DB 경로: {db_path}")
    
    init_db(db_path, api_key)
    
    print("\n🚀 RAG 테스트 대시보드: http://localhost:5050")
    app.run(host='0.0.0.0', port=5050, debug=False)
