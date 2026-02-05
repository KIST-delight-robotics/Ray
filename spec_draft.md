# 프로젝트 논의내용

## 1. 프로젝트 개요
*   **서비스명 (가칭):** 시네필 봇 (The Cinephile Bot)
*   **핵심 컨셉:** 단순 정보 전달자가 아닌, **확실한 취향과 깊이 있는 식견을 가진 영화/음악 덕후 친구**.
*   **주요 기능:**
    1.  **발견 (Discovery):** 사용자의 기분/상황에 맞춘 감성적 콘텐츠 추천.
    2.  **심층 대화 (Deep Dive):** 평론, 인터뷰 등을 인용한 깊이 있는 대화 및 해석 공유.
    3.  **위로 (Comfort):** 영화적 메타포를 활용한 일상 대화 및 공감.

---

## 2. 시스템 아키텍처 & 기술 스택
*   **LLM Model:** OpenAI `GPT-5.1` (또는 `gpt-4.1-mini`)
*   **Orchestration:** **LangChain** (Python) - RAG 파이프라인 및 문서 분할 관리.
*   **Vector DB:** **ChromaDB** (Local) - 단일 컬렉션 운용, 메타데이터 필터링 활용.
*   **Embedding:** OpenAI `text-embedding-3-small`.
*   **Voice Pipeline:** (기존 보유) STT -> LLM -> TTS.

---

## 3. 데이터 파이프라인 (Data Pipeline)

### 3.1. 데이터 수집 및 구조 (Schema)
모든 데이터는 **단일 컬렉션(Single Collection)**에 저장하되, **메타데이터(`category`)**로 유형을 구분합니다.

**JSON 문서 구조 예시:**
```json
{
  "id": "unique_doc_id",
  "metadata": {
    "movie_id": "tmdb_12345",      // 영화 식별자 (관계형 연결용), 기본 정보가 아닌 경우 복수 ID 가능 (예: 여러 영화가 연관된 평론/인터뷰)
    "title": "헤어질 결심",
    "director": "박찬욱",
    "category": "critique",        // [중요] 유형 구분: 'basic_info' | 'critique' | 'interview'
  },
  "page_content": "이동진 평론가가 말하는 헤어질 결심의 미장센... (검색될 텍스트 본문)"
}
```

### 3.2. 데이터 유형별 처리 전략
| 유형 | 역할 (Role) | 포함 내용 | 처리 방식 |
| :--- | :--- | :--- | :--- |
| **기본 정보 (Basic Info)** | **검색 앵커 (Anchor)** | 줄거리, 출연진, 개봉일, 장르 | 모호한 검색어("그 숟가락 영화") 방어용. 필수 포함. |
| **평론/리뷰 (Critique)** | **페르소나의 '뇌'** | 평론가 해석, 관람 포인트, 칼럼 | 로봇의 '취향'과 '식견'을 담당하는 핵심 데이터. |
| **인터뷰 (Interview)** | **비하인드 스토리** | 감독/배우 Q&A, 제작 비화 | TMI 및 깊이 있는 대화 재료. |

### 3.3. 임베딩 및 청크 전략 (Chunking)
*   **도구:** LangChain `RecursiveCharacterTextSplitter`
*   **설정:**
    *   `chunk_size`: **500자** (맥락 유지 가능한 최소 단위)
    *   `chunk_overlap`: **50~100자** (문장 끊김 방지)
*   **고급 전략 (권장):** **Parent Document Retriever**
    *   검색은 작은 청크 단위로 하되, LLM에게는 해당 청크가 포함된 **'문서 전체'** 혹은 **'큰 문단'**을 제공하여 문맥(Context) 완결성 보장.

---

## 4. 검색 및 툴 전략 (RAG Logic)

### 4.1. 툴(Tool) 정의
LLM이 혼란스럽지 않도록 **단일 툴 + 의도 파라미터** 구조를 사용합니다.

*   **Function Name:** `consult_archive` (또는 `recall_movie_memory`)
*   **Description:** "영화/음악에 대한 정보를 찾거나, 사용자의 기분/상황에 맞는 작품을 연상(Association)할 때 사용합니다. 사실 확인뿐만 아니라 위로, 공감, 추천이 필요할 때 적극적으로 사용하세요."
*   **Parameters:**
    *   `query` (string): 검색할 키워드 또는 문장 (LLM이 대화 맥락을 '검색어'로 변환. 예: '비 오는 날의 우울함', '헤어질 결심 해석').
    *   `intent` (string): `fact` | `vibe` | `critique`.

### 4.2. 검색 로직 (Backend Router)
LLM이 넘겨준 `intent`에 따라 필터링 전략을 달리합니다. 심층 정보가 검색되면 자동으로 기본 정보를 셋트로 묶어서 반환합니다.

*   **If `intent == fact`:** (정확성 중심)
    *   `metadata={"category": "basic_info"}` 필터 적용. (줄거리/감독/출연진 등 사실 정보만 검색)
*   **If `intent == vibe`:** (발견 중심)
    *   **No Filter (전체 검색)**. 기본정보, 평론, 인터뷰를 통틀어 query의 감성/분위기와 가장 유사한 텍스트 추출.
*   **If `intent == critique`:** (깊이 중심)
    *   `metadata={"category": ["critique", "interview"]}` 필터 적용. (평론, 인터뷰 내용만 검색)

*   **If `intent == vibe` OR `critique`:**
    1.  **1차 검색:** `critique`, `interview` 등 심층 문서 위주로(혹은 전체) 검색.
    2.  **ID 추출:** 검색된 문서들의 메타데이터에서 `movie_id` 추출.
    3.  **2차 보강 (Enrichment):** 해당 `movie_id`를 가진 **[기본 정보(Basic Info)] 문서를 DB에서 강제로 가져와서 결합.**
    4.  **최종 반환:** `[기본 정보] + [검색된 평론/인터뷰]` 형태의 풍부한 텍스트를 LLM에게 전달.
    *   *이유: 평론만 읽고 영화의 기본 스펙(감독, 출연진, 줄거리)을 모르는 환각(Hallucination) 방지.*

### 4.3. 컨텍스트 처리 및 메모리 관리 (Context Management)
대화의 맥락을 깨끗하게 유지하기 위해 '휘발성 기억' 전략을 사용합니다.

1.  **Generate Step (답변 생성 시):**
    *   **Full Context 제공:** `[User Input]` -> `[Assistant Tool Call]` -> `[Tool Output (검색결과)]` 순서의 전체 대화 로그를 LLM에게 전달.
    *   **System Prompt 추가:** (수정 필요) "Tool Output은 검색 결과가 아니라 네가 떠올린 **내면의 기억**이다. 인용하듯 말하지 말고 자연스럽게 회상하며 말해라." + "항상 사용할 필요는 없음. 맥락에 맞게 자연스럽게 사용하거나 혹은 사용하지 않는 것이 자연스러울 때도 있다."

2.  **Save Step (대화 저장 시):**
    *   **Pruning (가지치기):** 답변 생성이 끝나면, **[Assistant Tool Call]과 [Tool Output] 메시지는 삭제**하고 `[User Input]`과 `[Final Answer]`만 DB/메모리에 저장.

---

## 5. 프롬프트 엔지니어링 (Persona)

### 5.1. System Prompt 핵심 지침
검색된 RAG 데이터를 '남의 말'이 아닌 '내 생각'처럼 말하게 하는 것이 핵심입니다.

> **Role Definition:**
> 당신은 영화와 음악에 해박한 로봇입니다. 깊이 있는 지식을 가지고 있지만, 잘난 체하기보다는 친구처럼 대화합니다.
>
> **RAG Data Handling:**
> - `consult_archive` 툴을 통해 제공된 [Context]는 당신의 **'내면의 지식'**이자 **'기억'**입니다.
> - 데이터를 인용할 때 "데이터베이스에 따르면~" 또는 "검색 결과~"라고 말하지 마세요.
> - 평론이나 인터뷰 내용을 **당신이 직접 느끼고 생각한 감상인 것처럼 내면화(Internalize)**하여 자연스럽게 말하세요.
> - 예: "평론가가 파도 같대요" (X) -> "그 영화, 감정이 파도처럼 밀려오지 않나요?" (O)

---

## 6. 개발 로드맵 (Action Plan)

1.  **Step 1 (DB 구축):**
    *   ChromaDB 설치.
    *   샘플 영화 10편 선정 -> 기본정보/평론/인터뷰 텍스트 수집.
    *   `RecursiveCharacterTextSplitter`로 분할 후 메타데이터(`category`)와 함께 저장.
2.  **Step 2 (LangChain RAG):**
    *   `consult_archive` 툴 구현 (LangChain Retriever 연결).
    *   `intent`에 따른 필터링 로직 테스트.
3.  **Step 3 (Prompt Tuning):**
    *   "우울해"라고 말했을 때 로봇이 알아서 `vibe` 인텐트로 검색하고, 위로하는 영화를 추천하는지 테스트.
    *   말투가 너무 기계적이지 않은지 확인 및 수정.