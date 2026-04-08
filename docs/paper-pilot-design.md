# Paper Pilot — System Design

## Overview

arXiv 논문을 대상으로 한 RAG 기반 Research Agent. cs.AI / cs.CL / cs.LG 카테고리 논문을 수집·인덱싱하고, 사용자가 자연어로 논문 검색, 내용 질문, 논문 간 비교를 할 수 있도록 한다.

**목표:**
- 단순 키워드 검색이 아닌 의미 기반 검색 + 재순위화로 retrieval 품질 확보
- LangChain 고수준 추상화 최소화 — 핵심 컴포넌트는 직접 구현하여 설계 의도를 명확히 함
- 실사용 가능한 수준의 UX (스트리밍, 비동기 인제스천)

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Streamlit UI                      │
└────────────────────────┬────────────────────────────┘
                         │ HTTP / SSE
┌────────────────────────▼────────────────────────────┐
│                   FastAPI Backend                    │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │  Agent Layer │  │  RAG Layer   │  │  Ingest   │  │
│  │  (Tools)     │  │  (Retrieval) │  │  Worker   │  │
│  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘  │
└─────────┼─────────────────┼────────────────┼────────┘
          │                 │                │
┌─────────▼─────────────────▼────────────────▼────────┐
│              PostgreSQL + pgvector                   │
│   papers | chunks | embeddings | fts_index           │
└─────────────────────────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │    arXiv API         │
              └─────────────────────┘
```

---

## Tech Stack

| 레이어 | 기술 | 선택 이유 |
|--------|------|-----------|
| 백엔드 | FastAPI | async 지원, SSE 스트리밍 |
| LLM 오케스트레이션 | LangChain (부분적) | Tool 정의, LLM 호출 — 고수준 chain은 사용 안 함 |
| LLM | OpenAI GPT-4o / GPT-4o-mini | 비용/성능 균형 |
| 임베딩 | text-embedding-3-small | 비용 효율적, 충분한 성능 |
| 벡터 DB | pgvector (PostgreSQL) | 벡터 + RDB 통합, 운영 단순화 |
| Reranker | cross-encoder (ms-marco-MiniLM) | 로컬 실행 가능, 추가 비용 없음 |
| 웹 UI | Streamlit | Python only, 데모용 충분 |
| 데이터 수집 | arxiv Python SDK | 크롤링 불필요 |
| 비동기 작업 | asyncio + BackgroundTasks | 인제스천 파이프라인 분리 |

---

## Data Pipeline

### 수집
- arXiv API로 cs.AI / cs.CL / cs.LG 카테고리 논문 수집
- 초기: 최근 2년치 약 3만 건 (abstract + metadata)
- 주기적 업데이트: 매일 신규 논문 수집 (백그라운드 작업)
- 전문(full text)은 PDF 파싱 — 선택적 지원 (PyMuPDF)

### 저장 스키마

```sql
-- 논문 메타데이터
CREATE TABLE papers (
    id          TEXT PRIMARY KEY,   -- arXiv ID
    title       TEXT,
    authors     TEXT[],
    abstract    TEXT,
    categories  TEXT[],
    published   DATE,
    updated     DATE,
    pdf_url     TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 청크 (임베딩 단위)
CREATE TABLE chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id    TEXT REFERENCES papers(id),
    section     TEXT,               -- abstract / introduction / method / ...
    content     TEXT,
    embedding   VECTOR(1536),       -- text-embedding-3-small
    token_count INT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- 벡터 인덱스
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Full-text search 인덱스 (hybrid search용)
CREATE INDEX ON chunks USING gin(to_tsvector('english', content));
```

---

## RAG Design

### 청킹 전략

논문 구조를 인식한 섹션 단위 청킹 (naive fixed-size 청킹 지양):

```
Abstract        → 1 chunk (항상 포함)
Introduction    → 512 token 단위, 50 token overlap
Method/Model    → 512 token 단위, 50 token overlap
Experiments     → 512 token 단위, 50 token overlap
Conclusion      → 1-2 chunks
```

- PDF 파싱 시: PyMuPDF로 섹션 헤더 감지
- Abstract only 모드: 섹션 구분 없이 abstract 1 chunk

### Hybrid Search

시맨틱 검색과 키워드 검색을 병행하여 Reciprocal Rank Fusion(RRF)으로 통합:

```python
# 1. 시맨틱 검색 (pgvector cosine similarity)
semantic_results = await vector_search(query_embedding, top_k=20)

# 2. 키워드 검색 (PostgreSQL FTS)
keyword_results = await fts_search(query, top_k=20)

# 3. RRF 통합
fused_results = reciprocal_rank_fusion(
    [semantic_results, keyword_results], k=60
)[:20]
```

### Reranking

Hybrid Search 결과를 cross-encoder로 재순위화:

```python
# ms-marco-MiniLM-L-6-v2 (로컬 실행)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = reranker.predict([(query, chunk.content) for chunk in fused_results])
reranked = sorted(zip(fused_results, scores), key=lambda x: x[1], reverse=True)
final_context = reranked[:5]
```

### Query Rewriting (선택)

복잡한 질문의 경우 LLM으로 검색 쿼리 최적화:

```
사용자 입력: "트랜스포머 어텐션 메커니즘 계산 복잡도 줄이는 방법"
→ 검색 쿼리: "efficient attention mechanism linear complexity transformer"
```

---

## Agent Design

LangChain의 고수준 Agent 체인 대신, Tool을 직접 정의하고 ReAct 루프를 명시적으로 제어한다.

### Tools

```python
tools = [
    SearchPapersTool,       # 논문 검색 (hybrid + rerank)
    AskPaperTool,           # 특정 논문에 대한 Q&A
    SummarizePaperTool,     # 논문 요약
    ComparePapersTool,      # 논문 간 비교 (Tier 2)
    FindRelatedPapersTool,  # 관련 논문 추천 (Tier 2)
]
```

**SearchPapersTool** (핵심)
- 입력: 자연어 쿼리, 날짜 범위 (선택), 카테고리 필터 (선택)
- 출력: 논문 목록 (title, authors, abstract, score)
- 내부: hybrid search + reranking 파이프라인

**AskPaperTool**
- 입력: arXiv ID, 질문
- 출력: 해당 논문 청크 기반 답변 (출처 chunk 포함)
- 내부: 해당 paper_id 필터링 후 retrieval → generation

**SummarizePaperTool**
- 입력: arXiv ID, 요약 유형 (전체 / 방법론 / 실험 결과)
- 출력: 구조화된 요약 (Problem / Method / Results / Limitation)

### Agent 실행 흐름

```
사용자 입력
    → Intent 분류 (단일 Tool 호출 vs 멀티스텝)
    → Tool 선택 및 실행
    → 결과 통합
    → 스트리밍 응답 생성
```

멀티스텝 예시:
```
"FlashAttention이랑 Mamba 비교해줘"
→ SearchPapersTool("FlashAttention")
→ SearchPapersTool("Mamba SSM")
→ ComparePapersTool(paper_id_1, paper_id_2)
→ 비교 결과 스트리밍 출력
```

---

## API Design

```
POST /api/chat              # Agent 대화 (SSE 스트리밍)
POST /api/ingest            # 논문 수집/인덱싱 트리거
GET  /api/papers            # 논문 목록 조회
GET  /api/papers/{id}       # 논문 상세
GET  /api/papers/{id}/chunks # 논문 청크 목록
GET  /api/health            # 헬스체크
```

### Chat API (SSE)

```python
# Request
{
  "message": "FlashAttention 논문 설명해줘",
  "session_id": "uuid",
  "filters": {
    "date_from": "2023-01-01",
    "categories": ["cs.AI", "cs.CL"]
  }
}

# Response (SSE stream)
data: {"type": "tool_call", "tool": "SearchPapersTool", "input": "..."}
data: {"type": "tool_result", "papers": [...]}
data: {"type": "token", "content": "FlashAttention은 ..."}
data: {"type": "done", "usage": {"prompt_tokens": 1200, "completion_tokens": 350}}
```

---

## Web UI (Streamlit)

**주요 화면:**

1. **Chat 인터페이스** (메인)
   - 스트리밍 응답
   - Tool 실행 과정 표시 (중간 단계 투명하게)
   - 인용된 논문 카드 (제목 / 저자 / arXiv 링크)

2. **논문 탐색** (사이드바)
   - 카테고리 필터
   - 날짜 범위 선택
   - 최근 인덱싱된 논문 목록

3. **인덱싱 현황** (관리)
   - 총 논문 수 / 청크 수
   - 마지막 업데이트 시각

---

## Evaluation

Retrieval 품질을 측정하는 최소한의 평가 파이프라인:

| 지표 | 설명 |
|------|------|
| Hit@k | 정답 논문이 상위 k개 안에 있는 비율 |
| MRR | Mean Reciprocal Rank |
| Latency | 검색 + reranking 응답 시간 (p50, p95) |

- 평가 데이터셋: arXiv 논문 제목 → 해당 논문 검색되는지 검증 (자동 구성 가능)
- 결과는 README에 테이블로 기록 (Hybrid vs Semantic-only 비교)

---

## Development Milestones

### Week 1-2: 기반 구축
- [ ] 프로젝트 세팅 (FastAPI, PostgreSQL + pgvector, Streamlit)
- [ ] arXiv 데이터 수집 파이프라인
- [ ] 청킹 + 임베딩 + pgvector 저장
- [ ] 기본 시맨틱 검색 동작 확인

### Week 3-4: RAG 고도화
- [ ] Hybrid search (FTS + vector, RRF)
- [ ] Cross-encoder reranking
- [ ] AskPaperTool, SummarizePaperTool 구현
- [ ] SSE 스트리밍 응답

### Week 5-6: Agent + UI 완성
- [ ] ReAct Agent 루프 구현
- [ ] ComparePapersTool (Tier 2)
- [ ] Streamlit UI 완성 (Tool 실행 과정 시각화 포함)
- [ ] Evaluation 파이프라인 + README에 결과 기록
- [ ] Docker Compose로 로컬 실행 환경 통일

---

## 디렉토리 구조

```
paper-pilot/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI 라우터
│   │   ├── agent/        # Tool 정의, Agent 루프
│   │   ├── rag/          # retrieval, reranking, hybrid search
│   │   ├── ingest/       # arXiv 수집, 청킹, 임베딩
│   │   └── db/           # PostgreSQL 연결, 스키마
│   ├── tests/
│   └── pyproject.toml
├── frontend/
│   └── app.py            # Streamlit UI
├── scripts/
│   └── ingest_initial.py # 초기 데이터 수집 스크립트
├── docker-compose.yml
└── README.md
```
