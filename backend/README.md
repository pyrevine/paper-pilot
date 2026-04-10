## 로컬 세팅

1. `docker compose up -d`
2. pgvector 활성화: `docker compose exec db psql -U paperpilot -c "CREATE EXTENSION IF NOT EXISTS vector;"`
3. migration: `alembic upgrade head`
4. 서버 실행: `uvicorn backend.app.main:app --reload`