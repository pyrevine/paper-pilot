## Alembic

- Alembic = DB 스키마의 깃
    - models.py: 파이썬으로 작성된 디비 설계도
    - versions/ 작업 지시서들. 변경 이력
    - upgrade head: DB에 변경 내역을 적용하기

alembic revision --autogenerate -m "create papers"
alembic upgrade head
