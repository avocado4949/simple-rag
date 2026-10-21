# simple-rag

Python 표준 라이브러리와 Ollama로 RAG(Retrieval-Augmented Generation) 기본 원리를 직접 구현하는 최소 실습 프로젝트입니다. LangChain이나 벡터 DB는 아직 사용하지 않습니다.

## 처리 과정

~~~text
Markdown / Text 파일
  -> 문자 단위 청킹 (overlap)
  -> Ollama 임베딩 (bge-m3)
  -> 로컬 JSON 저장 (.local/index.json)

사용자 질문 -> 같은 임베딩 모델 -> 코사인 유사도 Top-K 검색
            -> 검색 문맥 + 질문 -> Ollama LLM (qwen3:4b)
            -> 한국어 답변 + 참조 문서
~~~

## 준비

- Python 3.11 이상
- 실행 중인 Ollama (기본 주소: http://localhost:11434)
- 모델 다운로드:

~~~bash
ollama pull bge-m3
ollama pull qwen3:4b
~~~

별도 Python 패키지 설치는 필요 없습니다.

## 실행

저장소 루트에서:

~~~bash
# 1. 샘플 문서 색인
python3 rag.py index

# 2. 검색 + 답변
python3 rag.py ask --question "RAG는 어떤 과정으로 동작하나요?"

# 3. 직접 준비한 UTF-8 .md/.txt 파일 색인
python3 rag.py index --docs /path/to/my-documents
python3 rag.py ask --question "문서의 주요 내용은?"

# 4. Ollama 없이 단위 테스트
python3 -m unittest discover -s tests -v
~~~

자세한 인자: python3 rag.py index --help / python3 rag.py ask --help

- 기본 청킹: 600자, 100자 겹침 (토큰 수 아님)
- 기본 검색: 코사인 유사도 기준 Top 3
- 모델 변경: 색인 시 --embed-model, 질문 시 --chat-model
- Ollama 서버 변경: 환경변수 OLLAMA_URL 또는 하위 명령 앞에 --ollama-url
- 색인 경로: 색인 시 --output, 질문 시 --index

## 파일 구조

~~~text
simple-rag/
├── rag.py                 # 로딩/청킹/임베딩/검색/생성
├── documents/
│   └── rag-basics.md      # 샘플 문서
├── tests/
│   └── test_rag.py        # Ollama 없는 단위 테스트
├── pyproject.toml         # Python 버전 및 프로젝트 메타데이터
└── .gitignore             # .local/ 비공개 색인 제외
~~~

## 구현상의 한계

1. PDF, DOCX 등은 아직 미지원이며 UTF-8 Markdown/Text만 지원합니다.
2. 문자 기준 청킹이라 문장 경계와 토큰 길이를 고려하지 않습니다.
3. 인덱스를 JSON에 저장하고 메모리에서 전체 검색합니다. 대량 문서는 Qdrant 등으로 교체해야 합니다.
4. 검색 품질 평가, 임계값, 하이브리드 검색, 증분 색인은 아직 없습니다.
5. LLM의 출처 표시와 사실성이 보장되지는 않습니다. 결과를 검증하세요.
6. 공개 저장소에 개인/민감 문서를 커밋하지 마세요. 생성 인덱스는 .gitignore에서 제외합니다.

## 다음 단계

문서 파서 분리 -> PDF 파싱 -> 증분 색인 -> Qdrant -> 검색 평가 -> API(FastAPI).
