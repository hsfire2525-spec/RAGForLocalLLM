"""社内共有のWeb UI と API。

**パイプラインはそのまま使う。** ``QueryPipeline.run(question) → QueryState``
に UI が必要なもの（回答・引用・使用コンテキスト・段別 trace）が全部入って
いるため、コア側に変更は要らない。ここはその薄い皮である。

設計上の判断が3つある。

1. **インデックスは起動時に読み、リクエストごとに作らない。** 埋め込み器の
   常駐は 1.5GB あり（design.md §9 Phase 2）、都度構築するとメモリも
   レイテンシも破綻する
2. **パイプラインの実行を直列化する。** 埋め込みモデルはスレッド安全とは
   限らず、LM Studio 側も同時要求で不安定になりうる。同時アクセスは
   待たせる。**推測で並列化して壊れるより、遅くても正しく動くほうがよい**
3. **根拠の表示を必須にする。** 機密資料では「回答が正しいか」を利用者が
   検証できないと使えない。引用と原文をUIから常に見せる
"""

# **このモジュールでは `from __future__ import annotations` を使わない。**
# 注釈が文字列化されると、FastAPI が create_app のローカルスコープで定義した
# 型（AskRequest / Request）を解決できず、ボディがクエリパラメータとして
# 扱われて 422 になる。fastapi を任意依存に保つため型のimportは関数内に置いており、
# その両立にはここで注釈を即時評価する必要がある。
import threading
import time
from pathlib import Path
from typing import Any

from ragforlocalllm.core.cache import Cache
from ragforlocalllm.core.config import ConfigError, ExperimentConfig, load_config
from ragforlocalllm.core.indexing import build_index
from ragforlocalllm.core.pipeline import QueryPipeline
from ragforlocalllm.server.security import AuditLog, SecurityPolicy

CONFIG_DIR = Path("configs")
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_QUESTION_CHARS = 2000


class PipelineRegistry:
    """設定名ごとにパイプラインを保持する。

    初回アクセス時に構築し、以降は使い回す。**構築は排他する**。
    同じ設定に同時アクセスが来たときに二重で埋め込みモデルを読むと、
    メモリが2倍になる。
    """

    def __init__(self, config_dir: Path = CONFIG_DIR) -> None:
        self.config_dir = config_dir
        self._pipelines: dict[str, tuple[ExperimentConfig, QueryPipeline]] = {}
        self._build_lock = threading.Lock()
        self._run_lock = threading.Lock()

    def available(self) -> list[str]:
        return sorted(p.stem for p in self.config_dir.glob("*.yaml"))

    def get(self, name: str) -> tuple[ExperimentConfig, QueryPipeline]:
        if name in self._pipelines:
            return self._pipelines[name]
        with self._build_lock:
            if name not in self._pipelines:  # 待っている間に他が構築済みかもしれない
                config = load_config(self.config_dir / f"{name}.yaml", search_dir=self.config_dir)
                with Cache() as cache:
                    built = build_index(config, cache=cache)
                pipeline = QueryPipeline.from_config(
                    config, embedder=built.embedder, index=built.index
                )
                self._pipelines[name] = (config, pipeline)
        return self._pipelines[name]

    def warmup(self, names: list[str]) -> list[tuple[str, str | None]]:
        """設定を事前に読み込む。

        **初回アクセスの待ち時間を利用者に押し付けない。** 埋め込みモデルは
        遅延読み込みのため、素のままだと最初の質問だけ 17 秒かかる（実測）。
        共有サービスでは「壊れている」と受け取られる。

        1設定あたり約1.5GBを常駐させるので、**必要なものだけ**温める。
        """
        outcomes: list[tuple[str, str | None]] = []
        for name in names:
            try:
                _, pipeline = self.get(name)
                # 埋め込み器はモデルを遅延読み込みする。実際に引かせる。
                pipeline.retriever.retrieve(["ウォームアップ"], 1)
                outcomes.append((name, None))
            except Exception as exc:
                outcomes.append((name, f"{type(exc).__name__}: {exc}"))
        return outcomes

    def ask(self, name: str, question: str) -> dict[str, Any]:
        config, pipeline = self.get(name)
        # 直列化。埋め込みモデルと LM Studio の同時要求を避ける。
        with self._run_lock:
            state = pipeline.run(question)

        answer = state.answer
        return {
            "config": config.name,
            "question": question,
            "answer": None if answer is None else answer.text,
            "abstained": bool(answer and answer.abstained),
            "citations": [] if answer is None else list(answer.citations),
            "contexts": [
                {
                    "chunk_id": item.chunk.chunk_id,
                    "score": round(item.score, 4),
                    "page": item.chunk.page,
                    "section_path": item.chunk.section_path,
                    "text": item.chunk.text,
                    "cited": bool(answer and item.chunk.chunk_id in answer.citations),
                    "in_prompt": bool(
                        state.prompt and item.chunk.chunk_id in state.prompt.context_chunk_ids
                    ),
                }
                for item in state.contexts
            ],
            "trace": [
                {
                    "stage": t.stage,
                    "impl": t.impl,
                    "duration_ms": round(t.duration_ms, 1),
                    "info": t.info,
                }
                for t in state.trace
            ],
            "latency_ms": round(state.total_duration_ms, 1),
            "n_dropped": 0 if state.prompt is None else state.prompt.n_dropped,
        }


def create_app(policy: SecurityPolicy, *, config_dir: Path = CONFIG_DIR) -> Any:
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel, Field
    from starlette.requests import Request

    registry = PipelineRegistry(config_dir)
    audit = AuditLog(policy.audit_log)
    app = FastAPI(title="RAGForLocalLLM", docs_url=None, redoc_url=None)
    app.state.registry = registry

    class AskRequest(BaseModel):
        question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
        # 設定名はファイル名になる。`../` を弾いて経路の遡上を防ぐ
        # （実在チェックもするが、入力側でも閉じておく）。
        config: str = Field(default="baseline", pattern=r"^[A-Za-z0-9_.-]+$")

    def require_token(authorization: str | None = Header(default=None)) -> None:
        presented = None
        if authorization and authorization.lower().startswith("bearer "):
            presented = authorization[7:].strip()
        if not policy.check(presented):
            raise HTTPException(status_code=401, detail="トークンが正しくありません")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        # 認証不要。疎通確認に使うため、機密は一切返さない。
        return {"status": "ok", "auth_required": policy.requires_token}

    @app.get("/api/configs", dependencies=[Depends(require_token)])
    def configs() -> dict[str, Any]:
        return {"configs": registry.available()}

    @app.post("/api/ask", dependencies=[Depends(require_token)])
    def ask(payload: AskRequest, request: Request) -> Any:
        client = request.client.host if request.client else "unknown"
        if payload.config not in registry.available():
            raise HTTPException(status_code=404, detail=f"設定がありません: {payload.config}")

        started = time.perf_counter()
        try:
            result = registry.ask(payload.config, payload.question.strip())
        except ConfigError as exc:
            audit.record(
                client=client,
                question=payload.question,
                config=payload.config,
                chunk_ids=[],
                abstained=False,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=str(exc),
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            audit.record(
                client=client,
                question=payload.question,
                config=payload.config,
                chunk_ids=[],
                abstained=False,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            )
            # 内部の詳細（パスなど）を利用者に返さない
            return JSONResponse(status_code=500, content={"detail": "回答の生成に失敗しました"})

        audit.record(
            client=client,
            question=payload.question,
            config=payload.config,
            chunk_ids=[c["chunk_id"] for c in result["contexts"] if c["in_prompt"]],
            abstained=result["abstained"],
            latency_ms=result["latency_ms"],
        )
        return result

    @app.get("/")
    def index() -> Any:
        return FileResponse(STATIC_DIR / "index.html")

    return app
