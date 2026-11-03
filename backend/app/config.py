from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_origins: str = ""
    app_origin: str = ""

    session_secret: str = ""
    admin_token: str = ""
    worker_token: str = ""

    supabase_url: str = ""
    supabase_publishable_key: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "SUPABASE_SERVICE_ROLE_KEY",
            "SUPABASE_SECRET_KEY",
        ),
    )

    model_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "MODEL_API_KEY",
            "META_API_KEY",
        ),
    )
    instavm_api_key: str = ""
    browser_use_api_key: str = ""
    gemini_api_key: str = ""
    flymyai_api_key: str = ""
    flymyai_base_url: str = "https://api.flymy.ai"
    flymyai_image_model: str = "flymyai/nano-banana-2_1"

    model_name: str = Field(
        default="muse-spark-1.3-contributor",
        validation_alias=AliasChoices("MUSE_SPARK_MODEL_NAME"),
    )
    model_fallback_names: str = ""
    model_retry_max: int = 4
    model_stream_retry_max: int = 2
    model_api_url: str = "https://api.meta.ai/v1/responses"
    instavm_url: str = "https://api.instavm.io"
    published_origin: str = ""

    supervisor_enabled: bool = True
    supervisor_interval_ms: int = 15_000
    supervisor_leader_lease_ms: int = 45_000
    supervisor_stale_queued_ms: int = 45_000
    supervisor_stuck_progress_ms: int = 900_000
    supervisor_orphan_browser_ms: int = 240_000
    supervisor_probe_timeout_ms: int = 8_000
    supervisor_backoff_max_ms: int = 300_000
    supervisor_batch_size: int = 25
    supervisor_recovery_log_limit: int = 400
    supervisor_breaker_threshold: int = 4
    supervisor_breaker_cooldown_ms: int = 120_000

    chat_job_max_attempts: int = 4
    chat_job_retry_base_ms: int = 5_000
    chat_job_retry_max_ms: int = 300_000
    chat_job_sink_retry_max: int = 6
    chat_job_sink_buffer_max: int = 400
    chat_job_event_buffer_max: int = 200

    agent_checkpoint_enabled: bool = True
    agent_checkpoint_interval_steps: int = 4
    agent_checkpoint_max_bytes: int = 262_144
    agent_step_retry_max: int = 3
    agent_loop_repeat_limit: int = 3
    agent_sandbox_recreate_max: int = 2
    agent_sandbox_probe_interval_steps: int = 20
    agent_soft_deadline_margin_ms: int = 240_000
    agent_uncertainty_limit: int = 3

    chat_deployment_max_request_ms: str = ""
    chat_sandbox_allowed_domains: str = ""
    chat_browser_allowed_hosts: str = ""
    chat_nixpkgs_rev: str = ""
    chat_nix_version: str = ""
    chat_nix_installer_sha256: str = ""

    static_dir: str = "../public"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
