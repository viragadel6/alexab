ALTER TABLE public.chat_jobs
  ADD COLUMN IF NOT EXISTS attempts integer NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS next_attempt_at timestamptz,
  ADD COLUMN IF NOT EXISTS last_error text,
  ADD COLUMN IF NOT EXISTS progress_at timestamptz,
  ADD COLUMN IF NOT EXISTS checkpoint jsonb;

ALTER TABLE public.chat_conversations
  ADD COLUMN IF NOT EXISTS client_id text;

CREATE INDEX IF NOT EXISTS chat_jobs_recovery_idx
  ON public.chat_jobs (status, next_attempt_at, created_at)
  WHERE status IN ('queued', 'running');

CREATE INDEX IF NOT EXISTS chat_jobs_lease_idx
  ON public.chat_jobs (lease_expires_at)
  WHERE status = 'running';

CREATE INDEX IF NOT EXISTS chat_conversations_client_idx
  ON public.chat_conversations (client_id);

CREATE TABLE IF NOT EXISTS public.agent_supervisor_leases (
  name text PRIMARY KEY,
  owner_id text NOT NULL,
  acquired_at timestamptz NOT NULL DEFAULT now(),
  renewed_at timestamptz NOT NULL DEFAULT now(),
  expires_at timestamptz NOT NULL,
  sweep_count bigint NOT NULL DEFAULT 0,
  healed_count bigint NOT NULL DEFAULT 0,
  state jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS public.agent_recovery_log (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  job_id uuid,
  kind text NOT NULL,
  owner_id text NOT NULL DEFAULT '',
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS agent_recovery_log_created_idx
  ON public.agent_recovery_log (created_at DESC);

CREATE INDEX IF NOT EXISTS agent_recovery_log_job_idx
  ON public.agent_recovery_log (job_id, id DESC);

ALTER TABLE public.agent_supervisor_leases ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.agent_recovery_log ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON TABLE public.agent_supervisor_leases FROM PUBLIC, anon, authenticated;
REVOKE ALL ON TABLE public.agent_recovery_log FROM PUBLIC, anon, authenticated;

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.agent_supervisor_leases TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.agent_recovery_log TO service_role;

CREATE OR REPLACE FUNCTION public.claim_supervisor_lease(
  _name text,
  _owner_id text,
  _lease_ms integer,
  _state jsonb
)
RETURNS TABLE(owner_id text, acquired boolean)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
DECLARE
  current_row public.agent_supervisor_leases%ROWTYPE;
  lease_ms integer := greatest(coalesce(_lease_ms, 1000), 1000);
  now_ts timestamptz;
  expires_ts timestamptz;
BEGIN
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended(coalesce(_name, ''), 0)
  );

  now_ts := pg_catalog.clock_timestamp();
  expires_ts := now_ts
    + pg_catalog.make_interval(secs => lease_ms::double precision / 1000.0);

  SELECT leases.*
  INTO current_row
  FROM public.agent_supervisor_leases AS leases
  WHERE leases.name = _name
  FOR UPDATE;

  IF NOT FOUND THEN
    INSERT INTO public.agent_supervisor_leases (
      name,
      owner_id,
      acquired_at,
      renewed_at,
      expires_at,
      state
    )
    VALUES (
      _name,
      _owner_id,
      now_ts,
      now_ts,
      expires_ts,
      coalesce(_state, '{}'::jsonb)
    )
    ON CONFLICT (name) DO NOTHING;

    SELECT leases.*
    INTO current_row
    FROM public.agent_supervisor_leases AS leases
    WHERE leases.name = _name
    FOR UPDATE;

    IF NOT FOUND THEN
      RETURN QUERY SELECT ''::text AS owner_id, false AS acquired;
      RETURN;
    END IF;
  END IF;

  IF current_row.owner_id = _owner_id OR current_row.expires_at <= now_ts THEN
    UPDATE public.agent_supervisor_leases AS leases
    SET owner_id = _owner_id,
        acquired_at = CASE
          WHEN leases.owner_id = _owner_id AND leases.expires_at > now_ts
            THEN leases.acquired_at
          ELSE now_ts
        END,
        renewed_at = now_ts,
        expires_at = expires_ts,
        state = CASE
          WHEN leases.owner_id = _owner_id THEN coalesce(_state, leases.state)
          ELSE coalesce(_state, '{}'::jsonb)
        END
    WHERE leases.name = _name;

    RETURN QUERY SELECT _owner_id::text AS owner_id, true AS acquired;
    RETURN;
  END IF;

  RETURN QUERY SELECT current_row.owner_id::text AS owner_id, false AS acquired;
END;
$$;

CREATE OR REPLACE FUNCTION public.release_supervisor_lease(_name text, _owner_id text)
RETURNS boolean
LANGUAGE sql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
  WITH released AS (
    UPDATE public.agent_supervisor_leases AS leases
    SET owner_id = '',
        renewed_at = pg_catalog.clock_timestamp(),
        expires_at = pg_catalog.clock_timestamp()
    WHERE leases.name = _name AND leases.owner_id = _owner_id
    RETURNING 1
  )
  SELECT EXISTS (SELECT 1 FROM released);
$$;

CREATE OR REPLACE FUNCTION public.claim_chat_job_retry(
  _job_id uuid,
  _max_attempts integer,
  _retry_base_ms integer,
  _retry_max_ms integer,
  _error text
)
RETURNS TABLE(status text, attempts integer, retry boolean)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
DECLARE
  job_row public.chat_jobs%ROWTYPE;
  base_ms integer := greatest(coalesce(_retry_base_ms, 0), 0);
  max_ms integer := greatest(coalesce(_retry_max_ms, 0), 0);
  attempt_limit integer := greatest(coalesce(_max_attempts, 0), 0);
  next_attempts integer;
  delay_ms numeric;
  now_ts timestamptz;
BEGIN
  now_ts := pg_catalog.clock_timestamp();

  SELECT jobs.*
  INTO job_row
  FROM public.chat_jobs AS jobs
  WHERE jobs.id = _job_id
  FOR UPDATE;

  IF NOT FOUND THEN
    RETURN QUERY SELECT 'missing'::text AS status, 0 AS attempts, false AS retry;
    RETURN;
  END IF;

  IF job_row.status IN ('completed', 'cancelled', 'failed') THEN
    RETURN QUERY
      SELECT job_row.status::text AS status,
             coalesce(job_row.attempts, 0) AS attempts,
             false AS retry;
    RETURN;
  END IF;

  IF job_row.status = 'queued'
     AND job_row.next_attempt_at IS NOT NULL
     AND job_row.next_attempt_at > now_ts THEN
    RETURN QUERY
      SELECT 'queued'::text AS status,
             coalesce(job_row.attempts, 0) AS attempts,
             false AS retry;
    RETURN;
  END IF;

  next_attempts := coalesce(job_row.attempts, 0) + 1;

  IF job_row.stop_requested OR next_attempts >= attempt_limit THEN
    UPDATE public.chat_jobs AS jobs
    SET status = CASE WHEN job_row.stop_requested THEN 'cancelled' ELSE 'failed' END,
        attempts = next_attempts,
        last_error = coalesce(_error, job_row.last_error),
        lease_expires_at = NULL,
        next_attempt_at = NULL,
        finished_at = now_ts,
        updated_at = now_ts
    WHERE jobs.id = _job_id;

    RETURN QUERY
      SELECT (CASE WHEN job_row.stop_requested THEN 'cancelled' ELSE 'failed' END)::text AS status,
             next_attempts AS attempts,
             false AS retry;
    RETURN;
  END IF;

  delay_ms := least(
    greatest(max_ms::numeric, base_ms::numeric),
    base_ms::numeric * power(2::numeric, least(next_attempts - 1, 16))
  );
  delay_ms := greatest(least(delay_ms, 2147483647::numeric), 0::numeric);

  UPDATE public.chat_jobs AS jobs
  SET status = 'queued',
      attempts = next_attempts,
      last_error = coalesce(_error, job_row.last_error),
      lease_expires_at = NULL,
      next_attempt_at = now_ts
        + pg_catalog.make_interval(secs => delay_ms::double precision / 1000.0),
      updated_at = now_ts
  WHERE jobs.id = _job_id;

  RETURN QUERY SELECT 'queued'::text AS status, next_attempts AS attempts, true AS retry;
END;
$$;

CREATE OR REPLACE FUNCTION public.create_chat_job(
  _conversation_id uuid,
  _job_id uuid,
  _client_id text,
  _title text,
  _messages jsonb,
  _history jsonb
)
RETURNS TABLE(conversation_id uuid, job_id uuid, created boolean)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
DECLARE
  existing_owner text;
  conversation_exists boolean := false;
  now_ts timestamptz := pg_catalog.clock_timestamp();
BEGIN
  SELECT conversations.client_id
  INTO existing_owner
  FROM public.chat_conversations AS conversations
  WHERE conversations.id = _conversation_id
  FOR UPDATE;

  IF FOUND THEN
    conversation_exists := true;
  ELSE
    BEGIN
      INSERT INTO public.chat_conversations (
        id,
        title,
        messages,
        client_id,
        created_at,
        updated_at
      )
      VALUES (
        _conversation_id,
        _title,
        coalesce(_messages, '[]'::jsonb),
        _client_id,
        now_ts,
        now_ts
      );
      conversation_exists := false;
    EXCEPTION WHEN unique_violation THEN
      SELECT conversations.client_id
      INTO existing_owner
      FROM public.chat_conversations AS conversations
      WHERE conversations.id = _conversation_id
      FOR UPDATE;

      IF NOT FOUND THEN
        RAISE EXCEPTION 'A beszélgetés nem hozható létre.'
          USING ERRCODE = '40001';
      END IF;

      conversation_exists := true;
    END;
  END IF;

  IF conversation_exists THEN
    IF existing_owner IS NOT NULL AND existing_owner <> _client_id THEN
      RAISE EXCEPTION 'A beszélgetés más ügyfélhez tartozik.'
        USING ERRCODE = '42501';
    END IF;

    UPDATE public.chat_conversations AS conversations
    SET title = CASE
          WHEN conversations.title IS NULL OR conversations.title = '' THEN _title
          ELSE conversations.title
        END,
        messages = coalesce(_messages, conversations.messages),
        client_id = coalesce(conversations.client_id, _client_id),
        updated_at = now_ts
    WHERE conversations.id = _conversation_id;
  END IF;

  INSERT INTO public.chat_jobs (
    id,
    conversation_id,
    history,
    status,
    created_at,
    updated_at
  )
  VALUES (
    _job_id,
    _conversation_id,
    coalesce(_history, '[]'::jsonb),
    'queued',
    now_ts,
    now_ts
  );

  RETURN QUERY
    SELECT _conversation_id AS conversation_id,
           _job_id AS job_id,
           NOT conversation_exists AS created;
END;
$$;

CREATE OR REPLACE FUNCTION public.kick_chat_worker(_base_url text, _job_id uuid)
RETURNS bigint
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $$
DECLARE
  worker_token text := pg_catalog.current_setting('app.chat_worker_token', true);
BEGIN
  IF worker_token IS NULL OR worker_token = '' THEN
    RAISE EXCEPTION 'Az app.chat_worker_token adatbázis-beállítás hiányzik.'
      USING ERRCODE = '55000';
  END IF;

  IF _base_url IS NULL OR _base_url = '' THEN
    RAISE EXCEPTION 'A háttérmunkás alap URL-címe hiányzik.'
      USING ERRCODE = '22023';
  END IF;

  RETURN net.http_post(
    url := _base_url || '/api/public/chat-worker',
    body := pg_catalog.jsonb_build_object('job_id', _job_id),
    headers := pg_catalog.jsonb_build_object(
      'Content-Type', 'application/json',
      'X-Worker-Token', worker_token
    ),
    timeout_milliseconds := 21900000
  );
END;
$$;

REVOKE ALL ON FUNCTION public.claim_supervisor_lease(text, text, integer, jsonb)
  FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.release_supervisor_lease(text, text)
  FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.claim_chat_job_retry(uuid, integer, integer, integer, text)
  FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.create_chat_job(uuid, uuid, text, text, jsonb, jsonb)
  FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.kick_chat_worker(text, uuid)
  FROM PUBLIC, anon, authenticated;

GRANT EXECUTE ON FUNCTION public.claim_supervisor_lease(text, text, integer, jsonb)
  TO service_role;
GRANT EXECUTE ON FUNCTION public.release_supervisor_lease(text, text)
  TO service_role;
GRANT EXECUTE ON FUNCTION public.claim_chat_job_retry(uuid, integer, integer, integer, text)
  TO service_role;
GRANT EXECUTE ON FUNCTION public.create_chat_job(uuid, uuid, text, text, jsonb, jsonb)
  TO service_role;
GRANT EXECUTE ON FUNCTION public.kick_chat_worker(text, uuid)
  TO service_role;
