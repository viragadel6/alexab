CREATE EXTENSION IF NOT EXISTS pg_net;

CREATE OR REPLACE FUNCTION public.kick_chat_worker(_base_url text, _job_id uuid)
RETURNS bigint
LANGUAGE sql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
  SELECT net.http_post(
    url := _base_url || '/api/public/chat-worker',
    body := jsonb_build_object('job_id', _job_id),
    headers := '{"Content-Type":"application/json"}'::jsonb,
    timeout_milliseconds := 21900000
  );
$$;

REVOKE ALL ON FUNCTION public.kick_chat_worker(text, uuid) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.kick_chat_worker(text, uuid) TO service_role;