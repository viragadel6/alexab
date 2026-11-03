CREATE TABLE public.chat_browser_sessions (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), job_id uuid NOT NULL, browser_id text NOT NULL UNIQUE, cdp_url text NOT NULL, live_url text, control_mode text NOT NULL DEFAULT 'agent', status text NOT NULL DEFAULT 'active', created_at timestamptz NOT NULL DEFAULT now());
GRANT ALL ON public.chat_browser_sessions TO service_role;
ALTER TABLE public.chat_browser_sessions ENABLE ROW LEVEL SECURITY;
CREATE INDEX chat_browser_sessions_job_idx ON public.chat_browser_sessions(job_id);