ALTER TABLE public.chat_browser_sessions ADD COLUMN agent_busy boolean NOT NULL DEFAULT false;
CREATE FUNCTION public.claim_chat_browser_step(_job_id uuid) RETURNS boolean LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
BEGIN
PERFORM id FROM public.chat_browser_sessions WHERE job_id = _job_id AND status = 'active' FOR UPDATE;
IF EXISTS (SELECT 1 FROM public.chat_browser_sessions WHERE job_id = _job_id AND status = 'active' AND control_mode <> 'agent') THEN RETURN false; END IF;
UPDATE public.chat_browser_sessions SET agent_busy = true WHERE job_id = _job_id AND status = 'active';
RETURN true;
END;
$$;
REVOKE ALL ON FUNCTION public.claim_chat_browser_step(uuid) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.claim_chat_browser_step(uuid) TO service_role;