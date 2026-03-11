# tasks.py — error handling lives here
from celery import shared_task
from .models import DocuProcess
from .agents import build_supervisor_agent
from DocuGyan.celery import stop_task


@shared_task(bind=True, name="run_agentic_pipeline_task")
def run_agentic_pipeline_task(self, project_id, user_uuid):
    """Agentic Ingestion Pipeline"""
    try:
        result = build_supervisor_agent(project_id, user_uuid)

        return result
    except Exception as e:
        stop_task(self.request.id)
        raise e