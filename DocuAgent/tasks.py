# tasks.py — error handling lives here
from celery import shared_task
from .models import DocuProcess
from DocuAgent.agents.orchestrator.graph import build_docu_pipeline_orchestrator
from DocuGyan.celery import stop_task


@shared_task(bind=True, name="run_agentic_pipeline_task")
def run_agentic_pipeline_task(self, project_id, user_uuid):
    """Agentic Ingestion Pipeline"""
    try:
        result = build_docu_pipeline_orchestrator(project_id, user_uuid)
        return result
    except Exception as e:
        base_instance = DocuProcess.objects.filter(project_id=project_id, user_uuid=user_uuid).first()
        base_instance.status = getattr(DocuProcess.StatusChoices, 'FAILED', 'FAILED')
        base_instance.save(update_fields=['status'])
        stop_task(self.request.id)
        raise e