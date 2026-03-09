# tasks.py — error handling lives here
from celery import shared_task
from .models import DocuProcess
from .agents import build_supervisor_agent
from DocuGyan.celery import stop_task


@shared_task(bind=True, name="run_agentic_pipeline_task")
def run_agentic_pipeline_task(self, project_id, user_uuid):
    """Agentic Ingestion Pipeline"""
    docu_process = DocuProcess.objects.get(project_id=project_id)
    docu_process.status = DocuProcess.StatusChoices.PROCESSING
    docu_process.task_id = self.request.id
    docu_process.save(update_fields=['status', 'task_id'])

    try:
        result = build_supervisor_agent(project_id, user_uuid)

        docu_process.status = DocuProcess.StatusChoices.COMPLETED
        docu_process.metadata = {
            "rag_strategy": result.get("rag_strategy"),
            "refined_questions": result.get("refined_questions"),
        }
        docu_process.save(update_fields=['status', 'metadata'])

    except Exception as e:
        docu_process.status = DocuProcess.StatusChoices.FAILED
        docu_process.error_message = str(e)
        docu_process.save(update_fields=['status', 'error_message'])
        stop_task(self.request.id)
        raise e