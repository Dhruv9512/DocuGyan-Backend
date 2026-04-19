# tasks.py — error handling lives here
from celery import shared_task
from .models import DocuProcess
from DocuAgent.agents.orchestrator.graph import build_docu_pipeline_orchestrator
from DocuAgent.utils.utility import delete_blobs_in_collection, get_collection_name
from DocuGyan.celery import stop_task
import logging
logger = logging.getLogger(__name__)

@shared_task(bind=True, queue="DocuAgent_tasks" ,time_limit=1800, soft_time_limit=1750, name="run_agentic_pipeline_task")
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
    
@shared_task(bind=True, queue="DocuGyan", name="delete_blob_collection_task", max_retries=3) 
def process_project_deletion(self, project_id, user_uuid):
    logger.info(f"Starting background deletion for project: {project_id} (User: {user_uuid})")
    
    try:
        # 1. Delete the actual files from Vercel Blob Storage
        blob_collection = get_collection_name(project_id)
        delete_blobs_in_collection(blob_collection)
        logger.info(f"Successfully deleted Vercel blobs for project: {project_id}")
        
        # 2. Hard Delete: Permanently remove the record from the database
        deleted_count, _ = DocuProcess.objects.filter(
            project_id=project_id, 
            user_uuid=user_uuid
        ).delete()
        
        if deleted_count > 0:
            logger.info(f"Successfully hard-deleted project {project_id} from database.")
        else:
            logger.warning(f"Project {project_id} was missing from DB during hard delete.")

    except Exception as e:
        logger.error(f"Error during deletion of project {project_id}: {str(e)}. Retrying...")
        try:
            # If Vercel API throws an error, try again in 60s
            self.retry(countdown=60, exc=e)
            
        except self.MaxRetriesExceededError:
            # Graceful Failure: Update status to FAILED and log a critical error
            logger.critical(f"PERMANENT FAILURE: Could not delete project {project_id} after 3 retries.")
            
            DocuProcess.objects.filter(
                project_id=project_id, 
                user_uuid=user_uuid
            ).update(
                status=DocuProcess.StatusChoices.DELETED
            )


