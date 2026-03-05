from celery import shared_task
from .models import DocuProcess
from .agents.supervisor.graph import SupervisorAgent 

@shared_task
def run_agentic_pipeline_task(project_id, reference_file_urls):
    """
    Phase 1: Agentic Ingestion Pipeline
    """
    docu_process = DocuProcess.objects.get(project_id=project_id)
    docu_process.status = DocuProcess.StatusChoices.PROCESSING
    docu_process.save(update_fields=['status'])
    
    try:
        # 1. Initialize the Supervisor Agent
        supervisor_agent = SupervisorAgent()
        
        # 2. Run the graph! 
        final_state = supervisor_agent.run(project_id=project_id, file_urls=reference_file_urls)
        
        print(f"Graph finished using strategy: {final_state.get('chosen_strategy')}")
        
        # 3. Mark as completed
        docu_process.status = DocuProcess.StatusChoices.COMPLETED
        docu_process.save(update_fields=['status'])
        
    except Exception as e:
        docu_process.status = DocuProcess.StatusChoices.FAILED
        docu_process.save(update_fields=['status'])
        raise e