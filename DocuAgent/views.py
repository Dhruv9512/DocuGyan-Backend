import uuid
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import DocuProcess
from .tasks import run_agentic_pipeline_task

class InitDocuProcessView(APIView):
    """Step 1: Creates the project ID so the frontend can upload files."""
    def post(self, request):
        user_uuid = request.data.get('user_uuid')
        project_id = uuid.uuid4().hex[:10]

        # Create a placeholder record
        DocuProcess.objects.create(
            project_id=project_id,
            user_uuid=user_uuid,
            status=DocuProcess.StatusChoices.PENDING, # Or create an "INITIALIZING" status
        )

        return Response({"project_id": project_id}, status=status.HTTP_201_CREATED)


class DocuProcessView(APIView):
    """Step 2: Updates the record with URLs and starts the AI."""
    def post(self, request):
        project_id = request.data.get('project_id')
        reference_urls = request.data.get('reference_urls', [])
        question_urls = request.data.get('question_urls', [])
        text_questions = request.data.get('questions', [])
        user_uuid = request.data.get('user_uuid')

        if not project_id:
            return Response({"error": "project_id is required."}, status=status.HTTP_400_BAD_REQUEST)

        # 1. Fetch the project we created in Step 1
        try:
            docu_process = DocuProcess.objects.get(project_id=project_id, user_uuid=user_uuid)
        except DocuProcess.DoesNotExist:
            return Response({"error": "Project not found."}, status=status.HTTP_404_NOT_FOUND)

        # 2. Update it with the uploaded URLs
        docu_process.reference_urls = reference_urls
        docu_process.question_urls = question_urls
        docu_process.text_questions = text_questions
        docu_process.save()

        # 3. Trigger Celery Task
        task = run_agentic_pipeline_task.delay(project_id=project_id, user_uuid=user_uuid)

        docu_process.task_id = task.id
        docu_process.status = DocuProcess.StatusChoices.PROCESSING
        docu_process.save(update_fields=['task_id', 'status'])

        return Response({
            "message": "Document processing started successfully.",
            "project_id": project_id,
            "task_id": task.id
        }, status=status.HTTP_202_ACCEPTED)