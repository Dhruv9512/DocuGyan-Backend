import uuid
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny

from DocuAgent.utils.utility import get_collection_name
from .models import DocuProcess
from .tasks import run_agentic_pipeline_task

class InitDocuProcessView(APIView):
    permission_classes = [AllowAny]  
    """Step 1: Creates the project ID so the frontend can upload files."""
    def post(self, request):
        user_uuid = request.data.get('user_uuid')
        title = request.data.get('text', '')
        description = request.data.get('description', '')
        project_id = uuid.uuid4()

        if not user_uuid:
            return Response({"error": "user_uuid is required."}, status=status.HTTP_400_BAD_REQUEST)
        
        if not title:
            title = f"Untitled_DocuProcess"
        
        if not description:
            description = f"No description provided."
            
        # Create a placeholder record
        DocuProcess.objects.create(
            project_id=project_id,
            user_uuid=user_uuid,
            title=title,
            description=description,
            status=DocuProcess.StatusChoices.PENDING, 
        )

        blob_collection = get_collection_name(project_id)

        return Response({"project_id": project_id,"blob_collection": blob_collection}, status=status.HTTP_201_CREATED)


class DocuProcessView(APIView):
    permission_classes = [AllowAny]
    """Step 2: Updates the record with URLs and starts the AI."""
    def post(self, request):
        project_id = request.data.get('project_id')
        reference_urls = request.data.get('reference_urls', [])
        question_urls = request.data.get('question_urls', [])
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
    
# Show the DocuProcess Data for a given project_id
class DocuProcessDataView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        project_id = request.query_params.get('project_id')
        user_uuid = request.query_params.get('user_uuid')

        if not project_id:
            return Response({"error": "project_id is required."}, status=status.HTTP_400_BAD_REQUEST)
        
        if not user_uuid:
            return Response({"error": "user_uuid is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            docu_process = DocuProcess.objects.get(project_id=project_id, user_uuid=user_uuid)
            data = {
                "project_id": str(docu_process.project_id),
                "user_uuid": str(docu_process.user_uuid),
                "title": docu_process.title,
                "description": docu_process.description,
                "reference_urls": docu_process.reference_urls,
                "question_urls": docu_process.question_urls,
                "result_urls": docu_process.result_urls,
                "created_at": docu_process.created_at,
                "status": docu_process.status,
            }
            return Response(data, status=status.HTTP_200_OK)
        except DocuProcess.DoesNotExist:
            return Response({"error": "Project not found."}, status=status.HTTP_404_NOT_FOUND)


# List of all DocuProcesses for a given user_uuid
class DocuProcessListView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        user_uuid = request.query_params.get('user_uuid')

        if not user_uuid:
            return Response({"error": "user_uuid is required."}, status=status.HTTP_400_BAD_REQUEST)

        docu_processes = DocuProcess.objects.filter(user_uuid=user_uuid).order_by('-created_at')
        data = []
        for process in docu_processes:
            data.append({
                "project_id": str(process.project_id),
                "user_uuid": str(process.user_uuid),
                "title": process.title,
                "description": process.description,
                "reference_urls": process.reference_urls,
                "question_urls": process.question_urls,
                "result_urls": process.result_urls,
                "created_at": process.created_at,
                "status": process.status,
            })
        return Response(data, status=status.HTTP_200_OK)