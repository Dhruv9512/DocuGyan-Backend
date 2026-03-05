import uuid
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .models import DocuProcess
from .tasks import run_agentic_pipeline_task


class DocuProcessView(APIView):
    """
    API endpoint to trigger the DocuPDF processing pipeline that include multiple reference and question URLs.
    """
    def post(self, request):
        # 1. Receive generalized file URLs
        reference_urls = request.data.get('reference_urls', [])
        question_urls = request.data.get('question_urls', [])
        text_questions = request.data.get('questions', [])
        user_uuid = request.data.get('user_uuid')

        # 2. Validation
        if not reference_urls:
            return Response(
                {"error": "At least one 'reference_urls' is required."},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not question_urls and not text_questions:
            return Response(
                {"error": "You must provide either 'question_urls' or 'questions' (text)."},
                status=status.HTTP_400_BAD_REQUEST
            )

        project_id = uuid.uuid4().hex[:10]

        # 3. Create a DocuProcess record
        docu_process = DocuProcess.objects.create(
            project_id=project_id,
            user_uuid=user_uuid,
            status=DocuProcess.StatusChoices.PENDING,
            reference_urls=reference_urls,
            question_urls=question_urls,
            text_questions=text_questions,
        )

        # 4. Trigger Celery Task
        task = run_agentic_pipeline_task.delay(
            project_id=project_id
        )

        # 5. Save the task_id back to the record
        docu_process.task_id = task.id
        docu_process.status = DocuProcess.StatusChoices.PROCESSING
        docu_process.save(update_fields=['task_id', 'status'])

        return Response({
            "message": "Document processing started successfully.",
            "project_id": project_id,
            "task_id": task.id,
            "status": "Processing in background"
        }, status=status.HTTP_202_ACCEPTED)
    


