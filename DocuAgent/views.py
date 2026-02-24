import uuid
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

# Import your celery task
from .tasks import run_batch_extraction_pipeline 

class DocuPDFProcessView(APIView):
    """
    API endpoint to trigger the DocuPDF processing pipeline that include multiple reference and question URLs.
    """
    def post(self, request):
        # 1. Receive generalized file URLs
        reference_urls = request.data.get('reference_urls', [])
        question_urls = request.data.get('question_urls', [])
        text_questions = request.data.get('questions', [])

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

        job_id = f"job_{uuid.uuid4().hex[:10]}"

        # 3. Trigger Celery Task 
        # Notice the parameter names are now format-agnostic
        task = run_batch_extraction_pipeline.delay(
            job_id=job_id, 
            reference_file_urls=reference_urls, 
            question_file_urls=question_urls,
            text_questions=text_questions
        )

        return Response({
            "message": "Document processing started successfully.",
            "job_id": job_id,
            "task_id": task.id,
            "status": "Processing in background"
        }, status=status.HTTP_202_ACCEPTED)