import uuid
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from core.authentication import CookieJWTAuthentication
from rest_framework.pagination import PageNumberPagination
from django.db.models import Q

from DocuAgent.utils.utility import get_collection_name
from .models import DocuProcess
from .tasks import run_agentic_pipeline_task, process_project_deletion
from django.db import transaction

class InitDocuProcessView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [CookieJWTAuthentication]
    """Step 1: Creates the project ID so the frontend can upload files."""
    def post(self, request):
        # user_uuid = request.data.get('user_uuid')
        user_uuid = str(request.user.user_uuid)
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
    permission_classes = [IsAuthenticated]
    authentication_classes = [CookieJWTAuthentication]
    """Updates the record with URLs and starts the AI."""
    def post(self, request):
        project_id = request.data.get('project_id')
        reference_urls = request.data.get('reference_urls', [])
        question_urls = request.data.get('question_urls', "")
        # user_uuid = request.data.get('user_uuid')
        user_uuid = str(request.user.user_uuid)

        if not project_id:
            return Response({"error": "project_id is required."}, status=status.HTTP_400_BAD_REQUEST)

        if not user_uuid:
            return Response({"error": "user_uuid is required."}, status=status.HTTP_400_BAD_REQUEST)
        
        if not reference_urls and not question_urls:
            return Response({"error": "At least one of reference_urls or question_urls is required."}, status=status.HTTP_400_BAD_REQUEST)
        
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
    permission_classes = [IsAuthenticated]
    authentication_classes = [CookieJWTAuthentication]

    def get(self, request):
        project_id = request.query_params.get('project_id')
        # user_uuid = request.query_params.get('user_uuid')
        user_uuid = str(request.user.user_uuid)

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
                "result_urls": docu_process.results_url,
                "grooming_data": docu_process.grooming_data,
                "created_at": docu_process.created_at,
                "status": docu_process.status,
            }
            return Response(data, status=status.HTTP_200_OK)
        except DocuProcess.DoesNotExist:
            return Response({"error": "Project not found."}, status=status.HTTP_404_NOT_FOUND)



class DocuProcessListView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [CookieJWTAuthentication]

    def get(self, request):
        user_uuid = str(request.user.user_uuid)

        if not user_uuid:
            return Response({"error": "user_uuid is required."}, status=status.HTTP_400_BAD_REQUEST)

        # 1. Base Query with .values() 
        # This returns lightweight dicts instead of heavy Model instances!
        docu_processes = DocuProcess.objects.filter(user_uuid=user_uuid).exclude(
            status__in=[
                DocuProcess.StatusChoices.DELETING, 
                DocuProcess.StatusChoices.DELETED
            ]
        ).values(
            'project_id', 'user_uuid', 'title', 'description', 
            'reference_urls', 'question_urls', 'results_url', 
            'created_at', 'status'
        )

        # 2. Search filtering
        search_keyword = request.query_params.get('search', None)
        if search_keyword:
            docu_processes = docu_processes.filter(
                Q(title__icontains=search_keyword) | 
                Q(description__icontains=search_keyword)
            )

        # 3. Apply ordering
        docu_processes = docu_processes.order_by('-created_at')

        # 4. Paginate directly on the values queryset
        paginator = PageNumberPagination()
        paginator.page_size = 10 
        paginator.page_size_query_param = 'page_size' 
        paginator.max_page_size = 50 

        paginated_data = paginator.paginate_queryset(docu_processes, request)

        # 5. Skip the manual for-loop completely! 
        # Since .values() already gives us dictionaries, we can pass it straight to the response.
        return paginator.get_paginated_response(paginated_data)
    

class GroomingView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [CookieJWTAuthentication]

    def post(self, request):
       
        user_uuid = str(request.user.user_uuid)
        project_id = request.data.get('project_id')
        grooming_data = request.data.get('grooming_data', {})

        if not user_uuid:
            return Response({"error": "user_uuid is required."}, status=status.HTTP_400_BAD_REQUEST)
        if not project_id:
            return Response({"error": "project_id is required."}, status=status.HTTP_400_BAD_REQUEST)
        if not grooming_data:
            return Response({"error": "grooming_data is required."}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            DocuProcess.objects.filter(project_id=project_id, user_uuid=user_uuid).update(grooming_data=grooming_data)
            return Response({"message": "Grooming data updated successfully."}, status=status.HTTP_200_OK)
        except DocuProcess.DoesNotExist:
            return Response({"error": "Project not found."}, status=status.HTTP_404_NOT_FOUND)
    
    def get(self, request):
        user_uuid = str(request.user.user_uuid)
        project_id = request.query_params.get('project_id')

        if not user_uuid:
            return Response({"error": "user_uuid is required."}, status=status.HTTP_400_BAD_REQUEST)
        if not project_id:
            return Response({"error": "project_id is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            docu_process = DocuProcess.objects.get(project_id=project_id, user_uuid=user_uuid)
            return Response({"grooming_data": docu_process.grooming_data}, status=status.HTTP_200_OK)
        except DocuProcess.DoesNotExist:
            return Response({"error": "Project not found."}, status=status.HTTP_404_NOT_FOUND)


class CurrentUserDetailView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [CookieJWTAuthentication]

    def get(self, request):
        user = request.user
        
        # Manually build the dictionary
        data = {
            "id": user.id,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "is_active": user.is_active,
            "date_joined": user.date_joined,
        }
        
        return Response(data)


class DeleteDocuProcessView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [CookieJWTAuthentication]

    def delete(self, request):
        project_id = request.data.get('project_id')
        user_uuid = str(request.user.user_uuid)

        if not project_id:
            return Response({"error": "project_id is required."}, status=status.HTTP_400_BAD_REQUEST)
        
        if not user_uuid:
            return Response({"error": "user_uuid is required."}, status=status.HTTP_400_BAD_REQUEST)

        with transaction.atomic():
            try:
                docu_process = DocuProcess.objects.select_for_update().get(
                    project_id=project_id, 
                    user_uuid=user_uuid,
                )
            except DocuProcess.DoesNotExist:
                return Response({"error": "Project not found."}, status=status.HTTP_404_NOT_FOUND)

            # 3. State Check
            if docu_process.status in [DocuProcess.StatusChoices.DELETING, DocuProcess.StatusChoices.DELETED]:
                return Response(
                    {"error": "Project is already deleted or currently deleting."}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            # 4. Lock the state
            docu_process.status = DocuProcess.StatusChoices.DELETING
            docu_process.save(update_fields=['status'])

        # 5. Hand off to Celery (Must be OUTSIDE the transaction block)
        process_project_deletion.delay(project_id, user_uuid)

        return Response(
            {"message": "Project queued for deletion."}, 
            status=status.HTTP_202_ACCEPTED
        )

