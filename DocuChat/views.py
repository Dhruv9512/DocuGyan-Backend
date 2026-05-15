from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated


from core.authentication import CookieJWTAuthentication
from DocuChat.models import ChatSession, ChatMessage


class ChatSessionListView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [CookieJWTAuthentication]
    def get(self, request):
        try:
            project_id = request.query_params.get('project_id')
            print(f"Fetching chat sessions for project_id: {project_id} and user: {request.user.email}")
            if not project_id:
                return Response(
                    {"error": "project_id is required"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            sessions = ChatSession.objects.filter(
                project_id=project_id, 
                user=request.user
            ).order_by('-created_at').values('session_id', 'title', 'created_at')

            # Convert the QuerySet of dictionaries directly to a list
            return Response(list(sessions), status=status.HTTP_200_OK)

        except Exception as e:
            # Log the actual error for your debugging purposes
            print(f"Error in ChatSessionListView: {str(e)}")
            
            # Return a clean, safe error message to the frontend
            return Response(
                {"error": "An unexpected error occurred while fetching sessions."}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
class ChatSessionDetailView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [CookieJWTAuthentication]
    def get(self, request, session_id):
        try:
            session = ChatSession.objects.get(session_id=session_id, user=request.user)
            messages = ChatMessage.objects.filter(session=session).order_by('created_at').values(
                'user_message', 'assistant_response', 'created_at'
            )

            return Response({
                "session_id": session.session_id,
                "title": session.title,
                "created_at": session.created_at,
                "messages": list(messages)
            }, status=status.HTTP_200_OK)

        except ChatSession.DoesNotExist:
            return Response(
                {"error": "Chat session not found."}, 
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            print(f"Error in ChatSessionDetailView: {str(e)}", exc_info=True)
            return Response(
                {"error": "An unexpected error occurred while fetching the chat session."}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
    def delete(self, request, session_id):
        try:
            session = ChatSession.objects.get(session_id=session_id, user=request.user)
            session.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)

        except ChatSession.DoesNotExist:
            return Response(
                {"error": "Chat session not found."}, 
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            print(f"Error in ChatSessionDetailView DELETE: {str(e)}", exc_info=True)
            return Response(
                {"error": "An unexpected error occurred while deleting the chat session."}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )



