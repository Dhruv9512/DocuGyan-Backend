from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework import status
from django.contrib.auth import get_user_model
from rest_framework_simplejwt.tokens import RefreshToken
from users.views import ACCESS_COOKIE, REFRESH_COOKIE, issue_tokens_for_user

ACCESS_MAX_AGE = 30 * 60                # 30 minutes
REFRESH_MAX_AGE = 7 * 24 * 60 * 60      # 7 days


class TokenRefreshView(APIView):
    # This view should be accessible even if the access token is expired
    permission_classes = [AllowAny] 

    def post(self, request):
        # 1. Get the refresh token from the secure cookie
        raw_refresh = request.COOKIES.get(REFRESH_COOKIE)
        
        if not raw_refresh:
            return Response({"error": "Refresh token missing"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # 2. Validate and parse the token
            refresh = RefreshToken(raw_refresh)
            
            # 3. Extract the user identity from the token claims
            # In DocuGyan, we use 'user_uuid' instead of the default 'user_id'
            user_uuid = refresh.get('user_uuid')
            
            if not user_uuid:
                return Response({"error": "Invalid token payload"}, status=status.HTTP_401_UNAUTHORIZED)

            # 4. Fetch the user
            User = get_user_model()
            try:
                user = User.objects.get(user_uuid=user_uuid)
            except User.DoesNotExist:
                return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

            # 5. Generate new pair of tokens (Access + Refresh)
            # This follows the pattern in your login/register views
            access_token, refresh_token =  issue_tokens_for_user(user)
            
            # 6. Prepare response and set the HTTP-only cookies
            response = Response({"message": "Token refreshed successfully"}, status=status.HTTP_200_OK)
            
            response.set_cookie(ACCESS_COOKIE, access_token, max_age=ACCESS_MAX_AGE)
            response.set_cookie(REFRESH_COOKIE, refresh_token, max_age=REFRESH_MAX_AGE)
            
            return response

        except Exception as e:
            # Handle expired or blacklisted refresh tokens
            return Response({"error": "Invalid or expired refresh token"}, status=status.HTTP_401_UNAUTHORIZED)
        
class WSTokenView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        # This endpoint can be used to issue a short-lived token for WebSocket authentication
        # For simplicity, we can issue an access token without a refresh token here
        raw_refresh = request.COOKIES.get(REFRESH_COOKIE)
        
        if not raw_refresh:
            return Response({"error": "Refresh token missing"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            refresh = RefreshToken(raw_refresh)
            user_uuid = refresh.get('user_uuid')
            
            if not user_uuid:
                return Response({"error": "Invalid token payload"}, status=status.HTTP_401_UNAUTHORIZED)

            User = get_user_model()
            try:
                user = User.objects.get(user_uuid=user_uuid)
            except User.DoesNotExist:
                return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

            access_token, _ =  issue_tokens_for_user(user)
            
            return Response({"access_token": access_token}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": "Invalid or expired refresh token"}, status=status.HTTP_401_UNAUTHORIZED)