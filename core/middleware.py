# core/middleware.py
import logging
from urllib.parse import parse_qs
from django.conf import settings
from channels.middleware import BaseMiddleware
from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser
from rest_framework_simplejwt.tokens import AccessToken
from rest_framework_simplejwt.exceptions import TokenError

logger = logging.getLogger(__name__)
User = get_user_model()

@database_sync_to_async
def get_user_from_token(user_uuid):
    try:
        # Match the query to your custom user model's UUID field
        return User.objects.get(user_uuid=user_uuid)
    except User.DoesNotExist:
        return AnonymousUser()

@database_sync_to_async
def get_dev_user():
    """Fetches the first user in the DB to act as the authenticated dev user."""
    return User.objects.first() or AnonymousUser()

class JWTAuthWebSocketMiddleware(BaseMiddleware):
    """
    Custom Middleware to authenticate WebSockets using JWT from the query string.
    """
    async def __call__(self, scope, receive, send):
        # 1. Extract query string
        query_string = scope.get("query_string", b"").decode("utf-8")
        query_params = parse_qs(query_string)
        
        # 2. Extract token
        token = query_params.get("token", [None])[0]

        # 3. Authenticate
        scope['user'] = AnonymousUser()
        if token:
            try:
                access_token = AccessToken(token)
                
                # Use 'user_uuid' because of your SIMPLE_JWT settings in settings.py
                user_uuid = access_token.get('user_uuid')
                if user_uuid:
                    scope['user'] = await get_user_from_token(user_uuid)
                    
            except TokenError as e:
                logger.warning(f"WebSocket Auth failed: Invalid token - {str(e)}")
            except Exception as e:
                logger.error(f"WebSocket Auth error: {str(e)}")

        # 4. --- DEVELOPMENT BYPASS ---
        # If running locally (DEBUG=True) and no valid token was provided
        if settings.DEBUG and not scope['user'].is_authenticated:
            logger.info("DEBUG is True: Bypassing WebSocket Auth and assigning a default dev user.")
            scope['user'] = await get_dev_user()

        # 5. Pass control to the next application/middleware
        return await super().__call__(scope, receive, send)