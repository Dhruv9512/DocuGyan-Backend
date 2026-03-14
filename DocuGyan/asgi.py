import os
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'DocuGyan.settings')

# Initialize Django ASGI application early to ensure AppRegistry is populated
django_asgi_app = get_asgi_application()

from channels.routing import ProtocolTypeRouter, URLRouter
import DocuAgent.websocket.routing
import DocuChat.routing
from core.middleware import JWTAuthWebSocketMiddleware


application = ProtocolTypeRouter({
    # Handle standard HTTP requests
    "http": django_asgi_app,
    
    # Handle WebSocket traffic
    "websocket": JWTAuthWebSocketMiddleware(
        URLRouter(
            DocuAgent.websocket.routing.websocket_urlpatterns +
            DocuChat.routing.websocket_urlpatterns
        )
    ),
})