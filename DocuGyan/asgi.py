import os
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'DocuGyan.settings')

# Initialize Django ASGI application early to ensure AppRegistry is populated
django_asgi_app = get_asgi_application()

from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import DocuAgent.routing
import DocuChat.routing
application = ProtocolTypeRouter({
    # Handle standard HTTP requests
    "http": django_asgi_app,
    
    # Handle WebSocket traffic
    "websocket": AuthMiddlewareStack(
        URLRouter(
            DocuAgent.routing.websocket_urlpatterns +
            DocuChat.routing.websocket_urlpatterns
        )
    ),
})