# DocuChat/routing.py
from django.urls import re_path
from DocuChat.websocket import consumers

websocket_urlpatterns = [
    re_path(r'ws/chat/(?P<project_id>[\w\-]+)/(?P<session_id>[\w\-]+)/$', consumers.ChatConsumer.as_asgi()),
]