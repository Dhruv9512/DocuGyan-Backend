# DocuChat/urls.py
from django.urls import path
from DocuChat.views import ChatSessionListView, ChatSessionDetailView, ChatSessionDetailView

urlpatterns = [
    path('sessions/',ChatSessionListView.as_view(),name='chat-session-list'),
    path('sessions/<uuid:session_id>/', ChatSessionDetailView.as_view(), name='chat-session-detail'),
    path('api/chat/sessions/<uuid:session_id>/', ChatSessionDetailView.as_view(), name='chat-session-detail'),
]