from django.urls import path
from .views import TokenRefreshView, WSTokenView
urlpatterns = [
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('ws-token/', WSTokenView.as_view(), name='ws_token'),
]