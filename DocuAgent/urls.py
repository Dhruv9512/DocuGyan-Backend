from django.urls import path
from .views import DocuProcessView


urlpatterns = [
   path('process/', DocuProcessView.as_view(), name='docu-process'),
]