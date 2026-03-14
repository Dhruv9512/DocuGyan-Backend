from django.urls import path
from .views import DocuProcessView,InitDocuProcessView


urlpatterns = [
   path('init-docu-process/', InitDocuProcessView.as_view(), name='init-docu-process'),
   path('process/', DocuProcessView.as_view(), name='docu-process'),
]