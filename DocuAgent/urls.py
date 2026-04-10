from django.urls import path
from .views import DocuProcessView, InitDocuProcessView, DocuProcessDataView, DocuProcessListView


urlpatterns = [
   path('init-docu-process/', InitDocuProcessView.as_view(), name='init-docu-process'),
   path('process/', DocuProcessView.as_view(), name='docu-process'),
   path('process-data/', DocuProcessDataView.as_view(), name='docu-process-data'),
   path('process-list/', DocuProcessListView.as_view(), name='docu-process-list'),
]