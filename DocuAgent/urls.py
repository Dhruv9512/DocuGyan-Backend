from django.urls import path
from .views import DocuProcessView, InitDocuProcessView, DocuProcessDataView, DocuProcessListView, GroomingView, CurrentUserDetailView, DeleteDocuProcessView


urlpatterns = [
   path('init-docu-process/', InitDocuProcessView.as_view(), name='init-docu-process'),
   path('process/', DocuProcessView.as_view(), name='docu-process'),
   path('process-data/', DocuProcessDataView.as_view(), name='docu-process-data'),
   path('process-list/', DocuProcessListView.as_view(), name='docu-process-list'),
   path('grooming/', GroomingView.as_view(), name='grooming'),
   path('user-profile/', CurrentUserDetailView.as_view(), name='current-user'),
   path('delete-process/', DeleteDocuProcessView.as_view(), name='delete-docu-process'),
]


