from django.urls import path

from .views import (
    AddAutomation,
    ShowAutomation
)
app_name = 'trader'

urlpatterns = [
    path('show_automation/', ShowAutomation.as_view(), name='show_automation'),
    path('add_automation/', AddAutomation.as_view(), name='add_automation'),
]
