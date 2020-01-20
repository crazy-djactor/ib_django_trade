from django.shortcuts import render
from django.views.generic import FormView


# Create your views here.

class ShowAutomation(FormView):
    template_name = "show_automation.html"

    def dispatch(self, request, *args, **kwargs):
        # Sets a test cookie to make sure the user has cookies enabled
        request.session.set_test_cookie()
        return super().dispatch(request, *args, **kwargs)


class AddAutomation(FormView):
    template_name = "add_automation.html"

    def dispatch(self, request, *args, **kwargs):
        # Sets a test cookie to make sure the user has cookies enabled
        request.session.set_test_cookie()
        return super().dispatch(request, *args, **kwargs)