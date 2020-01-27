import asyncio
import json
import urllib

from django.contrib import messages
from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import render
from django.urls import reverse_lazy
from django.views.generic import FormView
from .forms import ShowAutomationForm, AddAutomationForm
from .util import Util, tradingThread, openSample
from tornado.platform.asyncio import AnyThreadEventLoopPolicy
from django.utils.translation import gettext_lazy as _

# Create your views here.

asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())
util = Util()
google_data = openSample()

class ShowAutomation(FormView):
    template_name = "show_automation.html"
    form_class = ShowAutomationForm
    sheet = {"expiration_call": '20200207',
             "strike_call": '325',
             "expiration_put": '20200207',
             "strike_put": '312.50',
             "ten_years_yield": '1.84',
             "time_exp": '0.049',
             "opt_diff": '1.020',
             "num_of_contracts": '0',
             "long_threshold": '0.58',
             "short_threshold": '0.50'}

    def get(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            # form = self.form_class(initial=self.initial)
            # self.sheet = Util().getSheet()
            if request.session['strText'] is not None:
                response = HttpResponse()
                response['Content-Type'] = 'text/plain'
                response['Content-Disposition'] = 'attachment; filename=result.txt'
                response.write(request.session['strText'])
                return response
            # return render(request, self.template_name, {'form': form, 'sheet': self.sheet})
        else:
            return HttpResponseRedirect('/accounts/log-in/')

    def get_initial(self):
        initial = super(ShowAutomation, self).get_initial()
        if self.request.user.is_authenticated:
            initial.update({'sheet': self.sheet})
        return initial

    def dispatch(self, request, *args, **kwargs):
        # Sets a test cookie to make sure the user has cookies enabled
        request.session.set_test_cookie()
        return super().dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        return super(ShowAutomation, self).form_valid(form)


class AddAutomationView(FormView):
    template_name = "add_automation.html"
    form_class = AddAutomationForm
    sheet = {
            "symbol_1": "AAPL",
            "expiration_call": '20200207',
             "strike_call": '325',
             "expiration_put": '20200207',
             "strike_put": '312.50',
             "ten_years_yield": '1.84',
             "time_exp": '0.049',
             "opt_diff": '1.020',
             "num_of_contracts": '0',
             "long_threshold": '0.58',
             "short_threshold": '0.50'}

    success_urls = {
        'contact': reverse_lazy('contact-form-redirect'),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.util = util
        self.google_data = openSample()

    def dispatch(self, request, *args, **kwargs):
        # Sets a test cookie to make sure the user has cookies enabled
        request.session.set_test_cookie()
        return super().dispatch(request, *args, **kwargs)

    # def get_context_data(self, **kwargs):
    #     context = super(AddAutomationView, self).get_context_data(**kwargs)
    #     context['sheet'] = self.sheet
    #     return context

    # def form_valid(self, form):
    #     return super(AddAutomationView, self).form_valid(form)

    def get(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            form = self.form_class(initial=self.initial)
            self.sheet = globals()['util'].getSheet()
            # self.google_data = openSample()
            return render(request, self.template_name, {'form': form, 'sheet': self.sheet, 'sheets': globals()['google_data']})
        else:
            return HttpResponseRedirect('/accounts/log-in/')

    def post(self, request, *args, **kwargs):
        form = self.form_class(request.POST)
        if 'run_algo' in request.POST:
            if form.is_valid():
                # thread1 = tradingThread(self.util, inputArg=form.data)
                # thread1.start()
                strText = ""
                retval = ""
                try:
                    # [strText, retval] = globals()['util'].getTrade(form.data)
                    [strText, retval] = self.util.getTrade(form.data)
                except:
                    strText = ''
                    pass
                if strText == '':
                    messages.error(request, _('Can not get information!'))
                else:
                    request.session['strText'] = strText
                    request.session['retVal'] = json.dumps(retval, sort_keys=True)
                    return HttpResponseRedirect('/trader/show_automation/')
            return render(request, self.template_name, {'form': form, 'sheets': globals()['google_data'], 'sheet': self.sheet})
        elif 'update_sheet' in request.POST:
            globals()['google_data'] = openSample()
            return render(request, self.template_name, {'form': form, 'sheets': globals()['google_data'], 'sheet': self.sheet})
