from django.conf.urls import url, include
from jsonrpc.backend.django import api

urlpatterns = [
    url(r'', include(api.urls)),
    url(r'^prefix/', include(api.urls)),
]
