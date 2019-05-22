SECRET_KEY = 'secret'
ROOT_URLCONF = 'jsonrpc.tests.test_backend_django.urls'
ALLOWED_HOSTS = ['testserver']
DATABASE_ENGINE = 'django.db.backends.sqlite3'
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': ':memory:',
    }
}
JSONRPC_MAP_VIEW_ENABLED = True
