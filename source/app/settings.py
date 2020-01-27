import os
import environ

IS_PRODUCTION = os.environ.get('IS_PRODUCTION')

env = environ.Env(IS_PRODUCTION=(bool, False))
environ.Env.read_env()

IS_PRODUCTION = env('IS_PRODUCTION')

if IS_PRODUCTION:
    from .conf.production.settings import *
else:
    from .conf.development.settings import *
