import os
import environ

env = environ.Env(IS_PRODUCTION=(bool, True))
environ.Env.read_env()

IS_PRODUCTION = env('IS_PRODUCTION')
IS_PRODUCTION = True
if IS_PRODUCTION:
    from .conf.production.settings import *
else:
    from .conf.development.settings import *
