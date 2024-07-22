# Interactive Brokers Trading with Django

An Django project with Algo

## Functionality

- Trading using ib_insync package
- Log in/Sign Up
    - via username & password
    - via email & password
    - via email or username & password
    - with a remember me checkbox (optional)
- Create an account
- Multilingual: English, Russian, and Simplified Chinese


## Installing

### Clone the project

```
https://github.com/crazy-djactor/ib_django_trade.git
```

### Install dependencies & activate virtualenv

```
pip install pipenv

pipenv install
pipenv shell
```

### Configure the settings (connection to the database, connection to an SMTP server, and other options)

1. Edit `source/app/conf/development/settings.py` if you want to develop the project.

2. Edit `source/app/conf/production/settings.py` if you want to run the project in production.

### Apply migrations

```
python source/manage.py migrate
```

### Collect static files (only on a production server)

```
python source/manage.py collectstatic
```

### Running

#### A development server

Just run this command:

```
python source/manage.py runserver
```

Anydesk means mobile number. I just wanted to see where you connected - Wfe or Mmy.
Sorry, because some guy from Wfe notified me she didn't get yet. 
Maybe Wfe is confused, or that guy, I believe. but just wanted to check again. Sorry bro.
