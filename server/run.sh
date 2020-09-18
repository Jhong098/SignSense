#!/bin/bash
python3 -m venv venv
source venv/bin/activate
pip install Flask

export FLASK_APP=app.py
flask run