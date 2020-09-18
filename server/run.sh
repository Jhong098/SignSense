#!/bin/bash
python3 -m venv venv
source venv/bin/activate
pip install Flask
pip install flask_cors

export FLASK_APP=app.py
export FLASK_ENV=development
flask run