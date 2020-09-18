from flask import Flask, jsonify, make_response
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)


@app.route('/', methods=['GET'])
def test():
    return jsonify({"data": "hello react"})
