# SignSense

## Folder Structure

`client/` contains all the FE React app that will interface with the user

`server/` contains the BE Flask server that will call other scripts to process input and return predictions

## Running Locally

### Frontend

Prerequisites:

- [Node](https://github.com/nvm-sh/nvm)

1. `cd client && npm i && npm start`

now you should see the React app in your browser at `localhost:3000`

### Backend

Prerequisites:

- Python 3
- pip
- venv

1. `cd server`
2. `./run.sh`

now you should see the Flask server at `localhost:5000`
