from flask import Flask, request
import requests

app = Flask(__name__)

@app.route('/trigger', methods=['POST'])
def trigger():
    # Replace this URL with the URL of your Streamlit application
    streamlit_app_url = "http://127.0.0.1:8501"

    # Send a request to your Streamlit application to trigger a specific action
    response = requests.post(f"{streamlit_app_url}/api/trigger_action", json=request.json)
    return response.content

if __name__ == '__main__':
    app.run(port=5050)
