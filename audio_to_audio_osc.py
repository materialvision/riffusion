import io
import typing as T
from pathlib import Path
import numpy as np
import pydub
import streamlit as st
from PIL import Image
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc.udp_client import SimpleUDPClient

from riffusion.datatypes import InferenceInput, PromptInput
from riffusion.spectrogram_params import SpectrogramParams
from riffusion.streamlit import util as streamlit_util
from riffusion.streamlit.pages.interpolation import get_prompt_inputs, run_interpolation
from riffusion.util import audio_util

# OSC Server settings
ip = "127.0.0.1"
osc_port = 8000
osc_client = SimpleUDPClient(ip, osc_port)
osc_dispatcher = dispatcher.Dispatcher()

# Function to load the soundfile and trigger the render process
def load_soundfile_handler(unused_addr, soundfile_path):
    soundfile_path = Path(soundfile_path)
    if soundfile_path.exists() and soundfile_path.is_file():
        with open(soundfile_path, "rb") as f:
            content = f.read()
            audio_file = io.BytesIO(content)
            audio_file.name = str(soundfile_path)
            st.session_state["audio_file"] = audio_file
            osc_client.send_message("/load_soundfile_status", "Soundfile loaded successfully.")
    else:
        osc_client.send_message("/load_soundfile_status", "Error: Invalid soundfile path.")


# Function to save the final result as a new file
def save_final_result(file_name, audio_segment, extension="wav"):
    output_path = Path(file_name + "." + extension)
    with output_path.open("wb") as f:
        audio_segment.export(f, format=extension)
    osc_client.send_message("/save_final_result_status", f"Result saved to {output_path}")


# Add the load_soundfile_handler to the OSC dispatcher
osc_dispatcher.map("/load_soundfile", load_soundfile_handler)

# Start the OSC server
server = osc_server.ThreadingOSCUDPServer((ip, osc_port), osc_dispatcher)
print(f"Serving on {server.server_address}")

# Run the Streamlit app in a separate thread
from threading import Thread

streamlit_thread = Thread(target=st.run, args=("streamlit run riffusion/streamlit/playground.py --browser.serverAddress 127.0.0.1 --browser.serverPort 8501",))
streamlit_thread.start()

server.serve_forever()
