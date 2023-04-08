import io
import typing as T
from pathlib import Path

import numpy as np
import pydub
from PIL import Image
from pythonosc import dispatcher
from pythonosc import osc_server

from riffusion.datatypes import InferenceInput, PromptInput
from riffusion.spectrogram_params import SpectrogramParams
from riffusion.util import audio_util

# Import the required functions from your project here
# For example:
# from my_project import run_interpolation, get_prompt_inputs, scale_image_to_32_stride

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

def handle_osc_message(address, *args):
    print(f"Received OSC message: {address}, {args}")

    if address == "/render_audio_to_audio":
        input_file = args[0]
        output_file = args[1]
        start_time = float(args[2]) if len(args) > 2 else 0.0
        end_time = float(args[3]) if len(args) > 3 else None
        render_audio_to_audio(input_file, output_file, start_time, end_time)
    else:
        print(f"Unhandled OSC address: {address}")

def render_audio_to_audio(device, extension, use_magic_mix, num_inference_steps, guidance, scheduler, audio_file, clip_p, interpolate, counter, prompt_input_a, prompt_input_b=None, magic_mix_kmin=None, magic_mix_kmax=None, magic_mix_mix_factor=None):

    if not audio_file:
        print("Upload audio to get started")
        return

    print("Original")
    print(f"Duration: {segment.duration_seconds:.2f}s, Sample Rate: {segment.frame_rate}Hz")

    start_time_s = clip_p["start_time_s"]
    clip_duration_s = clip_p["clip_duration_s"]
    overlap_duration_s = clip_p["overlap_duration_s"]

    duration_s = min(clip_p["duration_s"], segment.duration_seconds - start_time_s)
    increment_s = clip_duration_s - overlap_duration_s
    clip_start_times = start_time_s + np.arange(0, duration_s - clip_duration_s, increment_s)

    # ... rest of the code remains the same as before ...
    # Combine clips with a crossfade based on overlap
    
    combined_segment = audio_util.stitch_segments(result_segments, crossfade_s=overlap_duration_s)

def handle_osc_message(*args):
    # Extract arguments and call the render_audio_to_audio function
    pass

def main():
    ip = "127.0.0.1"
    port = 8000

    dispatcher = Dispatcher()
    dispatcher.map("/render_audio_to_audio", handle_osc_message)

    server = BlockingOSCUDPServer((ip, port), dispatcher)
    print(f"Serving on {ip}:{port}")
    server.serve_forever()

if __name__ == "__main__":
    main()
