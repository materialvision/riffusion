import io
import random
import time
import os
import typing as T
from typing import Dict, Any
from pathlib import Path

import numpy as np
import pydub
#import streamlit as st
from PIL import Image

from riffusion.datatypes import InferenceInput, PromptInput
from riffusion.spectrogram_params import SpectrogramParams
from riffusion.streamlit import util as streamlit_util
from riffusion.streamlit.tasks.interpolation import get_prompt_inputs, run_interpolation
from riffusion.util import audio_util

def extract_text_from_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('1,'):
                # Remove "1," and strip any leading/trailing whitespace
                extracted_text = line[2:].strip()
                extracted_text = extracted_text[:-1] #remove last semicolon
                return extracted_text

    # If no line starts with "1,", return an default string
    return 'dreampop'

def extract_modeltext_from_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('1,'):
                # Remove "1," and strip any leading/trailing whitespace
                extracted_text = line[2:].strip()
                # Remove the last semicolon
                extracted_text = extracted_text[:-1]
                
                # Split the text into a number and a remaining string by the first space
                parts = extracted_text.split(' ', 1)
                if len(parts) == 2:
                    return parts[0], parts[1]
                else:
                    # If there's no space, return the whole text as the second part with an empty number
                    return '', parts[0]
    # If no line starts with "1,", return default values
    return '10', ''

def main_loop():

    device = "mps"
    checkpoint = "/Users/espensommereide/Developer/diffusion_convert_ckpt/phonophani-cap-style-80000"

    num_inference_steps = 21 #25 best 12 er under 10 sek 11 under 10 for 5 sek

    guidance =7.0


    '''scheduler = st.selectbox(
        "Scheduler",
        options=streamlit_util.SCHEDULER_OPTIONS,
        index=0,
        help="Which diffusion scheduler to use",
    )
    assert scheduler is not None'''


    clip_p = get_clip_params()
    #counter = streamlit_util.StreamlitCounter()
    params = SpectrogramParams(num_griffin_lim_iters=11)
    #params.num_griffin_lim_iters=11

    with open("/Users/espensommereide/Developer/riffusion/riffusion/streamlit/ready_file.txt", "w") as file:
        file.write("1, 1;")

    while True:
        start_time = time.time()
        while True: #wait for trigger file
            if os.path.exists("/Users/espensommereide/Developer/riffusion/riffusion/streamlit/trigger_file.txt"):
                # Extract text from file
                file_path = "/Users/espensommereide/Developer/riffusion/riffusion/streamlit/trigger_file.txt"
                extracted_text = extract_text_from_file(file_path)
                #st.write(extracted_text)
                print(extracted_text)
                # Extract new model from file
                modelfile_path = "/Users/espensommereide/Developer/riffusion/riffusion/streamlit/model_file.txt"
                model_time,model_name = extract_modeltext_from_file(modelfile_path)

                print(model_time+" "+model_name)
                if model_name!="":
                    checkpoint=model_name
                if model_time!="":
                    live_duration=int(model_time)
                else: 
                    live_duration=int(10)
                os.remove("/Users/espensommereide/Developer/riffusion/riffusion/streamlit/trigger_file.txt")
                # Overwrite the original file with the new content
                with open("/Users/espensommereide/Developer/riffusion/riffusion/streamlit/ready_file.txt", "w") as file:
                    file.write("1, 0;")
                break
            time.sleep(1)

        params = SpectrogramParams(num_griffin_lim_iters=11)
        predefined_prompt = extracted_text
        guidance = 7.0
        seed = random.randint(10, 80)
        denoising = 0.55 #95 tregere men renere
        negative_prompt = None #'speech' #None

        prompt_input_a = PromptInput(
            prompt=predefined_prompt,
            seed=seed,
            negative_prompt=negative_prompt,
            denoising=denoising,
            guidance=guidance
        )

        #st.write(f"## Counter: {counter.value}")
        audio_file="/Users/espensommereide/Developer/diffusion_convert_ckpt/sleeping_riffusion.wav"
        #st.write("#### Original")
        #st.write(f"Seed: {seed}")
        #st.audio(audio_file)

        segment = streamlit_util.load_audio_file(audio_file)

        # TODO(hayk): Fix
        if segment.frame_rate != 44100:
            print("Audio must be 44100Hz. Converting")
            segment = segment.set_frame_rate(44100)
        #st.write(f"Duration: {segment.duration_seconds:.2f}s, Sample Rate: {segment.frame_rate}Hz")

        start_time_s = clip_p["start_time_s"]
        clip_duration_s = clip_p["clip_duration_s"]
        overlap_duration_s = clip_p["overlap_duration_s"]

        duration_s = min(live_duration, segment.duration_seconds - start_time_s)#min(clip_p["duration_s"], segment.duration_seconds - start_time_s)
        increment_s = clip_duration_s - overlap_duration_s
        clip_start_times = start_time_s + np.arange(0, duration_s - clip_duration_s, increment_s)

        '''write_clip_details(
            clip_start_times=clip_start_times,
            clip_duration_s=clip_duration_s,
            overlap_duration_s=overlap_duration_s,
        )'''
        clip_segments = slice_audio_into_clips(
            segment=segment,
            clip_start_times=clip_start_times,
            clip_duration_s=clip_duration_s,
        )
        #result_images: T.List[Image.Image] = []
        result_segments: T.List[pydub.AudioSegment] = []
        #counter.increment()

        if (len(clip_segments)==0):
            clip_segments.append(segment)
        print("clip_segments: ",len(clip_segments))
        for i, clip_segment in enumerate(clip_segments):

            #st.write(f"### Clip {i} at {clip_start_times[i]:.2f}s")

            audio_bytes = io.BytesIO()
            clip_segment.export(audio_bytes, format="wav")
            print("spectrogramming")
            init_image = streamlit_util.spectrogram_image_from_audio(
                clip_segment,
                params=params,
                device=device,
            )
            # TODO(hayk): Roll this into spectrogram_image_from_audio?
            #print("scaling")
            #init_image_resized = scale_image_to_32_stride(init_image)
            #This line would call a function scale_image_to_32_stride to resize the init_image so that its dimensions are multiples of 32. This is often necessary for image processing tasks, especially when dealing with models that require input images with specific size constraints.
            #progress_callback = None
            print("img to img")
            image = streamlit_util.run_img2img(
                    prompt=prompt_input_a.prompt,
                    init_image=init_image,
                    denoising_strength=prompt_input_a.denoising,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance,
                    negative_prompt=prompt_input_a.negative_prompt,
                    seed=prompt_input_a.seed,
                    progress_callback=None,
                    checkpoint=checkpoint,
                    device=device,
                    scheduler="DPMSolverMultistepScheduler",
                )
            

            #print("resize")
            # Resize back to original size
            #image = image.resize(init_image.size, Image.BICUBIC)
            #print("append result images")
            #result_images.append(image)

            #if show_clip_details:
            #    empty_bin.empty()
            #    right.image(image, use_column_width=False)
            print("riffed segment")
            riffed_segment = streamlit_util.audio_segment_from_spectrogram_image(
                image=image,
                params=params,
                device=device,
            )
            print("append riffed segments")
            result_segments.append(riffed_segment)
            print("bytes")
            audio_bytes = io.BytesIO()
            print("export")
            riffed_segment.export(audio_bytes, format="wav")
            """
            if show_clip_details:
                right.audio(audio_bytes)

            if show_clip_details and show_difference:
                diff_np = np.maximum(
                    0, np.asarray(init_image).astype(np.float32) - np.asarray(image).astype(np.float32)
                )
                diff_image = Image.fromarray(255 - diff_np.astype(np.uint8))
                diff_segment = streamlit_util.audio_segment_from_spectrogram_image(
                    image=diff_image,
                    params=params,
                    device=device,
                )

                audio_bytes = io.BytesIO()
                diff_segment.export(audio_bytes, format=extension)
                st.audio(audio_bytes)
            """
        # Combine clips with a crossfade based on overlap
        print("combined_segment")
        print("result_segments: ",len(result_segments))
        combined_segment = audio_util.stitch_segments(result_segments, crossfade_s=overlap_duration_s)
        #combined_segment=result_segments[0]
        combined_segment.export("/Users/espensommereide/Developer/diffusion_convert_ckpt/sleeping_riffusion_out.wav", format="wav")
        #st.write(f"#### Final Audio ({combined_segment.duration_seconds}s)")
        #ready file with 1 for max coll
        with open("/Users/espensommereide/Developer/riffusion/riffusion/streamlit/ready_file.txt", "w") as file:
            file.write("1, 1;")
        #input_name = Path(audio_file.name).stem
        #output_name = f"{input_name}_{prompt_input_a.prompt.replace(' ', '_')}"
        #streamlit_util.display_and_download_audio(combined_segment, output_name, extension=extension)
        #st.experimental_rerun()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"total elapsed time: {elapsed_time:.4f} seconds")  

def get_clip_params(advanced: bool = False) -> Dict[str, Any]:
    """
    Get the parameters of slicing audio into clips.
    """
    p: Dict[str, Any] = {}
    # Hardcoding the values instead of using streamlit for inputs.
    p["start_time_s"] = 0.0
    p["duration_s"] = 10.0
    
    if advanced:
        p["clip_duration_s"] = 5.0 # Adjust as needed
        p["overlap_duration_s"] = 0.2 # Adjust as needed
    else:
        p["clip_duration_s"] = 5.0
        p["overlap_duration_s"] = 0.2
    
    return p


'''def write_clip_details(
    clip_start_times: np.ndarray, clip_duration_s: float, overlap_duration_s: float
):
    """
    Write details of the clips to be sliced from an audio segment.
    """
    clip_details_text = (
        f"Slicing {len(clip_start_times)} clips of duration {clip_duration_s}s "
        f"with overlap {overlap_duration_s}s"
    )

    with st.expander(clip_details_text):
        st.dataframe(
            {
                "Start Time [s]": clip_start_times,
                "End Time [s]": clip_start_times + clip_duration_s,
                "Duration [s]": clip_duration_s,
            }
        )'''


def slice_audio_into_clips(
    segment: pydub.AudioSegment, clip_start_times: T.Sequence[float], clip_duration_s: float
) -> T.List[pydub.AudioSegment]:
    """
    Slice an audio segment into a list of clips of a given duration at the given start times.
    """
    clip_segments: T.List[pydub.AudioSegment] = []
    for i, clip_start_time_s in enumerate(clip_start_times):
        clip_start_time_ms = int(clip_start_time_s * 1000)
        clip_duration_ms = int(clip_duration_s * 1000)
        clip_segment = segment[clip_start_time_ms : clip_start_time_ms + clip_duration_ms]

        # TODO(hayk): I don't think this is working properly
        if i == len(clip_start_times) - 1:
            silence_ms = clip_duration_ms - int(clip_segment.duration_seconds * 1000)
            if silence_ms > 0:
                clip_segment = clip_segment.append(pydub.AudioSegment.silent(duration=silence_ms))

        clip_segments.append(clip_segment)

    return clip_segments


def scale_image_to_32_stride(image: Image.Image) -> Image.Image:
    """
    Scale an image to a size that is a multiple of 32.
    """
    closest_width = int(np.ceil(image.width / 32) * 32)
    closest_height = int(np.ceil(image.height / 32) * 32)
    return image.resize((closest_width, closest_height), Image.BICUBIC)

if __name__ == "__main__":
    main_loop()