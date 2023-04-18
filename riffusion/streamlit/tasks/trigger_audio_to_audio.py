import io
import random
import time
import os
import typing as T
from pathlib import Path

import numpy as np
import pydub
import streamlit as st
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

def render() -> None:
    st.subheader("✨ Trigger Audio to Audio")
    st.write(
        """
    Modify existing audio from a text prompt or interpolate between two.
    """
    )

    with st.expander("Help", False):
        st.write(
            """
            This tool allows you to upload an audio file of arbitrary length and modify it with
            a text prompt. It does this by sweeping over the audio in overlapping clips, doing
            img2img style transfer with riffusion, then stitching the clips back together with
            cross fading to eliminate seams.

            Try a denoising strength of 0.4 for light modification and 0.55 for more heavy
            modification. The best specific denoising depends on how different the prompt is
            from the source audio. You can play with the seed to get infinite variations.
            Currently the same seed is used for all clips along the track.

            If the Interpolation check box is enabled, supports entering two sets of prompt,
            seed, and denoising value and smoothly blends between them along the selected
            duration of the audio. This is a great way to create a transition.
            """
        )

    device = streamlit_util.select_device(st.sidebar)

    with st.sidebar:
        num_inference_steps = T.cast(
            int,
            st.number_input(
                "Steps per sample", value=25, help="Number of denoising steps per model run"
            ),
        )

        guidance = st.number_input(
            "Guidance",
            value=7.0,
            help="How much the model listens to the text prompt",
        )

        scheduler = st.selectbox(
            "Scheduler",
            options=streamlit_util.SCHEDULER_OPTIONS,
            index=0,
            help="Which diffusion scheduler to use",
        )
        assert scheduler is not None

 
    show_clip_details = st.sidebar.checkbox("Show Clip Details", True)
    show_difference = st.sidebar.checkbox("Show Difference", False)
  
    clip_p = get_clip_params()
    counter = streamlit_util.StreamlitCounter()
    params = SpectrogramParams()


    while True:
        while True: #wait for trigger file
            if os.path.exists("/Users/espensommereide/Developer/riffusion/riffusion/streamlit/trigger_file.txt"):
                # Extract text from file
                file_path = "/Users/espensommereide/Developer/riffusion/riffusion/streamlit/trigger_file.txt"
                extracted_text = extract_text_from_file(file_path)
                st.write(extracted_text)
                print(extracted_text)
                os.remove("/Users/espensommereide/Developer/riffusion/riffusion/streamlit/trigger_file.txt")
                # Overwrite the original file with the new content
                with open("/Users/espensommereide/Developer/riffusion/riffusion/streamlit/ready_file.txt", "w") as file:
                    file.write("1, 0;")
                break
            time.sleep(1)
 
        predefined_prompt = extracted_text
        guidance = 7.0
        seed = random.randint(10, 80)
        denoising = 0.55
        negative_prompt = None

        prompt_input_a = PromptInput(
            prompt=predefined_prompt,
            seed=seed,
            negative_prompt=negative_prompt,
            denoising=denoising,
            guidance=guidance
        )

        st.write(f"## Counter: {counter.value}")
        audio_file="/Users/espensommereide/Dropbox/Projects/appendix/sleeping_instruments Project/sleeping_riffusion.wav"
        st.write("#### Original")
        st.write(f"Seed: {seed}")
        #st.audio(audio_file)

        segment = streamlit_util.load_audio_file(audio_file)

        # TODO(hayk): Fix
        if segment.frame_rate != 44100:
            st.warning("Audio must be 44100Hz. Converting")
            segment = segment.set_frame_rate(44100)
        st.write(f"Duration: {segment.duration_seconds:.2f}s, Sample Rate: {segment.frame_rate}Hz")


        start_time_s = clip_p["start_time_s"]
        clip_duration_s = clip_p["clip_duration_s"]
        overlap_duration_s = clip_p["overlap_duration_s"]

        duration_s = min(clip_p["duration_s"], segment.duration_seconds - start_time_s)
        increment_s = clip_duration_s - overlap_duration_s
        clip_start_times = start_time_s + np.arange(0, duration_s - clip_duration_s, increment_s)

        write_clip_details(
            clip_start_times=clip_start_times,
            clip_duration_s=clip_duration_s,
            overlap_duration_s=overlap_duration_s,
        )
        clip_segments = slice_audio_into_clips(
            segment=segment,
            clip_start_times=clip_start_times,
            clip_duration_s=clip_duration_s,
        )
        result_images: T.List[Image.Image] = []
        result_segments: T.List[pydub.AudioSegment] = []
        counter.increment()
        for i, clip_segment in enumerate(clip_segments):

            st.write(f"### Clip {i} at {clip_start_times[i]:.2f}s")

            audio_bytes = io.BytesIO()
            clip_segment.export(audio_bytes, format="wav")

            init_image = streamlit_util.spectrogram_image_from_audio(
                clip_segment,
                params=params,
                device=device,
            )
            # TODO(hayk): Roll this into spectrogram_image_from_audio?
            init_image_resized = scale_image_to_32_stride(init_image)
            progress_callback = None

            image = streamlit_util.run_img2img(
                    prompt=prompt_input_a.prompt,
                    init_image=init_image_resized,
                    denoising_strength=prompt_input_a.denoising,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance,
                    negative_prompt=prompt_input_a.negative_prompt,
                    seed=prompt_input_a.seed,
                    progress_callback=progress_callback,
                    device=device,
                    scheduler=scheduler,
                )
            


            # Resize back to original size
            image = image.resize(init_image.size, Image.BICUBIC)

            result_images.append(image)

            #if show_clip_details:
            #    empty_bin.empty()
            #    right.image(image, use_column_width=False)

            riffed_segment = streamlit_util.audio_segment_from_spectrogram_image(
                image=image,
                params=params,
                device=device,
            )
            result_segments.append(riffed_segment)

            audio_bytes = io.BytesIO()
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
        combined_segment = audio_util.stitch_segments(result_segments, crossfade_s=overlap_duration_s)
        combined_segment.export("/Users/espensommereide/Dropbox/Projects/appendix/sleeping_instruments Project/sleeping_riffusion_out.wav", format="wav")
        st.write(f"#### Final Audio ({combined_segment.duration_seconds}s)")
        #ready file with 1 for max coll
        with open("/Users/espensommereide/Developer/riffusion/riffusion/streamlit/ready_file.txt", "w") as file:
            file.write("1, 1;")
        #input_name = Path(audio_file.name).stem
        #output_name = f"{input_name}_{prompt_input_a.prompt.replace(' ', '_')}"
        #streamlit_util.display_and_download_audio(combined_segment, output_name, extension=extension)
        #st.experimental_rerun()

def get_clip_params(advanced: bool = False) -> T.Dict[str, T.Any]:
    """
    Render the parameters of slicing audio into clips.
    """
    p: T.Dict[str, T.Any] = {}

    cols = st.columns(4)

    p["start_time_s"] = cols[0].number_input(
        "Start Time [s]",
        min_value=0.0,
        value=0.0,
    )
    p["duration_s"] = cols[1].number_input(
        "Duration [s]",
        min_value=0.0,
        value=10.0,
    )

    if advanced:
        p["clip_duration_s"] = cols[2].number_input(
            "Clip Duration [s]",
            min_value=3.0,
            max_value=10.0,
            value=5.0,
        )
    else:
        p["clip_duration_s"] = 5.0

    if advanced:
        p["overlap_duration_s"] = cols[3].number_input(
            "Overlap Duration [s]",
            min_value=0.0,
            max_value=10.0,
            value=0.2,
        )
    else:
        p["overlap_duration_s"] = 0.2

    return p


def write_clip_details(
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
        )


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
