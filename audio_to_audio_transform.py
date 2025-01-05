import numpy as np
import pydub
import torch
from scipy.io import wavfile
from pathlib import Path
from riffusion.riffusion_pipeline import RiffusionPipeline
from riffusion.util.audio_util import audio_from_waveform

def process_audio_with_text(
    input_audio_path: str,
    output_audio_path: str,
    text_prompt: str,
    model_path: str,
    denoising_strength: float = 0.55,
    overlap: int = 500,  # in milliseconds
    steps: int = 25
):
    """
    Transform audio using a custom diffusion model and a text prompt.
    
    Args:
        input_audio_path: Path to the input audio file.
        output_audio_path: Path to save the transformed audio.
        text_prompt: The text prompt guiding the audio transformation.
        model_path: Path to the custom diffusion model.
        denoising_strength: Degree of modification (default 0.55).
        overlap: Overlap between clips in milliseconds (default 500).
        steps: Number of inference steps for the diffusion model (default 25).
    """
    # Load input audio
    audio_segment = pydub.AudioSegment.from_file(input_audio_path)
    sample_rate = audio_segment.frame_rate

    # Initialize the diffusion pipeline
    pipeline = RiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        revision="main",
        safety_checker=None,
        device="cuda" if torch.cuda.is_available() else "mps"
    )

    # Prepare audio clips with overlap
    clip_duration = 5 * 1000  # 5 seconds in milliseconds
    clips = []
    for start in range(0, len(audio_segment), clip_duration - overlap):
        clip = audio_segment[start: start + clip_duration]
        if len(clip) < clip_duration:  # Pad the last clip if it's shorter
            clip = clip + pydub.AudioSegment.silent(duration=clip_duration - len(clip))
        clips.append(clip)

    # Process each clip with the diffusion pipeline
    processed_clips = []
    for clip in clips:
        samples = np.array(clip.get_array_of_samples()).astype(np.float32)
        samples = samples / np.iinfo(np.int16).max  # Normalize to [-1, 1]

        # Run inference with the pipeline
        modified_waveform = pipeline.audio_to_audio(
            audio=samples,
            prompt=text_prompt,
            denoising=denoising_strength,
            num_inference_steps=steps,
        )
        
        # Convert back to pydub AudioSegment
        processed_clip = audio_from_waveform(
            samples=modified_waveform,
            sample_rate=sample_rate,
            normalize=True
        )
        processed_clips.append(processed_clip)

    # Combine processed clips with crossfade
    output_audio = processed_clips[0]
    for next_clip in processed_clips[1:]:
        output_audio = output_audio.append(next_clip, crossfade=overlap)

    # Export the result
    output_audio.export(output_audio_path, format="wav")
    print(f"Processed audio saved to: {output_audio_path}")

# Example usage:
# process_audio_with_text(
#     input_audio_path="input.wav",
#     output_audio_path="output.wav",
#     text_prompt="a calming piano melody",
#     model_path="custom_diffusion_model"
# )

# Example usage:
process_audio_with_text(
    input_audio_path="/Users/espensommereide/Dropbox/Projects/paviljong/riffusion_tests/garasje2audio/garasje_fugleliv_juni_23_29_start_600265_ms_dur_5120_ms.mp3",
    output_audio_path="output.wav",
    text_prompt="a calming piano melody",
    model_path="/Users/espensommereide/Developer/diffusion_convert_ckpt/phonophani-quad-10000"
)
