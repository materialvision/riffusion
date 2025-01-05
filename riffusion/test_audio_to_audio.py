import requests

# Define the endpoint URL
url = "http://localhost:5000/audio-to-audio"

# Path to your input audio file
audio_file_path = "/Users/espensommereide/Dropbox/Projects/paviljong/riffusion_tests/garasje2audio/garasje_fugleliv_juni_23_29_start_600265_ms_dur_5120_ms.mp3"

# Parameters for the transformation
params = {
    "prompt": "A futuristic synth melody",  # Example prompt
    "seed": "42",
    "denoising_strength": "0.5"
}

# Open the audio file in binary mode
with open(audio_file_path, "rb") as audio_file:
    # Prepare the files and data for the POST request
    files = {"audio": audio_file}
    response = requests.post(url, files=files, data=params)

# Handle the response
if response.status_code == 200:
    # Save the transformed audio to a file
    output_path = "/Users/espensommereide/Dropbox/Projects/paviljong/riffusion_tests/garasje2audio/garasje_fugleliv_transformed_audio.wav"
    with open(output_path, "wb") as output_file:
        output_file.write(response.content)
    print(f"Transformed audio saved to {output_path}")
else:
    print(f"Request failed with status code {response.status_code}")
    print(f"Response: {response.text}")
