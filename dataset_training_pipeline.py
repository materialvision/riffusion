import os
import subprocess

def run_command(command):
    """Run a shell command."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, check=True)
    return result

def generate_training_commands(final_output_dir, base_output_dir, pretrained_model_name_or_path,dreambooth_label):
    # Construct the dynamic paths
    session_data_dir = os.path.join(base_output_dir, f"{os.path.basename(os.path.normpath(final_output_dir))}")
    instance_data_dir = os.path.join(base_output_dir, f"{os.path.basename(os.path.normpath(final_output_dir))}/instance_images")
    captions_dir = os.path.join(base_output_dir, f"{os.path.basename(os.path.normpath(final_output_dir))}/captions")
    output_dir = os.path.join(base_output_dir, f"{dreambooth_label}_final_output")

    # Command 1: Train text encoder
    train_text_encoder_command = (
        f"accelerate launch diffusers/examples/dreambooth/train_dreambooth.py \\\n"
        f"--offset_noise \\\n"
        f"--external_captions \\\n"
        f"--image_captions_filename \\\n"
        f"--train_text_encoder \\\n"
        f"--dump_only_text_encoder \\\n"
        f"--pretrained_model_name_or_path=\"{pretrained_model_name_or_path}\" \\\n"
        f"--instance_data_dir=\"{instance_data_dir}\" \\\n"
        f"--output_dir=\"{output_dir}\" \\\n"
        f"--captions_dir=\"{captions_dir}\" \\\n"
        f"--instance_prompt='' \\\n"
        f"--seed=1000 \\\n"
        f"--resolution=512 \\\n"
        f"--mixed_precision='fp16' \\\n"
        f"--train_batch_size=1 \\\n"
        f"--gradient_accumulation_steps=1 \\\n"
        f"--gradient_checkpointing \\\n"
        f"--use_8bit_adam \\\n"
        f"--learning_rate=1e-6 \\\n"
        f"--lr_scheduler='linear' \\\n"
        f"--lr_warmup_steps=0 \\\n"
        f"--max_train_steps=300"
    )

    # Command 2: Train UNet
    train_unet_command = (
        f"accelerate launch diffusers/examples/dreambooth/train_dreambooth.py \\\n"
        f"--offset_noise \\\n"
        f"--external_captions \\\n"
        f"--image_captions_filename \\\n"
        f"--train_only_unet \\\n"
        f"--save_starting_step=5000 \\\n"
        f"--save_n_steps=5000 \\\n"
        f"--Session_dir=\"{session_data_dir}\" \\\n"
        f"--pretrained_model_name_or_path=\"{pretrained_model_name_or_path}\" \\\n"
        f"--instance_data_dir=\"{instance_data_dir}\" \\\n"
        f"--output_dir=\"{output_dir}\" \\\n"
        f"--captions_dir=\"{captions_dir}\" \\\n"
        f"--instance_prompt='' \\\n"
        f"--seed=1000 \\\n"
        f"--resolution=512 \\\n"
        f"--mixed_precision='fp16' \\\n"
        f"--train_batch_size=1 \\\n"
        f"--gradient_accumulation_steps=1 \\\n"
        f"--use_8bit_adam \\\n"
        f"--learning_rate=2e-6 \\\n"
        f"--lr_scheduler='linear' \\\n"
        f"--lr_warmup_steps=0 \\\n"
        f"--max_train_steps=60000"
    )

    return train_text_encoder_command, train_unet_command

def main(audio_dir, dreambooth_label):
    # Extract the base name of the input directory
    audio_dir_base_name = os.path.basename(os.path.normpath(audio_dir))
    
    # Define output directories based on the input directory name
    audio_output_dir = os.path.join(os.path.dirname(audio_dir), f"{audio_dir_base_name}-audio")
    spec_output_dir = os.path.join(os.path.dirname(audio_dir), f"{audio_dir_base_name}-spec")
    final_output_dir = os.path.join(os.path.dirname(audio_dir), f"{audio_dir_base_name}-spec-nn")

    # Step 1: Sample clips
    sample_clips_command = (
        f"python -m riffusion.cli sample-clips-batch "
        f"--audio-dir {audio_dir} "
        f"--output-dir {audio_output_dir} "
        f"--num-clips-per-file 30"
    )
    run_command(sample_clips_command)

    # Step 2: Convert audio to images
    audio_to_images_command = (
        f"python -m riffusion.cli audio-to-images-batch "
        f"--audio-dir {audio_output_dir} "
        f"--output-dir {spec_output_dir} "
        f"--mono "
        f"--image-extension 'png'"
    )
    run_command(audio_to_images_command)

    # Step 3: Process for DreamBooth
    process_for_dreambooth_command = (
        f"python process_for_dreambooth.py "
        f"{spec_output_dir} "
        f"{final_output_dir} "
        f"{dreambooth_label}"
    )
    run_command(process_for_dreambooth_command)

    # Generate training commands
    base_output_dir = "/home/materialvision/Documents/db"
    pretrained_model_name_or_path = "/home/materialvision/Documents/db/stable-diffusion-custom"
    train_text_encoder_command, train_unet_command = generate_training_commands(final_output_dir, base_output_dir, pretrained_model_name_or_path,dreambooth_label)

    # Print the training commands
    print("\nGenerated Training Commands:\n")
    print(train_text_encoder_command)
    print("\n")
    print(train_unet_command)

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python script_name.py <audio-dir> <dreambooth-label>")
        sys.exit(1)

    audio_dir = sys.argv[1]
    dreambooth_label = sys.argv[2]
    main(audio_dir, dreambooth_label)

#python dataset_training_pipeline.py /Users/espensommereide/Dropbox/Projects/paviljong/riffusion_tests/pasvik-riff-dataset-denoise pasvik