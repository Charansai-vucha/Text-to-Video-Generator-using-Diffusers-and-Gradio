import gradio as gr
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import tempfile

# Function to generate video from text prompt
def generate_video(prompt, num_seconds=10):
    try:
        # Load the model
        pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        # Generate video frames
        video_frames = pipe(prompt, num_inference_steps=num_seconds * 25).frames  # Assuming 25 fps
        video_frames = video_frames.squeeze(0)  # Remove batch dimension

        # Export video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            video_path = temp_video_file.name
            export_to_video(video_frames, video_path)

        return video_path
    except Exception as e:
        return f"An error occurred: {e}"

# Gradio interface function
def gradio_interface(prompt, seconds):
    video_path = generate_video(prompt, seconds)
    if "An error occurred" in video_path:
        return gr.Video(None), f"An error occurred: {video_path}"
    return gr.Video(video_path)

# Define the Gradio interface
interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(lines=5, label="Enter your text prompt"),
        gr.Slider(minimum=1, maximum=10, step=1, label="Enter duration in seconds (1-10)")
    ],
    outputs=gr.Video(),
    title="Text-to-Video Generator",
    description="Generate a video based on a text prompt with specified duration.",
    examples=[["A cat playing with a ball.", 2], ["A boy playing with a dog.", 3]],
)

# Launch the Gradio interface
interface.launch()
