"""
Sketch2Shape AI - Gradio Web Application
TechZephyr 4.0 Hackathon (IIT Bhubaneswar)

Interactive 2D Sketch & Photo to 3D Shape Synthesis with:
- Prominent 2D Photo & Sketch Image Upload (Drop or browse)
- Live Drawing Sketchpad Canvas
- Fast Direct 3D Reconstruction for Photos (preserving full colors & textures)
- Sketch-guided 2D diffusion pipeline for rough line sketches
- Real-time 3D inspection (gr.Model3D) and .glb / .obj exports
"""

import os
import time
import logging
from typing import Optional, Tuple, List
import gradio as gr
from PIL import Image

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("sketch2shape.app")

from pipeline.preprocess import (
    normalize_sketch,
    preprocess_photo,
    is_canvas_empty,
    detect_input_type,
)
from pipeline.generator_2d import Generator2D, STYLE_PRESETS
from pipeline.segmentation import Segmenter
from pipeline.reconstruct_3d import Reconstructor3D

OUTPUT_DIR = os.path.abspath("./outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    DEVICE_NAME = torch.cuda.get_device_name(0) if HAS_CUDA else "CPU Only (Mock / Local Dev Mode)"
except Exception:
    HAS_CUDA = False
    DEVICE_NAME = "CPU Only (Mock / Local Dev Mode)"

generator_2d = None
segmenter = None
reconstructor_3d = None


def get_pipeline():
    global generator_2d, segmenter, reconstructor_3d
    if generator_2d is None:
        generator_2d = Generator2D()
    if segmenter is None:
        segmenter = Segmenter()
    if reconstructor_3d is None:
        reconstructor_3d = Reconstructor3D()
    return generator_2d, segmenter, reconstructor_3d


def run_pipeline(
    input_method: str,
    upload_data,
    sketchpad_data,
    pipeline_mode: str,
    prompt: str,
    style_preset: str,
    num_steps: int = 20,
    guidance_scale: float = 7.5,
    controlnet_scale: float = 1.1,
    seed: int = 42,
    force_mock: bool = False,
    progress=gr.Progress(track_tqdm=True),
):
    """
    End-to-End Execution:
    1. Resolve active input (uploaded photo vs sketchpad).
    2. Empty canvas check.
    3. Route to Photo Direct-to-3D or Sketch-to-Concept-to-3D.
    4. Foreground segmentation.
    5. 3D geometry reconstruction & export.
    """
    start_time = time.time()

    # Step 1: Select appropriate input
    if "Upload" in input_method:
        target_input = upload_data
    else:
        target_input = sketchpad_data

    # Fallback to whichever has content if active one is blank
    if target_input is None or is_canvas_empty(target_input):
        if upload_data is not None and not is_canvas_empty(upload_data):
            target_input = upload_data
        elif sketchpad_data is not None and not is_canvas_empty(sketchpad_data):
            target_input = sketchpad_data

    if target_input is None or is_canvas_empty(target_input):
        raise gr.Error(
            "No image or sketch detected! Please upload a photo/image or draw on the canvas before clicking Generate."
        )

    # Determine whether input is photo or sketch
    detected_kind = detect_input_type(target_input)
    is_direct_photo_mode = (
        "Direct 3D" in pipeline_mode
        or (detected_kind == "photo" and not prompt.strip())
    )

    gen_2d, seg, rec_3d = get_pipeline()
    if force_mock:
        gen_2d.use_mock = True
        rec_3d.use_mock = True

    progress(0.15, desc="Preprocessing input image...")

    if is_direct_photo_mode:
        # PHOTO MODE: Preserve full colors, textures, and details
        progress(0.3, desc="Enhancing 2D photo resolution...")
        norm_image = preprocess_photo(target_input, target_size=(512, 512))
        concept_2d = norm_image

        progress(0.5, desc="Isolating foreground object...")
        rgba_img, white_bg_img = seg.isolate_foreground(concept_2d)

        progress(0.75, desc="Reconstructing 3D textured mesh...")
        glb_path, obj_path, mesh = rec_3d.reconstruct(
            image=white_bg_img,
            output_dir=OUTPUT_DIR,
            resolution=256,
            has_vertex_color=True,
        )
    else:
        # SKETCH MODE: Normalize strokes, run diffusion synthesis, reconstruct 3D
        progress(0.25, desc="Normalizing sketch strokes...")
        norm_sketch = normalize_sketch(target_input, target_size=(512, 512))
        norm_image = norm_sketch

        progress(0.5, desc=f"Synthesizing 2D concept ({style_preset})...")
        concept_2d = gen_2d.generate(
            sketch_image=norm_sketch,
            prompt=prompt,
            style_preset=style_preset,
            num_inference_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            controlnet_conditioning_scale=float(controlnet_scale),
            seed=int(seed) if seed is not None else None,
        )

        progress(0.75, desc="Isolating foreground object...")
        rgba_img, white_bg_img = seg.isolate_foreground(concept_2d)

        progress(0.88, desc="Reconstructing 3D neural mesh...")
        glb_path, obj_path, mesh = rec_3d.reconstruct(
            image=white_bg_img,
            output_dir=OUTPUT_DIR,
            resolution=256,
            has_vertex_color=True,
        )

    progress(1.0, desc="Done!")
    elapsed = time.time() - start_time

    status_msg = (
        f"### Model Generated in {elapsed:.1f}s!\n"
        f"- **Input Mode:** {'2D Photo / Image' if is_direct_photo_mode else 'Hand Sketch'}\n"
        f"- **Geometry:** {len(mesh.vertices):,} vertices | {len(mesh.faces):,} faces\n"
        f"- **Runtime:** {'NVIDIA GPU' if HAS_CUDA and not rec_3d.use_mock else 'Fast CPU Mode'}\n"
        f"- **Files Exported:** `.glb` and `.obj`"
    )

    download_files = [glb_path, obj_path]

    return (
        glb_path,
        download_files,
        norm_image,
        concept_2d,
        white_bg_img,
        status_msg,
    )


CUSTOM_CSS = """
#header-card {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #1e293b, #0f172a);
    color: white;
    border-radius: 14px;
    margin-bottom: 20px;
    border: 1px solid #334155;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}
#header-card h1 {
    color: #38bdf8;
    margin: 0;
    font-size: 2.2rem;
    font-weight: 800;
}
#header-card p {
    color: #94a3b8;
    margin: 8px 0 0 0;
    font-size: 1.05rem;
}
#badge-container {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-top: 12px;
}
.status-pill {
    background: #1e293b;
    color: #7dd3fc;
    border: 1px solid #0284c7;
    padding: 5px 14px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
}
"""


def toggle_input_view(choice):
    if "Upload" in choice:
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)


def build_app() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft(), css=CUSTOM_CSS, title="Sketch2Shape AI") as demo:

        gr.HTML(
            f"""
            <div id="header-card">
                <h1>Sketch2Shape AI</h1>
                <p>TechZephyr 4.0 Hackathon (IIT Bhubaneswar) | Transform 2D Photos & Sketches into 3D Shapes</p>
                <div id="badge-container">
                    <span class="status-pill">Engine: TripoSR + ControlNet</span>
                    <span class="status-pill">Mode: {DEVICE_NAME}</span>
                    <span class="status-pill">Export: .GLB & .OBJ</span>
                </div>
            </div>
            """
        )

        with gr.Row():
            # Left Column: Input Selection & Controls
            with gr.Column(scale=5):
                gr.Markdown("### Step 1: Provide Your 2D Input")

                input_method = gr.Radio(
                    choices=["Upload 2D Photo / Image / Sketch", "Draw on Live Canvas"],
                    value="Upload 2D Photo / Image / Sketch",
                    label="Choose Input Source",
                    interactive=True,
                )

                upload_box = gr.Image(
                    label="Drop or Click to Upload 2D Photo, Image, or Sketch (.png, .jpg, .jpeg, .webp)",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=360,
                    visible=True,
                )

                sketchpad_box = gr.Sketchpad(
                    label="Draw on Live Canvas (Pencil, Brush, Eraser)",
                    type="pil",
                    image_mode="RGBA",
                    height=360,
                    visible=False,
                )

                input_method.change(
                    fn=toggle_input_view,
                    inputs=[input_method],
                    outputs=[upload_box, sketchpad_box],
                )

                gr.Markdown("### Step 2: Choose Pipeline Mode & Options")

                pipeline_mode = gr.Radio(
                    choices=[
                        "Fast Direct 3D (Recommended for Photos & Colored Images)",
                        "Sketch-to-3D Diffusion (Best for Rough Line Sketches)",
                    ],
                    value="Fast Direct 3D (Recommended for Photos & Colored Images)",
                    label="Pipeline Mode",
                )

                prompt_input = gr.Textbox(
                    label="Descriptive Prompt (Optional)",
                    placeholder="e.g., A stylish ceramic coffee mug, cute robot figurine, sneakers",
                    lines=2,
                )

                style_radio = gr.Radio(
                    label="Artistic Style Preset (For Sketch Diffusion)",
                    choices=list(STYLE_PRESETS.keys()),
                    value="Cartoon / 3D Stylized",
                )

                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Row():
                        steps_slider = gr.Slider(
                            label="Diffusion Steps",
                            minimum=10,
                            maximum=30,
                            value=20,
                            step=1,
                        )
                        guidance_slider = gr.Slider(
                            label="Guidance Scale",
                            minimum=3.0,
                            maximum=12.0,
                            value=7.5,
                            step=0.5,
                        )
                    with gr.Row():
                        control_slider = gr.Slider(
                            label="ControlNet Scale",
                            minimum=0.5,
                            maximum=2.0,
                            value=1.1,
                            step=0.1,
                        )
                        seed_input = gr.Number(
                            label="Random Seed (-1 for random)",
                            value=42,
                            precision=0,
                        )
                    force_mock_checkbox = gr.Checkbox(
                        label="Force CPU Fast Mode",
                        value=(not HAS_CUDA),
                    )

                generate_btn = gr.Button("Generate 3D Model Now", variant="primary", size="lg")

            # Right Column: 3D Previewer & Pipeline Inspection
            with gr.Column(scale=6):
                status_box = gr.Markdown("### Ready. Upload an image or draw a sketch, then click Generate.")

                with gr.Tabs():
                    with gr.TabItem("Interactive 3D Model"):
                        model_viewer = gr.Model3D(
                            label="3D Preview (Left-click: Rotate | Right-click: Pan | Scroll: Zoom)",
                            clear_color=[0.12, 0.14, 0.18, 1.0],
                            height=440,
                        )
                        download_files = gr.File(
                            label="Download 3D Mesh Assets (.glb and .obj)",
                            file_count="multiple",
                        )

                    with gr.TabItem("Pipeline Stages"):
                        gr.Markdown("Inspect the intermediate representations extracted by the pipeline:")
                        with gr.Row():
                            preview_norm = gr.Image(label="1. Preprocessed Input", height=220)
                            preview_2d = gr.Image(label="2. Concept Representation", height=220)
                            preview_seg = gr.Image(label="3. Foreground Isolated", height=220)

        generate_btn.click(
            fn=run_pipeline,
            inputs=[
                input_method,
                upload_box,
                sketchpad_box,
                pipeline_mode,
                prompt_input,
                style_radio,
                steps_slider,
                guidance_slider,
                control_slider,
                seed_input,
                force_mock_checkbox,
            ],
            outputs=[
                model_viewer,
                download_files,
                preview_norm,
                preview_2d,
                preview_seg,
                status_box,
            ],
        )

    return demo


if __name__ == "__main__":
    demo_app = build_app()
    # Explicitly enable queue for fast non-blocking progress updates
    demo_app.queue(default_concurrency_limit=2)
    share_flag = os.getenv("GRADIO_SHARE", "").strip().lower() in ("1", "true", "yes")
    demo_app.launch(server_name="0.0.0.0", server_port=7860, share=share_flag)
