"""
Sketch2Shape AI - Gradio Web Application
TechZephyr 4.0 Hackathon (IIT Bhubaneswar)

Interactive 2D Sketch to 3D Shape Synthesis with:
- Live Sketchpad & Image Upload
- Prompt injection & Artistic Style presets
- ControlNet + SD1.5 2D synthesis
- rembg U2-Net foreground isolation
- Stability AI TripoSR / procedural 3D reconstruction
- Real-time 3D inspection (gr.Model3D) and .glb / .obj exports
"""

import os
import time
import logging
from typing import Optional, Tuple, List
import gradio as gr
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("sketch2shape.app")

# Import pipeline modules
from pipeline.preprocess import normalize_sketch, is_canvas_empty
from pipeline.generator_2d import Generator2D, STYLE_PRESETS
from pipeline.segmentation import Segmenter
from pipeline.reconstruct_3d import Reconstructor3D

# Ensure output directory exists
OUTPUT_DIR = os.path.abspath("./outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Detect runtime hardware mode
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    DEVICE_NAME = torch.cuda.get_device_name(0) if HAS_CUDA else "CPU Only (Mock Development Mode)"
except Exception:
    HAS_CUDA = False
    DEVICE_NAME = "CPU Only (Mock Development Mode)"

# Initialize pipeline singletons (lazy or eager)
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
    sketchpad_input,
    upload_input,
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
    1. Validation & Input Selection
    2. Sketch Preprocessing & Dark-Mode Inversion (512x512)
    3. ControlNet + SD1.5 2D Concept Synthesis
    4. rembg Foreground Segmentation
    5. TripoSR Neural 3D Reconstruction & Export (.glb / .obj)
    """
    start_time = time.time()

    # Determine input source
    input_data = None
    if sketchpad_input is not None and not is_canvas_empty(sketchpad_input):
        input_data = sketchpad_input
    elif upload_input is not None and not is_canvas_empty(upload_input):
        input_data = upload_input

    # Step 1: Input Validation
    if input_data is None or is_canvas_empty(input_data):
        raise gr.Error("The sketch canvas is empty! Please draw a sketch on the canvas or upload an image first.")

    progress(0.1, desc="Preprocessing & Normalizing Sketch...")
    try:
        norm_sketch = normalize_sketch(input_data, target_size=(512, 512))
    except Exception as e:
        logger.error("Preprocessing failed: %s", str(e), exc_info=True)
        raise gr.Error(f"Failed to preprocess sketch: {str(e)}")

    # Initialize models
    gen_2d, seg, rec_3d = get_pipeline()
    if force_mock:
        gen_2d.use_mock = True
        rec_3d.use_mock = True

    # Step 2: 2D Concept Synthesis
    progress(0.35, desc=f"Synthesizing 2D Concept ({style_preset})...")
    try:
        concept_2d = gen_2d.generate(
            sketch_image=norm_sketch,
            prompt=prompt,
            style_preset=style_preset,
            num_inference_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            controlnet_conditioning_scale=float(controlnet_scale),
            seed=int(seed) if seed is not None else None,
        )
    except Exception as e:
        logger.error("2D Generation failed: %s", str(e), exc_info=True)
        raise gr.Error(f"2D Concept Synthesis failed: {str(e)}")

    # Step 3: Foreground Isolation
    progress(0.65, desc="Isolating Foreground Object (rembg)...")
    try:
        rgba_img, white_bg_img = seg.isolate_foreground(concept_2d)
    except Exception as e:
        logger.error("Segmentation failed: %s", str(e), exc_info=True)
        white_bg_img = concept_2d
        rgba_img = concept_2d.convert("RGBA")

    # Step 4: 3D Reconstruction & Export
    progress(0.85, desc="Extracting 3D Geometry (TripoSR)...")
    try:
        glb_path, obj_path, mesh = rec_3d.reconstruct(
            image=white_bg_img,
            output_dir=OUTPUT_DIR,
            resolution=256,
            has_vertex_color=True,
        )
    except Exception as e:
        logger.error("3D Reconstruction failed: %s", str(e), exc_info=True)
        raise gr.Error(f"3D Mesh Generation failed: {str(e)}")

    progress(1.0, desc="Complete!")
    elapsed = time.time() - start_time

    status_msg = (
        f"### ✅ Generation Complete in {elapsed:.1f}s\n"
        f"- **Vertices:** {len(mesh.vertices):,} | **Faces:** {len(mesh.faces):,}\n"
        f"- **Runtime Mode:** {'⚡ NVIDIA GPU' if HAS_CUDA and not gen_2d.use_mock else '💻 CPU Mock / Dev Mode'}\n"
        f"- **Formats Exported:** `.glb` (interactive) and `.obj` (materials included)"
    )

    download_files = [glb_path, obj_path]

    return (
        glb_path,          # 3D Model viewer
        download_files,    # Download list
        norm_sketch,       # Intermediate: normalized sketch
        concept_2d,        # Intermediate: generated concept
        white_bg_img,      # Intermediate: segmented foreground
        status_msg,        # Status markdown
    )


# Custom CSS for styling
CUSTOM_CSS = """
#header-card {
    text-align: center;
    padding: 18px;
    background: linear-gradient(135deg, #1f2937, #111827);
    color: white;
    border-radius: 12px;
    margin-bottom: 15px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
}
#header-card h1 {
    color: #60a5fa;
    margin: 0;
    font-size: 2.2rem;
    font-weight: 700;
}
#header-card p {
    color: #d1d5db;
    margin: 6px 0 0 0;
    font-size: 1.05rem;
}
#badge-container {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-top: 10px;
}
.status-pill {
    background: #374151;
    color: #93c5fd;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft(), css=CUSTOM_CSS, title="Sketch2Shape AI") as demo:

        # Header Banner
        gr.HTML(
            f"""
            <div id="header-card">
                <h1>🎨 Sketch2Shape AI</h1>
                <p>TechZephyr 4.0 Hackathon Submission (IIT Bhubaneswar) | Single-View Sketch to Textured 3D Mesh</p>
                <div id="badge-container">
                    <span class="status-pill">⚡ Engine: ControlNet + TripoSR</span>
                    <span class="status-pill">🖥️ Hardware: {DEVICE_NAME}</span>
                    <span class="status-pill">📦 Export: .GLB & .OBJ</span>
                </div>
            </div>
            """
        )

        with gr.Row():
            # Left Column: Inputs & Controls
            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.TabItem("✏️ Draw Sketch"):
                        sketchpad = gr.Sketchpad(
                            label="Sketchpad Canvas (Draw black or colored strokes)",
                            type="pil",
                            image_mode="RGBA",
                            height=360,
                        )
                    with gr.TabItem("📁 Upload Sketch Image"):
                        upload_img = gr.Image(
                            label="Upload Sketch / Doodle (.png, .jpg)",
                            type="pil",
                            sources=["upload", "clipboard"],
                            height=360,
                        )

                prompt_input = gr.Textbox(
                    label="Descriptive Prompt (Optional)",
                    placeholder="e.g., A cute sci-fi robot with round eyes, metallic finish",
                    lines=2,
                )

                style_radio = gr.Radio(
                    label="Artistic Style Preset",
                    choices=list(STYLE_PRESETS.keys()),
                    value="Cartoon / 3D Stylized",
                )

                with gr.Accordion("⚙️ Advanced Pipeline Parameters", open=False):
                    with gr.Row():
                        steps_slider = gr.Slider(
                            label="Diffusion Steps",
                            minimum=10,
                            maximum=30,
                            value=20,
                            step=1,
                        )
                        guidance_slider = gr.Slider(
                            label="Guidance Scale (CFG)",
                            minimum=3.0,
                            maximum=12.0,
                            value=7.5,
                            step=0.5,
                        )
                    with gr.Row():
                        control_slider = gr.Slider(
                            label="ControlNet Conditioning Scale",
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
                        label="Force CPU Mock Mode (Instant local testing without GPU weights)",
                        value=(not HAS_CUDA),
                    )

                generate_btn = gr.Button("🚀 Generate 3D Shape", variant="primary", size="lg")

            # Right Column: Output Visualizer & Intermediate Tabs
            with gr.Column(scale=6):
                status_box = gr.Markdown("### ℹ️ Ready for input. Draw a sketch and click Generate.")

                with gr.Tabs():
                    with gr.TabItem("🧊 Interactive 3D Model"):
                        model_viewer = gr.Model3D(
                            label="Interactive 3D Preview (Left-click: Rotate | Right-click: Pan | Scroll: Zoom)",
                            clear_color=[0.12, 0.14, 0.18, 1.0],
                            height=420,
                        )
                        download_files = gr.File(
                            label="📥 Download 3D Meshes (.glb & .obj)",
                            file_count="multiple",
                        )

                    with gr.TabItem("🔍 Pipeline Inspection"):
                        gr.Markdown("Inspect intermediate transformations at each pipeline stage:")
                        with gr.Row():
                            preview_norm = gr.Image(label="1. Normalized Sketch (512x512)", height=220)
                            preview_2d = gr.Image(label="2. Synthesized Concept (SD 1.5)", height=220)
                            preview_seg = gr.Image(label="3. Segmented Foreground (rembg)", height=220)

        # Wire click event
        generate_btn.click(
            fn=run_pipeline,
            inputs=[
                sketchpad,
                upload_img,
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

        # Example accordion
        with gr.Accordion("💡 Tips & Examples", open=False):
            gr.Markdown(
                """
                - **Drawing on Canvas:** Use single-stroke outlines or filled silhouettes. Dark mode strokes are automatically inverted!
                - **Style Presets:**
                  - **Cartoon / 3D Stylized:** Great for characters, toys, and figurines.
                  - **Hand-Painted / Texture Art:** Low-poly gaming assets with warm clay shading.
                  - **Clean Minimalist Product:** Industrial design, hard-surface objects, electronics.
                - **Cloud Execution:** For full NVIDIA T4 GPU acceleration, launch via `colab_runner.ipynb`.
                """
            )

    return demo


if __name__ == "__main__":
    app = build_app()
    # share=True can be enabled for tunneling or via environment variable
    share_flag = os.getenv("GRADIO_SHARE", "").strip().lower() in ("1", "true", "yes")
    app.launch(server_name="0.0.0.0", server_port=7860, share=share_flag)
