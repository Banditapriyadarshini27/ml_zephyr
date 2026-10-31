"""
Sketch2Shape - Gradio Web Application
"""

import gradio as gr


def run_pipeline(sketch, prompt):
    # Placeholder for end-to-end sketch to 3D mesh pipeline
    return None, None


def main():
    with gr.Blocks(title="Sketch2Shape") as demo:
        gr.Markdown("# 🎨 Sketch2Shape: 2D Sketch to 3D Shape")
        gr.Markdown(
            "Transform hand-drawn sketches into 3D meshes using diffusion "
            "and 3D reconstruction models."
        )

        with gr.Row():
            with gr.Column():
                sketch_input = gr.Image(label="Input Sketch", type="pil")
                prompt_input = gr.Textbox(
                    label="Prompt / Description",
                    placeholder="e.g., A stylish modern chair, 3D asset",
                )
                generate_btn = gr.Button("Generate 3D Mesh", variant="primary")

            with gr.Column():
                img_output = gr.Image(label="Generated Image")
                model_output = gr.Model3D(label="3D Mesh Output")

        generate_btn.click(
            fn=run_pipeline,
            inputs=[sketch_input, prompt_input],
            outputs=[img_output, model_output],
        )

    demo.launch()


if __name__ == "__main__":
    main()
