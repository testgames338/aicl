import base64
import io
import requests
from flask import Flask, request, send_file, render_template
from PIL import Image

app = Flask(__name__)

HUGGINGFACE_TOKEN = "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"  # Replace with your Hugging Face API Token

# Serve the HTML page (frontend)
@app.route("/")
def index():
    return render_template("index.html")

def extract_mask_and_image(base64_data):
    """Decode and split masked area (red = mask) from the image"""
    image_data = base64.b64decode(base64_data.split(',')[1])
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    mask = img.copy()

    return img, mask

@app.route("/inpaint", methods=["POST"])
def inpaint():
    data = request.json
    image_b64 = data["image"]

    image, mask = extract_mask_and_image(image_b64)

    # Prepare multipart request for inpainting
    buffer_img = io.BytesIO()
    buffer_mask = io.BytesIO()
    image.save(buffer_img, format="PNG")
    mask.save(buffer_mask, format="PNG")
    buffer_img.seek(0)
    buffer_mask.seek(0)

    # Call Hugging Face's Stable Diffusion 3.5 API for inpainting
    response = requests.post(
        "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large",
        headers={"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"},
        files={
            "image": buffer_img,  # The original image
            "mask_image": buffer_mask  # The mask (red-painted region)
        },
    )

    if response.status_code == 200:
        result = io.BytesIO(response.content)
        return send_file(result, mimetype="image/png")
    else:
        return {"error": "Inpainting failed. Please check your Hugging Face API setup."}, 400

if __name__ == "__main__":
    app.run(debug=True)

