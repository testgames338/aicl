# app.py
import base64
import io
import os
import requests
from flask import Flask, request, send_file
from PIL import Image
import numpy as np

app = Flask(__name__)

HUGGINGFACE_TOKEN = "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"

def extract_mask_and_image(base64_data):
    """Decode and split masked area (red = mask) from the image"""
    image_data = base64.b64decode(base64_data.split(',')[1])
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    np_img = np.array(img)

    mask = np.zeros((img.height, img.width), dtype=np.uint8)
    red_pixels = (np_img[:, :, 0] > 200) & (np_img[:, :, 1] < 100) & (np_img[:, :, 2] < 100)
    mask[red_pixels] = 255

    return img, Image.fromarray(mask)

@app.route("/inpaint", methods=["POST"])
def inpaint():
    data = request.json
    image_b64 = data["image"]

    image, mask = extract_mask_and_image(image_b64)

    # Prepare multipart request
    buffer_img = io.BytesIO()
    buffer_mask = io.BytesIO()
    image.save(buffer_img, format="PNG")
    mask.save(buffer_mask, format="PNG")
    buffer_img.seek(0)
    buffer_mask.seek(0)

    response = requests.post(
        "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-inpainting",
        headers={"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"},
        files={"image": buffer_img, "mask_image": buffer_mask},
    )

    result = io.BytesIO(response.content)
    return send_file(result, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)
