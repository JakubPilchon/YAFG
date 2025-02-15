from flask import Flask, render_template, url_for, request
from typing import Dict
from model_handling import preprocess, generate_image, generate_name
from model.model import VAE
import torch

LATENT_DIM = 16
app = Flask("YAFG",template_folder='app/templates', static_folder='app/static')

model = VAE()
model.load_state_dict(torch.load("app/saved_model.pt", weights_only=True, map_location=torch.device("cpu")), strict=False)
model.to("cpu")

@app.route('/')
def index():
    return render_template('index.html', latent_dim = LATENT_DIM)

@app.route("/image", methods=["POST", "GET"])
def send_data():
    if request.method == "POST":
            (means, std) = preprocess(request.form, LATENT_DIM)
            generate_image(model, means, std)

            return render_template("images.html", name = generate_name())
    else:
        return render_template("index.html")

if __name__ == "__main__": 
    app.run(host="0.0.0.0",port=5000, debug=False)