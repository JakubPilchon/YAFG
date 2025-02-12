from flask import Flask, render_template, url_for, request
from typing import Dict
from model_handling import preprocess, generate_image
from model.model import VAE
import torch


app = Flask("YAGG",template_folder='app/templates', static_folder='app/static')

model = VAE()
model.load_state_dict(torch.load("app/saved_model.pt", weights_only=True), strict=False)
model.to("cpu")

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/image", methods=["POST", "GET"])
def send_data():
    if request.method == "POST":
            (means, std) = preprocess(request.form)
            generate_image(model, means, std)

            return render_template("images.html")
    else:
        return render_template("index.html")

if __name__ == "__main__": 
    app.run(debug=True)