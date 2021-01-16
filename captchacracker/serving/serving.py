from flask import Flask, request
from captchacracker.model import CaptchaCracker
import numpy as np
from tempfile import NamedTemporaryFile

def main():

    # Create temp file
    tmp_file = NamedTemporaryFile()

    # Receive uploaded file and save to temp file
    img = request.files.get("image")
    img.save(tmp_file)

    # Load model and inference
    model = CaptchaCracker(weight_path="weight/normal.pth", backbone="normal")
    output = model.process(tmp_file.name)

    # close temp file
    tmp_file.close()

    resp = {
        "result": output
    }

    return resp

if __name__ == "__main__":
    app = Flask("serving")
    app.add_url_rule("/", "serving", main, methods=["POST"])
    app.run(host="0.0.0.0", debug=True)
