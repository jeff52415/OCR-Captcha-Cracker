from tempfile import NamedTemporaryFile

import numpy as np

from flask import Flask, request

from captchacracker.model import CaptchaCracker


def main():

    # Create temp file
    tmp_file = NamedTemporaryFile(delete=False)

    # Receive uploaded file and save to temp file
    img = request.files.get("image")
    img.save(tmp_file)

    # close temp file
    tmp_file.close()

    # Load model and inference
    model = CaptchaCracker(weight_path="weight/normal.pth", backbone="normal")
    output = model.process(tmp_file.name)


    resp = {"result": output}

    return resp


if __name__ == "__main__":
    app = Flask("serving")
    app.add_url_rule("/", "serving", main, methods=["POST"])
    app.run(host="0.0.0.0", debug=True)
