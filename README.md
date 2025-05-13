
# OCR-Captcha-Cracker

A comprehensive tutorial for building an OCR module based on CRNN + CTC structure using the PyTorch framework.

---

## Install

Please clone this repository and install dependencies by running the following commands:

### For OSX / Linux

### For Windows

To install other development dependencies, use:

---

## Information

| Module      | Description     |
|-------------|-----------------|
| CaptchaCracker | [Pretrained Model](https://drive.google.com/drive/folders/1S609zIzcB2mkhvG9Ai6Wai-nlpA1x2Zt?usp=sharing) |

---

## Inference

---

## Train

To train the model, please use the following command:

This command will run a [Flask](https://flask.palletsprojects.com/en/1.1.x/) server to serve your model. After the server is ready, you can:

- Use `curl` command to send POST requests with image data for inference.
- Use Docker to deploy the model and simplify the process.

Please make sure you have Docker installed and running on your machine. Visit [Docker Docs](https://docs.docker.com/engine/install/) for installation instructions.

---

Thank you for using OCR-Captcha-Cracker!
