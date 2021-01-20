# OCR-Captcha-Cracker

A Wrap up tutorial for building OCR module based on CRNN+CTC structure and PyTorch framework

--------------
</div>

## Install

Please clone this repo and install by using following command:

```shell
git clone https://github.com/jeff52415/OCR-Captcha-Cracker.git
cd OCR-Captcha-Cracker

## OSX / Linux
pip install -e .["torch"]

## Windows
pip install -e .
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```


To install other development dependencies, you need to use this command:

```shell
pip install -e .["dev"]
```

If you are using Windows, please install with following command:

```shell
pip install -e .["dev"] -f https://download.pytorch.org/whl/torch_stable.html
```

## Information



| Module 	  | pretrained |
|:-----------:|:-----------:|
| CaptchaCracker 		  | [Pretrained](https://drive.google.com/drive/folders/1S609zIzcB2mkhvG9Ai6Wai-nlpA1x2Zt?usp=sharing) |


General Functions
- `.fit`: training function
- `.process` : inference function
- `.save` : Save current checkpoint to disk


--------------
</div>

## Inference



```python
import string
import random
from captcha.image import ImageCaptcha
from captchacracker.model import CaptchaCracker
model = CaptchaCracker(weight_path='weights/light.pth', backbone='light')


characters = string.digits + string.ascii_uppercase
width, height, n_len, n_class = 128, 64, 4, len(characters)
generator = ImageCaptcha(width=width, height=height)
random_str = ''.join([random.choice(characters) for j in range(n_len)])
img = generator.generate_image(random_str)

output = model.process(img)
```

--------------
</div>

## Train



```python
from captchacracker.model import CaptchaCracker

model = CaptchaCracker(weight_path='weights/light.pth')

model.fit(batch_size=32, max_iterations=1000000, iteration_per_epoch=2000, save_path='lighter_backbone_pretrained/crnn_ctc_model.pth')
```

--------------

## Serving

To serving the model, please use this command:

```shell
python captchacracker/serving/serving.py
```

This command will run a [Flask](https://flask.palletsprojects.com/en/1.1.x/) server to serving your model.
After the server is ready, you can use [`curl`](https://curl.se/) command to use this model:

```shell
curl -X POST 'http://localhost:5000' \
  -F 'image=@tests/assets/test.png'
```

The response should be:

```
{
  "result": "3CN"
}
```
## Docker

To deploy this model, we suggest to use [Docker](https://www.docker.com/) to help you simplify the procedure.

You need to make sure you already setup the Docker on your machince. Please check following link to install the Docker:

* [Linux](https://docs.docker.com/engine/install/)
* [Mac](https://docs.docker.com/docker-for-mac/install/)
* [Windows](https://docs.docker.com/docker-for-windows/install/)

After the docker is ready, you can use the [build](scripts/build.sh) script to build the image with your model.

Then, after your image is ready, you can use the [run](scripts/run.sh) script to run the container and serving your model on any machine.
