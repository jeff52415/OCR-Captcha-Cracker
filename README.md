# OCR-Captcha-Cracker

A Wrap up tutorial for building OCR module based on CRNN+CTC structure and PyTorch framework

--------------
</div>

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
