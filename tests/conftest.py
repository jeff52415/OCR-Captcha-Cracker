import random
import string

from pathlib import Path

import pytest

from captcha.image import ImageCaptcha


characters = string.digits + string.ascii_uppercase
width, height, n_len, n_class = 128, 64, 4, len(characters)


@pytest.fixture()
def random_str():
    random_str = "".join([random.choice(characters) for j in range(n_len)])
    return random_str


@pytest.fixture()
def img(random_str):
    # random generate captcha
    generator = ImageCaptcha(width=width, height=height)
    img = generator.generate_image(random_str)

    return img


@pytest.fixture()
def test_assets():
    test_assets = Path(__file__).resolve().parent.joinpath("assets")
    return test_assets
