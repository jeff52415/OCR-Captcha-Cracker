import random
import string

from pathlib import Path

import pytest
import requests

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


@pytest.fixture(scope="session")
def normal_weight():
    output_path = "weight/normal.pth"
    if not Path(output_path).is_file():
        file_id = "1ssHy57gFbH96PdP6FzTyveHc4Vpta5AA"
        url = f"https://drive.google.com/uc?id={file_id}"

        resp = requests.get(url=url)

        Path("weight").mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(resp.content)

    return output_path
