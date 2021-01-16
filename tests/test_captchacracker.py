from captchacracker.model import CaptchaCracker


def test_inference(random_str, img):
    model = CaptchaCracker(weight_path="weight/normal.pth", backbone="normal")
    output = model.process(img)

    assert isinstance(output, str)
    assert output == random_str
