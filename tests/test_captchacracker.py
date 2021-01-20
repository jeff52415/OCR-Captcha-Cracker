from pathlib import Path

from captchacracker.model import CaptchaCracker


def test_inference(random_str, img, normal_weight):
    model = CaptchaCracker(weight_path=normal_weight, backbone="normal")
    output = model.process(img)

    assert isinstance(output, str)


def test_train():
    save_path = "lighter_backbone_pretrained/crnn_ctc_model.pth"
    model = CaptchaCracker()

    model.fit(
        batch_size=16, max_iterations=10, iteration_per_epoch=20, save_path=save_path,
    )

    assert Path(save_path).is_file()
