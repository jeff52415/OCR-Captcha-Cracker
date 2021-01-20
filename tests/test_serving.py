import pytest

from flask import Flask

from captchacracker.serving import main


@pytest.fixture()
def app():
    app = Flask("test")
    return app


@pytest.fixture()
def test_client(app):
    return app.test_client()


def test_serving(app, test_client, test_assets, normal_weight):
    app.add_url_rule("/", "serving", main, methods=["POST"])

    data = {
        "image": open(test_assets / "test.png", "rb"),
    }
    resp = test_client.post("/", data=data)
    assert resp.status_code == 200
    assert resp.content_type == "application/json"

    resp = resp.get_json()
    assert isinstance(resp, dict)
    assert resp == {"result": "3CN"}
