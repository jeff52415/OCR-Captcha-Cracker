#!/bin/bash

BASE_DIR=$(dirname "$0")
PROJECT_ROOT=$(cd "$BASE_DIR/.."; pwd -P)

docker run --rm -p 5000:5000 ocr-captcha-cracker
