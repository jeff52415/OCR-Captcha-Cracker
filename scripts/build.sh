#!/bin/bash

BASE_DIR=$(dirname "$0")
PROJECT_ROOT=$(cd "$BASE_DIR/.."; pwd -P)

docker build $PROJECT_ROOT -t ocr-captcha-cracker
