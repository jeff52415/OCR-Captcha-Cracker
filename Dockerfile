FROM python:3.8
LABEL maintainer="JacobChen <jacob.chen@cinnamon.is>"

WORKDIR /app
COPY setup.py .
RUN pip install --no-cache-dir .["serving"]

COPY captchacracker ./captchacracker/

COPY captchacracker/serving/serving.py .
COPY weight/ ./weight/

ENTRYPOINT [ "python" ]
CMD [ "serving.py" ]