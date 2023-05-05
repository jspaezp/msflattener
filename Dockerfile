
# Build as python -m build && docker build -t jspaezp/msflattener:latest -f Dockerfile .

FROM --platform=linux/amd64 python:slim
RUN apt update && apt install procps -y
LABEL MAINTAINER="J. Sebastian Paez"
COPY dist/*.whl /tmp/
WORKDIR /app
RUN WHEEL=$(ls /tmp/*.whl) && pip install ${WHEEL}