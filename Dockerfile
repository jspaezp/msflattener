
# Build as python -m build && docker build -t talusbio/msflattener:latest -f Dockerfile .
# use as docker run --rm -it -v ${PWD}:/data/ msflattener bruker --file /data/myfile.d --output /data/myfile.mzML

FROM --platform=linux/amd64 python:3.9-bullseye
LABEL MAINTAINER="J. Sebastian Paez"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN mkdir /data
RUN apt update && apt install -y git procps build-essential gcc python3-dev
RUN adduser worker
RUN chown worker /data
USER worker
WORKDIR /data
WORKDIR /home/worker

COPY --chown=worker:worker dist/*.whl /home/worker
RUN python3 -m pip install --disable-pip-version-check --no-cache-dir --user /home/worker/*.whl
RUN chmod 777 /home/worker/.local/lib/python3.9/site-packages/alphatims/ext/timsdata.so

RUN python3 -m pip cache purge
ENV PATH="/home/worker/.local/bin:${PATH}"
