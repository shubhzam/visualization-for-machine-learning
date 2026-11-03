# requirements.txt is a pip freeze of this image, so builds get the exact same versions
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libcairo2 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /opt/chartbreak/requirements.txt
RUN pip install --no-cache-dir --no-deps -r /opt/chartbreak/requirements.txt
COPY worker.py evaluate.py /opt/chartbreak/
ENV MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl HOME=/tmp
WORKDIR /output
