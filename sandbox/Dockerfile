FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libcairo2 ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir matplotlib==3.9.4 pandas==2.2.3 numpy==1.26.4 \
    'vis-evaluator @ https://github.com/microsoft/VisEval/archive/55162fac5765d8873180d962cce7e20e75b2c092.zip'
COPY worker.py evaluate.py /opt/chartbreak/
ENV MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl HOME=/tmp
WORKDIR /output
