FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jdk-headless curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://archive.apache.org/dist/spark/spark-3.5.3/spark-3.5.3-bin-hadoop3.tgz | tar -xz -C /opt

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

ENV SPARK_HOME=/opt/spark-3.5.3-bin-hadoop3
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH="$SPARK_HOME/bin:$PATH"

WORKDIR /app
EXPOSE 8888 4040
