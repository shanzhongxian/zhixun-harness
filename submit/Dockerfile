# 参赛镜像：只装 SDK + 推理/训练代码，不含任何模型权重、不含密钥。
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    OMP_NUM_THREADS=4

WORKDIR /opt/build
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY contest_harness /app/contest_harness
COPY app /app
COPY THIRD_PARTY.md /app/THIRD_PARTY.md

# 预建可写工作区：平台会挂载 /workspace，但未挂载或挂载权限不对时也要能跑
RUN mkdir -p /workspace && chown -R 65534:65534 /workspace

WORKDIR /app
USER 65534:65534

ENTRYPOINT ["python3", "/app/harness.py"]
