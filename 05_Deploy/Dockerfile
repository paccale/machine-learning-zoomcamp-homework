FROM agrigorev/zoomcamp-model:2025

COPY pyproject.toml ./

RUN pip install uv

RUN uv pip install --system --no-cache fastapi gunicorn uvicorn scikit-learn==1.6.1

COPY evaluate_fastApi.py ./
COPY pipeline_v1.bin ./

EXPOSE 8000

ENTRYPOINT ["uvicorn", "evaluate_fastApi:app", "--host", "0.0.0.0", "--port", "8000"]

