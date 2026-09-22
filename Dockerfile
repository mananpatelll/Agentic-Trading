FROM python:3.13.6

WORKDIR /app 

ENV PYTHONUNBUFFERED=1 
ENV PYTHONDONTWRITEBYTECODE=1 

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 
COPY . . 

RUN --mount=type=secret,id=api_env,target=/app/.env,required=true python src/refresh_universe.py && python src/scanner.py

CMD ["python", "src/main.py"]