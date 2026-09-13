FROM python:3.12-alpine

WORKDIR /app

COPY src/ /app/src/
COPY web/ /app/web/

ENV PORT=8080
EXPOSE 8080

CMD ["python3", "src/server.py"]
