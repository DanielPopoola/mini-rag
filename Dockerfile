FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080 8501

COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
