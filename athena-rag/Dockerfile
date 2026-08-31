FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m grpc_tools.protoc -I/app/proto --python_out=gen --grpc_python_out=gen /app/proto/rag.proto || true
EXPOSE 50051
CMD ["python", "server.py"]
