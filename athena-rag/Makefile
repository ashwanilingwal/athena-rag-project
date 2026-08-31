.PHONY: proto test serve fmt clean
PROTO_DIR := ../proto

proto:                     ## regenerate stubs — rerun after every rag.proto edit
	python -m grpc_tools.protoc -I$(PROTO_DIR) --python_out=gen --pyi_out=gen \
	  --grpc_python_out=gen $(PROTO_DIR)/rag.proto
	sed -i 's/^import rag_pb2 as rag__pb2$$/from . import rag_pb2 as rag__pb2/' gen/rag_pb2_grpc.py

test:                      ## offline: no model, no network, needs postgres only
	EMBED_PROVIDER=hash python -m pytest tests -q

serve:
	python server.py

clean:
	rm -rf gen/rag_pb2*.py gen/rag_pb2.pyi .pytest_cache **/__pycache__
