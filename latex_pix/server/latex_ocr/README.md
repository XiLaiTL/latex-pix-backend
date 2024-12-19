
Server: https://github.com/lucasvanmol/latex-ocr-server
Client: https://github.com/lucasvanmol/obsidian-latex-ocr

```shell
python -m grpc_tools.protoc -I./protos --python_out=. --pyi_out=. --grpc_python_out=. ./protos/latex_ocr.proto
```
