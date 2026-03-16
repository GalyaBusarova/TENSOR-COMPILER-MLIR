# Tensor Compiler MLIR

Проект по тензорным компиляторам: парсер ONNX моделей с генерацией MLIR.

## Возможности
- Парсинг ONNX файлов (Conv, Relu, Gemm, MatMul и др.)
- Построение графа вычислений
- Генерация MLIR типов для тензоров
- Визуализация графа (GraphViz)

## Сборка
```bash
mkdir build && cd build
cmake ..
make