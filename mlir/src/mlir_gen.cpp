#include <iostream>
#include <vector>
#include <string>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Builders.h>

#include "parser.h"

// Вспомогательная функция для удаления лишних пробелов (из твоего кода)
std::string remove_extra_spaces(const std::string& str) {
    std::string result;
    bool prev_space = false;
    for (char c : str) {
        if (c == ' ') {
            if (!prev_space) result += c;
            prev_space = true;
        } else {
            result += c;
            prev_space = false;
        }
    }
    return result;
}

// Функция для преобразования типа данных ONNX в MLIR тип
mlir::Type getMLIRTypeFromONNX(int onnxDataType, mlir::OpBuilder& builder) {
    // В ONNX типы данных имеют числовые коды:
    // 1 = float, 2 = uint8, 3 = int8, 4 = uint16, 5 = int16, 
    // 6 = int32, 7 = int64, 10 = float16, 11 = double
    switch (onnxDataType) {
        case 1:  // float
            return builder.getF32Type();
        case 6:  // int32
            return builder.getIntegerType(32);
        case 7:  // int64
            return builder.getIntegerType(64);
        case 11: // double
            return builder.getF64Type();
        default:
            // По умолчанию float
            return builder.getF32Type();
    }
}

// Функция для вывода информации о тензоре
void printTensorInfo(const std::string& name, const std::vector<int64_t>& shape, 
                     mlir::Type elementType, mlir::RankedTensorType tensorType) {
    std::cout << "Tensor '" << name << "': ";
    tensorType.dump();
    std::cout << " (shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << shape[i];
    }
    std::cout << "], type: ";
    elementType.dump();
    std::cout << ")\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx>\n";
        std::cerr << "Example: " << argv[0] << " ../tests/simple_matmul.onnx\n";
        return 1;
    }

    try {
        std::cout << "=== Loading ONNX model: " << argv[1] << " ===\n\n";
        
        // 1. Парсим ONNX модель (используем твой парсер)
        ONNXParser parser(argv[1]);
        Graph graph = parser.parse();
        
        // 2. Выводим мета-информацию (как в твоём main)
        std::cout << "=== Parsed Graph Info ===\n";
        std::cout << "IR version: " << graph.getIrVersion() << "\n";
        std::cout << "Producer: " << graph.getProducerName() 
                  << " v" << graph.getProducerVersion() << "\n";
        std::cout << "Graph name: " << graph.getGraphName() << "\n\n";
        
        // 3. Создаём MLIR контекст и builder
        std::cout << "=== Initializing MLIR ===\n";
        mlir::MLIRContext context;
        mlir::OpBuilder builder(&context);
        std::cout << "MLIR context created successfully\n\n";
        
        // 4. Обрабатываем все тензоры из графа
        std::cout << "=== Processing Tensors with MLIR ===\n";
        const auto& tensors = graph.get_initializers();  // веса и параметры
        
        if (tensors.empty()) {
            std::cout << "No initializer tensors found in graph\n";
        } else {
            std::cout << "Found " << tensors.size() << " initializer tensors:\n\n";
            
            for (const auto& [name, tensor] : tensors) {
                // Получаем форму тензора
                std::vector<int64_t> shape;
                for (int dim : tensor.get_dims()) {
                    shape.push_back(static_cast<int64_t>(dim));
                }
                
                // Получаем MLIR тип элемента
                mlir::Type elementType = getMLIRTypeFromONNX(tensor.get_data_type(), builder);
                
                // Создаём MLIR-тип тензора
                auto tensorType = mlir::RankedTensorType::get(
                    llvm::ArrayRef<int64_t>(shape),
                    elementType
                );
                
                // Выводим информацию
                printTensorInfo(name, shape, elementType, tensorType);
            }
        }
        
        // 5. Выводим информацию об узлах (операциях)
        std::cout << "\n=== Processing Operations ===\n";
        const auto& nodes = graph.get_nodes();
        std::cout << "Found " << nodes.size() << " operations:\n\n";
        
        for (const auto& node : nodes) {
            // пропускаем узлы с пустым op_type
            if (node.get_op_type().empty()) continue;
            
            std::cout << "Operation: " << node.get_op_type() << "\n";
            
            // Входы
            std::cout << "  Inputs: ";
            std::string inputs_str;
            for (const auto& in : node.get_inputs()) {
                if (!in.empty()) {
                    inputs_str += in + " ";
                }
            }
            std::cout << remove_extra_spaces(inputs_str) << "\n";
            
            // Выходы
            std::cout << "  Outputs: ";
            std::string outputs_str;
            for (const auto& out : node.get_outputs()) {
                if (!out.empty()) {
                    outputs_str += out + " ";
                }
            }
            std::cout << remove_extra_spaces(outputs_str) << "\n";
            
            // Атрибуты (как в твоём коде)
            for (const auto& [name, vals] : node.get_ints_attrs()) {
                std::cout << "  [" << name << ": ";
                for (auto v : vals) std::cout << v << " ";
                std::cout << "]\n";
            }
            
            for (const auto& [name, val] : node.get_float_attrs()) {
                std::cout << "  [" << name << ": " << val << "]\n";
            }
            
            for (const auto& [name, val] : node.get_string_attrs()) {
                std::cout << "  [" << name << ": " << val << "]\n";
            }
            
            std::cout << "\n";
        }
        
        std::cout << "=== MLIR processing completed successfully! ===\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}