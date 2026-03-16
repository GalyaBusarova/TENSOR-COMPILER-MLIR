#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Builders.h>
#include <iostream>
#include <vector>

// Вспомогательная функция для печати формы тензора
void printTensorInfo(mlir::RankedTensorType type, const std::string& name) {
    std::cout << name << ": ";
    type.dump();  // печатает сам тип
    
    // Получаем форму тензора
    llvm::ArrayRef<int64_t> shape = type.getShape();
    std::cout << " (форма: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << shape[i];
    }
    std::cout << "], элементов: " << type.getNumElements() << ")\n";
}

int main() {
    // 1. Создаём контекст
    mlir::MLIRContext context;
    
    // 2. Создаём builder для удобного создания типов
    mlir::OpBuilder builder(&context);
    
    // 3. Создаём базовые типы
    auto f32Type = builder.getF32Type();
    auto i32Type = builder.getIntegerType(32);
    
    // 4. Создаём тензоры разных форм
    std::vector<std::pair<std::vector<int64_t>, mlir::Type>> tensors = {
        {{2, 3}, f32Type},
        {{4, 128, 256}, f32Type},
        {{1, 1, 3, 3}, i32Type},
        {{10}, f32Type},  // одномерный тензор (вектор)
        {{}, i32Type}      // скаляр (тензор нулевой размерности)
    };
    
    // 5. Создаём и выводим информацию о каждом тензоре
    int counter = 1;
    for (const auto& [shape, elementType] : tensors) {
        auto tensorType = mlir::RankedTensorType::get(
            llvm::ArrayRef<int64_t>(shape),
            elementType
        );
        
        printTensorInfo(tensorType, "Тензор " + std::to_string(counter++));
    }
    
    return 0;
}