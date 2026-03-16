cmake_minimum_required(VERSION 3.10)

# Путь к собранному MLIR
set(MLIR_DIR "/Users/busarovagalina/TENSOR-COMPILER-MLIR/llvm-project/build/lib/cmake/mlir")

message(STATUS "MLIR found: ${MLIR_DIR}")
message(STATUS "MLIR include dirs: ${MLIR_INCLUDE_DIRS}")
message(STATUS "MLIR libraries: ${MLIR_LIBS}")

# Подключаем директории MLIR
# Подключаем все необходимые директории MLIR и LLVM
# Подключаем все необходимые директории MLIR и LLVM
# Подключаем все необходимые директории MLIR и LLVM
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
    /Users/busarovagalina/TENSOR-COMPILER-MLIR/llvm-project/llvm/include
    /Users/busarovagalina/TENSOR-COMPILER-MLIR/llvm-project/mlir/include
    /Users/busarovagalina/TENSOR-COMPILER-MLIR/llvm-project/build/include
    /Users/busarovagalina/TENSOR-COMPILER-MLIR/llvm-project/build/tools/mlir/include   # ЭТО НОВОЕ!
)
link_directories(${MLIR_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})
include_directories(/Users/busarovagalina/TENSOR-COMPILER-MLIR/llvm-project/mlir/include)

project(OnnxParser VERSION 1.0 LANGUAGES CXX)

# Стандарт C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Пути к исходникам
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

# Заголовочные файлы
set(HEADERS
    ${INCLUDE_DIR}/bin_reader.h
    ${INCLUDE_DIR}/parser.h
)

# Исходные файлы
set(SOURCES
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/parser.cpp
)

# Создаём исполняемый файл
add_executable(parser ${SOURCES} ${HEADERS})

# Подключаем директорию с заголовками
target_include_directories(parser PRIVATE ${INCLUDE_DIR})

# === Тесты (опционально) ===
enable_testing()

# Тест 1: simple_matmul.onnx
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/simple_matmul.onnx)
    add_test(NAME TestSimpleMatmul 
             COMMAND parser ${CMAKE_SOURCE_DIR}/tests/simple_matmul.onnx)
endif()

# Тест 2: complex_net.onnx
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/complex_net.onnx)
    add_test(NAME TestComplexNet 
             COMMAND parser ${CMAKE_SOURCE_DIR}/tests/complex_net.onnx)
endif()

# Тест 3: custom_net.onnx
if(EXISTS ${CMAKE_SOURCE_DIR}/tests/custom_net.onnx)
    add_test(NAME TestCustomNet 
             COMMAND parser ${CMAKE_SOURCE_DIR}/tests/custom_net.onnx)
endif()

# Наша тестовая программа
link_directories(/Users/busarovagalina/TENSOR-COMPILER-MLIR/llvm-project/build/lib)

add_executable(test_mlir mlir/src/test_mlir.cpp)

add_executable(mlir_tensor_test
    mlir/src/mlir_gen.cpp
    src/parser.cpp          
)

target_link_libraries(mlir_tensor_test
    MLIRIR
    MLIRSupport
    LLVMSupport
    LLVMDemangle
)

target_link_libraries(test_mlir
    MLIRIR
    MLIRSupport
    LLVMSupport
    LLVMDemangle
    LLVMCore
    LLVMBinaryFormat
    LLVMMC
    LLVMRemarks
    LLVMBitstreamReader
    LLVMBitWriter
    LLVMOption
    LLVMTableGen
    LLVMFrontendOpenMP
    LLVMFrontendOffloading
    LLVMFrontendOpenACC
    LLVMExtensions
    LLVMTransformUtils
    LLVMAnalysis
    LLVMProfileData
    LLVMDebugInfoCodeView
    LLVMDebugInfoDWARF
    LLVMDebugInfoGSYM
    LLVMDebugInfoLogicalView
    LLVMDebugInfoMSF
    LLVMDebugInfoBTF
    LLVMCoroutines
    LLVMipo
    LLVMVectorize
    LLVMLinker
    LLVMInstrumentation
    LLVMFrontendHLSL
    LLVMFrontendDriver
    LLVMTargetParser
    LLVMCodeGenTypes
    LLVMCFGuard
    LLVMCoverage
    LLVMTarget
    LLVMCGData
    LLVMTextAPI
    LLVMObjCARCOpts
    LLVMWindowsDriver
    LLVMXRay
    LLVMSymbolize
    LLVMDebugInfoPDB
    LLVMOrcJIT
    LLVMOrcShared
    LLVMOrcTargetProcess
    LLVMExecutionEngine
    LLVMRuntimeDyld
    LLVMObject
    LLVMMCParser
    LLVMIRReader
    LLVMAsmParser
    LLVMAsmPrinter
    LLVMMCDisassembler
    LLVMDebuginfod
    LLVMDlltoolDriver
    LLVMLibDriver
    LLVMInterpreter
    LLVMFuzzMutate
    LLVMLineEditor
    LLVMFileCheck
    LLVMSandboxIR
    LLVMWindowsManifest
    LLVMDemangle
    MLIRIR
    MLIRSupport
)

# Вывод информации
message(STATUS "")
message(STATUS "=== OnnxParser ===")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "")