#include <iostream>
#include <fstream>
#include <unordered_set>

#include "parser.h"

// имена возможных атрибутов
const std::unordered_set<std::string> ATTR_NAMES = {
    "pads", "alpha", "beta", "transA", "transB", "auto_pad", 
    "strides", "dilations", "kernel_shape", "group", "allowzero",
    "start", "end", "axes", "steps", "axis",
    "namespace"  
};

// функция для очистки имени от мусора
std::string clean_tensor_name(const std::string& name) 
{
    std::string result;
    
    // только печатные ASCII символы
    for (char c : name) {
        if (c >= 32 && c < 127) 
        {  
            result += c;
        }
    }
    
    // убираем всё после ':' (PyTorch метаданные)
    size_t colon_pos = result.find(':');
    if (colon_pos != std::string::npos) 
    {
        result = result.substr(0, colon_pos);
    }
    
    // пробелы в конце
    while (!result.empty() && (result.back() == ' ' || result.back() == '\t')) 
    {
        result.pop_back();
    }
    
    return result;
}

// проверяет является ли атрибутом
bool is_attr_name(const std::string& name) 
{
    const std::vector<std::string> attr_prefixes = {
        "pads", "alpha", "beta", "transA", "transB", "auto_pad", 
        "strides", "dilations", "kernel_shape", "group", "allowzero",
        "start", "end", "axes", "steps", "axis", "namespace"
    };
    
    for (const auto& prefix : attr_prefixes) {
        if (name == prefix || name.find(prefix + "_") == 0 || name.find(prefix + ":") == 0) {
            return true;
        }
    }
    return false;
}


// убирает двойные пробелы
std::string remove_extra_spaces(const std::string& str) 
{
    std::string result;
    bool last_was_space = false;
    
    for (char c : str) 
    {
        if (c == ' ' || c == '\t') 
        {
            if (!last_was_space) 
            {
                result += ' ';
                last_was_space = true;
            }
        } else {
            result += c;
            last_was_space = false;
        }
    }
    
    // убираем пробел в конце
    while (!result.empty() && result.back() == ' ') 
    {
        result.pop_back();
    }
    
    return result;
}


int main(int argc, char* argv[]) 
{
    if (argc < 2) 
    { 
        std::cerr << "Usage: " << argv[0] << " <model.onnx>\n"; 
        return 1; 
    }

    try 
    {
        std::cout << "=== Loading: " << argv[1] << " ===\n\n";
        
        ONNXParser parser(argv[1]);
        Graph graph = parser.parse();
        
        // мета-информация
        std::cout << "=== Parsed Graph Info ===\n";
        std::cout << "IR version: " << graph.getIrVersion() << "\n";
        std::cout << "Producer: " << graph.getProducerName() 
                  << " v" << graph.getProducerVersion() << "\n";
        std::cout << "Graph name: " << graph.getGraphName() << "\n\n";
        
        std::cout << "=== Nodes ===\n";
        for (const auto& node : graph.get_nodes()) 
        {
            // пропускаем узлов с пустым op_type
            if (node.get_op_type().empty()) continue;
            
            // вывод типа операции
            std::cout << "Op: " << node.get_op_type() << "\n";
            
            // вывод входов 
            std::cout << "  Inputs: ";
            std::string inputs_str;
            for (const auto& in : node.get_inputs()) 
            {
                if (ATTR_NAMES.count(in) == 0 && !in.empty()) 
                {
                    inputs_str += in + " ";
                }
            }
            std::cout << remove_extra_spaces(inputs_str) << "\n";
            
            // Вывод выходов
            std::cout << "  Outputs: ";
            std::string outputs_str;
            for (const auto& out : node.get_outputs()) 
            {
                if (!out.empty()) 
                {
                    outputs_str += out + " ";
                }
            }
            std::cout << remove_extra_spaces(outputs_str) << "\n";
            
            // Int-атрибуты
            for (const auto& [name, vals] : node.get_ints_attrs()) 
            {
                std::cout << "  [" << name << ": ";
                for (auto v : vals) std::cout << v << " ";
                std::cout << "]\n";
            }
            
            // Float-атрибуты
            for (const auto& [name, val] : node.get_float_attrs()) 
            {
                std::cout << "  [" << name << ": " << val << "]\n";
            }
            
            // String-атрибуты
            for (const auto& [name, val] : node.get_string_attrs()) 
            {
                std::cout << "  [" << name << ": " << val << "]\n";
            }
            
            std::cout << "\n";  
        }
        
        std::cout << "Parsing completed successfully!\n";

        graph.export_to_dot("graph.dot");

        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}