#include <string>
#include <vector>

#include "parser.h"

// очистка строки от мусора
std::string clean_string(const std::vector<uint8_t>& bytes) 
{
    std::string result;
    
    for (uint8_t b : bytes) {
        // разрешаем только: буквы, цифры, underscore, точка, дефис, слэш, это всё, что может быть в валидном имени тензора ONNX
        if ((b >= 'a' && b <= 'z') || 
            (b >= 'A' && b <= 'Z') || 
            (b >= '0' && b <= '9') ||
            b == '_' || b == '.' || b == '-' || b == '/') {
            result += static_cast<char>(b);
        }

        else 
        {
            break;
        }
    }
    
    return result;
}


Graph ONNXParser::parse()
{
    bool graph_parsed = false;

    while (!reader.check_eof() && !graph_parsed)
    {
        try
        {
            uint8_t cur_byte = reader.watch_cur_byte();
            int wire_type = cur_byte & 0x07;
            int field_number = cur_byte >> 3;

            reader.read_byte();

            switch (field_number)
            {
            case 1: // ir_version
                if (wire_type == 0) // VARINT
                    graph.setIrVersion(reader.read_varint());
                break;

            case 2: // producer_name
                if (wire_type == 2) // LEN
                {
                    uint64_t len = reader.read_varint();
                    auto bytes = reader.read_bytes(len);

                    graph.setProducerName(std::string(bytes.begin(), bytes.end()));
                }
                break;

            case 3: // producer_version
                if (wire_type == 2) // LEN
                {
                    uint64_t len = reader.read_varint();
                    auto bytes = reader.read_bytes(len);

                    graph.setProducerVersion(std::string(bytes.begin(), bytes.end()));
                }
                break;

            case 7: // graph
                if (wire_type == 2)
                {
                    uint64_t graph_size = reader.read_varint();
                    parseGraph(graph_size);
                    graph_parsed = true;  // после графа выход
                }
                break;

            default:
                if (wire_type == 0) reader.read_varint();
                else if (wire_type == 1) reader.read_bytes(8);
                else if (wire_type == 2) { uint64_t l = reader.read_varint(); reader.read_bytes(l); }
                else if (wire_type == 5) reader.read_bytes(4);
                break;
            }
        }
        catch (const std::out_of_range&)
        {
            break;
        }
        catch (const std::runtime_error& e)
        {
            std::cerr << "Parse error: " << e.what() << "\n";
            throw;
        }
    }

    return graph;
}

// вспомогательная функция для парсинга атрибутов
void ONNXParser::parseAttribute(Node& node, uint64_t attr_len)
{
    size_t start_pos = reader.get_cur_pos();
    size_t end_pos = start_pos + attr_len;
    
    std::string attr_name;
    int64_t single_int = 0;
    float single_float = 0.0f;
    std::string string_val;
    std::vector<int64_t> ints_vals;
    bool has_int_value = false;
    
    while (reader.get_cur_pos() < end_pos)
    {
        // проверка: есть ли байт для tag
        if (reader.get_cur_pos() + 1 > end_pos) break;
        
        uint8_t tag = reader.read_byte();
        int field_number = tag >> 3;
        int wire_type = tag & 0x07;
        
        switch (field_number)
        {
        case 1: // name (string)
            {
                uint64_t len = reader.read_varint();
                if (reader.get_cur_pos() + len > end_pos) break;
                auto bytes = reader.read_bytes(len);
                attr_name = clean_string(bytes);
            }
            break;
            
        case 2: // f (float)
            {
                if (reader.get_cur_pos() + 4 > end_pos) break;
                auto bytes = reader.read_bytes(4);
                std::memcpy(&single_float, bytes.data(), 4);
            }
            break;
            
        case 3: // i (int) 
            {
                if (reader.get_cur_pos() < end_pos) 
                {
                    single_int = reader.read_varint();
                    // если вышли за границу — игнорируем значение
                    if (reader.get_cur_pos() > end_pos) 
                    {
                        single_int = 0;
                        has_int_value = false;
                    } else {
                        has_int_value = true;
                    }
                }
            }
            break;
            
        case 6: // s (string) — auto_pad
            {
                uint64_t len = reader.read_varint();
                if (reader.get_cur_pos() + len > end_pos) break;
                auto bytes = reader.read_bytes(len);
                string_val = clean_string(bytes);
            }
            break;
            
        case 7: // floats
            {
                uint64_t len = reader.read_varint();
                if (reader.get_cur_pos() + len > end_pos) break;
                reader.read_bytes(len);
            }
            break;
            
        case 8: // ints (repeated) 
            {
                if (reader.get_cur_pos() < end_pos) 
                {
                    ints_vals.push_back(reader.read_varint());

                    // Если вышли за границу — удаляем
                    if (reader.get_cur_pos() > end_pos) {
                        ints_vals.pop_back();
                    }
                }
            }
            break;
            
        case 20: // type (enum) — просто пропускаем
            {
                if (reader.get_cur_pos() < end_pos) 
                {
                    reader.read_varint();
                }
            }
            break;
            
        default:
            if (wire_type == 0) {
                if (reader.get_cur_pos() < end_pos) reader.read_varint();
            }
            else if (wire_type == 2) { 
                uint64_t l = reader.read_varint(); 
                if (reader.get_cur_pos() + l <= end_pos) reader.read_bytes(l);
            }
            else if (wire_type == 5) {
                if (reader.get_cur_pos() + 4 <= end_pos) reader.read_bytes(4);
            }
            break;
        }
    }
    
    while (reader.get_cur_pos() < end_pos) 
    {
        reader.read_byte();
    }
    
    // Сохраняем атрибуты
    if (!ints_vals.empty() && 
        (attr_name == "strides" || attr_name == "dilations" || 
         attr_name == "pads" || attr_name == "kernel_shape" || attr_name == "allowzero"))
    {
        node.add_ints_attr(attr_name, ints_vals);
    }

    if (has_int_value && single_int != 0 && attr_name == "group")
    {
        node.add_ints_attr(attr_name, {single_int});
    }

    if (single_int != 0 && (attr_name == "group" || attr_name == "transA" || attr_name == "transB"))
    {
        node.add_ints_attr(attr_name, {single_int});
    }
    
    // Для allowzero сохраняем даже 0
    if (attr_name == "allowzero") {
        node.add_ints_attr(attr_name, {single_int});
    }

    if (single_float != 0.0f && (attr_name == "alpha" || attr_name == "beta"))
    {
        node.add_float_attr(attr_name, single_float);
    }

    if (!string_val.empty())
    {
        node.add_string_attr(attr_name, string_val);
    }
}

// вспомогательная функция для парсинга одной ноды
Node ONNXParser::parseNode(uint64_t node_size)
{
    Node result;
    size_t end_pos = reader.get_cur_pos() + node_size;
    
    while (reader.get_cur_pos() < end_pos)
    {
        uint8_t cur_byte = reader.read_byte();
        int wire_type = cur_byte & 0x07;
        int field_number = cur_byte >> 3;
        
        switch (field_number)
        {
        case 1: // input (repeated string)
        {
            uint64_t len = reader.read_varint(); // длина очередной строки
            if (reader.get_cur_pos() + len > end_pos) break;

            auto bytes = reader.read_bytes(len);
            result.add_input(clean_string(bytes));  // добавляем в вектор inputs

            break;
        }
            
        case 2: // output
        {
            uint64_t len = reader.read_varint(); // длина очередной строки
            if (reader.get_cur_pos() + len > end_pos) break;

            auto bytes = reader.read_bytes(len);
            result.add_output(clean_string(bytes)); // добавляем в вектор outputs

            break;
        }
            
        case 3: // name
        {
            uint64_t len = reader.read_varint();
            if (reader.get_cur_pos() + len > end_pos) break;

            auto bytes = reader.read_bytes(len);
            result.set_name(clean_string(bytes));

            break;
        }
            
            
        case 4: // op_type
        {
            uint64_t len = reader.read_varint();
            if (reader.get_cur_pos() + len > end_pos) {
                break;
            }

            auto bytes = reader.read_bytes(len);
            result.set_op_type(clean_string(bytes));

            break;
        }
            
        case 5: // attribute
        {
            uint64_t attr_len = reader.read_varint(); // длина атрибута
            parseAttribute(result, attr_len);

            break;
        }
            
        default: // для неизвестных полей внутри узла
        {
            if (wire_type == 0) {
                reader.read_varint();
                } else if (wire_type == 2) {
                    uint64_t len = reader.read_varint();
                    if (reader.get_cur_pos() + len > end_pos) break;

                    reader.read_bytes(len);
                }
                break;
        } 
        }
    
    }

    size_t current = reader.get_cur_pos();
    if (current < end_pos) {
        reader.read_bytes(end_pos - current);  // дочитываем до конца узла
    }

    return result;
}


// вспомогательная функция для парсинга одного тензора
Tensor ONNXParser::parseTensor(uint64_t tensor_size)
{
    Tensor result;

    size_t end_pos = reader.get_cur_pos() + tensor_size;
    uint8_t cur_byte = reader.watch_cur_byte();

    while (reader.get_cur_pos() < end_pos)
    {
        cur_byte = reader.watch_cur_byte();
        int field_number = cur_byte >> 3;
        reader.read_byte();

        switch(field_number)
        {
            case 1: // dims
            {
                uint64_t dim = reader.read_varint(); 
                result.add_dim(dim);
                break;
            }

            case 2: // data_type
            {
                uint64_t type = reader.read_varint(); 
                result.set_data_type(type);
                break;
            }

            case 3: // name
            {
                uint64_t str_size = reader.read_varint();
                if (reader.get_cur_pos() + str_size > end_pos) break;

                auto bytes = reader.read_bytes(str_size);
                std::string tensor_name = clean_string(bytes);

                result.set_name(tensor_name);
                break;
            }

            case 5: // float_data
            {
                uint64_t len = reader.read_varint();
                if (reader.get_cur_pos() + len > end_pos) break;

                std::vector<uint8_t> bytes = reader.read_bytes(len);
                    
                result.set_raw_data(bytes);  
                break;
            }

            case 9: // raw_data
            {
                uint64_t len = reader.read_varint();
                if (reader.get_cur_pos() + len > end_pos) break;

                result.set_raw_data(reader.read_bytes(len));

                break;
            }
        }
    }

    return result;
}

// вспомогательная функция для строки для DOT
static std::string escape_dot(const std::string& str) 
{
    std::string result;
    for (char c : str) 
    {
        if (c == '"') result += "\\\"";
        else if (c == '\n') result += "\\n";
        else if (c == '\\') result += "\\\\";
        else result += c;
    }
    return result;
}

// Вспомогательная функция - цвет для типа операции
static std::string get_node_color(const std::string& op_type) 
{
    if (op_type == "Conv") return "lightblue";
    if (op_type == "Relu") return "lightgreen";
    if (op_type == "Gemm") return "lightyellow";
    if (op_type == "MatMul") return "lightcyan";
    if (op_type == "Add") return "lavender";
    if (op_type == "Mul") return "peachpuff";
    if (op_type == "Reshape") return "thistle";
    if (op_type == "Concat") return "plum";
    if (op_type == "Shape") return "lightgray";
    return "white";
}

// вспомогательная функция - форма для типа операции
static std::string get_node_shape(const std::string& op_type) 
{
    if (op_type == "Relu") 
    {
        return "ellipse";
    }

    if (op_type == "Reshape" || op_type == "Concat" || op_type == "Shape") 
    {
        return "diamond";
    }

    return "box";
}

// экранирования HTML-спецсимволов
static std::string escape_html(const std::string& str) 
{
    std::string result;

    for (char c : str) 
    {
        if (c == '<') result += "&lt;";
        else if (c == '>') result += "&gt;";
        else if (c == '&') result += "&amp;";
        else if (c == '"') result += "&quot;";
        else result += c;
    }

    return result;
}

void Graph::export_to_dot(const std::string& filename) const 
{
    std::ofstream dot(filename);
    if (!dot) 
    {
        std::cerr << "Не удалось создать файл: " << filename << "\n";
        return;
    }

    // заголовок 
    dot << "digraph ONNX_Graph {\n";
    dot << "    rankdir=TB;\n";                    // сверху вниз
    dot << "    node [fontname=\"Helvetica\", fontsize=10];\n";
    dot << "    edge [fontname=\"Helvetica\", fontsize=9];\n";
    dot << "    label=\"" << escape_dot(graph_name) << "\";\n";
    dot << "    labelloc=\"t\";\n";
    dot << "\n";

    // Легенда 
    dot << "    // === Легенда ===\n";
    dot << "    subgraph cluster_legend {\n";
    dot << "        label=\"Legend\";\n";
    dot << "        style=dashed;\n";
    dot << "        legend_conv [label=\"Conv\", style=filled, fillcolor=lightblue, shape=box];\n";
    dot << "        legend_relu [label=\"Relu\", style=filled, fillcolor=lightgreen, shape=ellipse];\n";
    dot << "        legend_gemm [label=\"Gemm/MatMul\", style=filled, fillcolor=lightyellow, shape=box];\n";
    dot << "    }\n\n";

    // узлы
    dot << "    // === Узлы ===\n";
    for (size_t i = 0; i < nodes.size(); ++i) 
    {
        const auto& node = nodes[i];
        std::string node_id = "n" + std::to_string(i);
        std::string op_type = node.get_op_type();
    

        if (op_type.empty() || op_type == "Unknown") 
        {
            continue;
        }

        // вектор атрибутов
        std::vector<std::string> attrs;
    
        for (const auto& [name, vals] : node.get_ints_attrs()) 
        {
            if (!vals.empty()) 
            {
                std::string attr_str = name + "=[";
                for (size_t j = 0; j < vals.size() && j < 4; ++j) 
                {
                    attr_str += std::to_string(vals[j]);
                    if (j + 1 < vals.size() && j < 3) 
                    {
                        attr_str += ",";
                    }
                }

                if (vals.size() > 4) attr_str += "...";
                attr_str += "]";
                attrs.push_back(attr_str);
            }
        }

        for (const auto& [name, val] : node.get_float_attrs()) 
        {
            attrs.push_back(name + "=" + std::to_string(val));
        }

        for (const auto& [name, val] : node.get_string_attrs()) 
        {
            attrs.push_back(name + "=" + val);
        }
    
        // label
        dot << "    " << node_id << " [label=<";
        dot << "<TABLE BORDER=\"1\" CELLBORDER=\"0\" CELLSPACING=\"2\" CELLPADDING=\"3\">";
    
        // заголовок (тип операции) 
        dot << "<TR><TD BGCOLOR=\"";
        dot << get_node_color(op_type);
        dot << "\"><B>";
        dot << escape_html(op_type);  

        dot << "</B></TD></TR>";
    
        // атрибуты 
        for (const auto& attr : attrs) 
        {
            dot << "<TR><TD>";
            dot << escape_html(attr);  
            dot << "</TD></TR>";
        }
    
        dot << "</TABLE>>";
        dot << ", style=filled, fillcolor=" << get_node_color(op_type) << ", ";
        dot << "shape=" << get_node_shape(op_type) << "];\n";
    }

    // === Рёбра  ===
    dot << "    // === Рёбра ===\n";
    
    // имя тензора → список узлов, которые его производят
    std::unordered_map<std::string, std::vector<std::string>> tensor_producers;

    for (size_t i = 0; i < nodes.size(); i++) 
    {
        const auto& node = nodes[i];
        std::string node_id = "n" + std::to_string(i);

        for (const auto& out : node.get_outputs()) 
        {
            if (!out.empty()) 
            {
                tensor_producers[out].push_back(node_id);
            }
        }
    }
    
    // делаем  рёбра: вход тензора → узел-потребитель
    for (size_t i = 0; i < nodes.size(); i++) 
    {
        const auto& node = nodes[i];
        std::string node_id = "n" + std::to_string(i);
        
        for (const auto& inp : node.get_inputs()) 
        {
            if (inp.empty()) continue;

            // пропуск не основных входов
            if (inp.find(".weight") != std::string::npos || 
                inp.find(".bias") != std::string::npos ||
                inp == "namespace") {
                continue;  
            }
            
            // делаем ребро
            if (tensor_producers.count(inp)) 
            {
                for (const auto& producer_id : tensor_producers[inp]) 
                {
                    dot << "    " << producer_id << " -> " << node_id;
                    
                    // подпись ребра - имя тензора 
                    if (inp.size() < 30) 
                    {
                        dot << " [label=\"" << escape_dot(inp) << "\"]";
                    }
                    dot << ";\n";
                }
            } else {
                std::string input_node_id = "input_" + std::to_string(i) + "_" + std::to_string(tensor_producers.size());
                dot << "    " << input_node_id << " [label=\"" << escape_dot(inp) << "\", ";
                dot << "shape=plaintext, style=dashed];\n";
                dot << "    " << input_node_id << " -> " << node_id << ";\n";
            }
        }
    }

    // === выходы графа ===
    if (!outputs.empty()) {
        dot << "\n    // === Выходы ===\n";
        for (const auto& out_name : outputs) 
        {
            std::string output_node_id = "out_" + std::to_string(&out_name - outputs.data());
            dot << "    " << output_node_id << " [label=\"" << escape_dot(out_name) << "\", ";
            dot << "shape=doubleellipse, style=filled, fillcolor=gold];\n";
            
            // поиск узла, которого производит этот выход
            for (size_t i = 0; i < nodes.size(); i++) 
            {
                const auto& node = nodes[i];
                std::string node_id = "n" + std::to_string(i);

                for (const auto& node_out : node.get_outputs()) 
                {
                    if (node_out == out_name) 
                    {
                        dot << "    " << node_id << " -> " << output_node_id << " [style=bold];\n";
                    }
                }
            }
        }
    }

    dot << "}\n";
    dot.close();
    
    std::cout << "Граф экспортирован в " << filename << "\n";
    std::cout << "Для просмотра: dot -Tpng " << filename << " -o graph.png && open graph.png\n";
}