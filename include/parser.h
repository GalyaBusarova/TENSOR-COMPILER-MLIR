#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <cstring>

#include "bin_reader.h"

// для расшифровки типа в onnx файле
enum DATA_TYPES
{
    UNDEFINED = 0,
    FLOAT = 1,
    UINT8 = 2,   
    INT8 = 3,    
    UINT16 = 4,  
    INT16 = 5,   
    INT32 = 6,   
    INT64 = 7,   
    STRING = 8,  
    BOOL = 9,    
    FLOAT16 = 10,
    DOUBLE = 11,
    UINT32 = 12,
    UINT64 = 13
};

// для расшифровки типов в парсинге атрибутов
enum ATTR_TYPES {
    ATTR_UNDEFINED = 0,
    ATTR_FLOAT = 1,     // поле f
    ATTR_INT = 2,       // поле i
    ATTR_STRING = 3,    // поле s
    ATTR_TENSOR = 4,    // поле t
    ATTR_GRAPH = 5,     // поле g
    ATTR_FLOATS = 6,    // поле floats
    ATTR_INTS = 7,      // поле ints
    ATTR_STRINGS = 8,   // поле strings
};

// очистка строки от мусора
std::string clean_string(const std::vector<uint8_t>& bytes);


// класс для хранения тензора
class Tensor
{
    std::string name;
    std::vector<int64_t> dims; // вектор размерностей
    int32_t data_type; // тип данных
    std::vector<uint8_t> raw_data;

public:
    // геттер для имени тензора
    std::string get_name()
    {
        return name;
    }

    // добавление размерности в массив размерностей
    void add_dim(int64_t dim)
    {
        dims.push_back(dim);
    }

    // сеттеры
    void set_name(std::string tensor_name)
    {
        name = tensor_name;
    }

    void set_data_type(int32_t type)
    {
        data_type = type;
    }

    void set_raw_data(const std::vector<uint8_t>& data) { raw_data = data; };

    // геттеры
    const std::string& get_name() const { return name; }
    const std::vector<int64_t>& get_dims() const { return dims; }
    const int32_t& get_data_type() const { return data_type; }
};



// класс, хранящий операцию и ее параметры
class Node
{
    std::string name;
    std::string op_type; // тип операции
    std::vector<std::string> inputs; // имена входных тензоров
    std::vector<std::string> outputs; // имена выходных тензоров

    // добавить атрибуты
public:
    // Словари для хранения атрибутов разных типов
    std::unordered_map<std::string, int64_t> int_attrs;        // для group
    std::unordered_map<std::string, float> float_attrs;        // для alpha, beta
    std::unordered_map<std::string, std::vector<int64_t>> ints_attrs; // для strides, dilations
    std::unordered_map<std::string, std::string> string_attrs; // для auto_pad

    // Методы для добавления атрибутов
    void add_int_attr(const std::string& name, int64_t value) { 
        int_attrs[name] = value; 
    }
    
    void add_float_attr(const std::string& name, float value) { 
        float_attrs[name] = value; 
    }
    
    void add_ints_attr(const std::string& name, const std::vector<int64_t>& value) { 
        ints_attrs[name] = value; 
    }
    
    void add_string_attr(const std::string& name, const std::string& value) { 
        string_attrs[name] = value; 
    }

    // добавить строку входа в имена входных тензоров
    void add_input(std::string input)
    {
        inputs.push_back(input);
    }

    // добавить строку выхода в имена выходных тензоров
    void add_output(std::string output)
    {
        outputs.push_back(output);
    }

    // сеттеры
    void set_name(std::string node_name)
    {
        name = node_name;
    }
    void set_op_type(std::string op_name)
    {
        op_type = op_name;
    }

    // геттеры
    const std::string& get_op_type() const { return op_type; }
    const std::vector<std::string>& get_inputs() const { return inputs; }
    const std::vector<std::string>& get_outputs() const { return outputs; }
    const std::string& get_name() const { return name; }
    // геттеры для атрибутов
    const std::unordered_map<std::string, std::vector<int64_t>>& get_ints_attrs() const { 
        return ints_attrs; 
    }

    const std::unordered_map<std::string, float>& get_float_attrs() const { 
        return float_attrs; 
    }

    const std::unordered_map<std::string, std::string>& get_string_attrs() const { 
        return string_attrs; 
    }
};

// класс для хранения графа
class Graph 
{
private:
    std::vector<Node> nodes;                          // все узлы
    std::unordered_map<std::string, Tensor> initializers;  // веса (поиск по имени)
    std::vector<std::string> inputs;                   // входы всей сети
    std::vector<std::string> outputs;                  // выходы всей сети

    int64_t ir_version;
    std::string producer_name;
    std::string producer_version;
    
    std::string graph_name;

public:
    // сеттеры
    void setIrVersion(int64_t version) { ir_version = version; }
    void setProducerName(const std::string& name) { producer_name = name; }
    void setProducerVersion(const std::string& version) { producer_version = version; }
    void setGraphName(const std::string& name) {graph_name = name; }

    // добавить новую ноду
    void add_node(Node node)
    {
        nodes.push_back(node);
    }

    // добавить новый тензор
    void add_tensor(Tensor tensor)
    {
        initializers[tensor.get_name()] = tensor;
    }

    // геттеры
    const std::vector<Node>& get_nodes() const { return nodes; }

    const std::unordered_map<std::string, Tensor>& get_initializers() const { return initializers; }

    const std::vector<std::string>& get_inputs() const { return inputs; }

    const std::vector<std::string>& get_outputs() const { return outputs; }

    // для отладки и тестов
    int64_t getIrVersion() const { return ir_version; }
    const std::string& getProducerName() const { return producer_name; }
    const std::string& getProducerVersion() const { return producer_version; }
    const std::string& getGraphName() const { return graph_name; }

    // функция для визуализации
    void export_to_dot(const std::string& filename) const;
};


class ONNXParser 
{
private:
    BinaryReader reader;      // читает байты
    Graph graph;              // сюда собираем результат
    
    // вспомогательная функция для парсинга графа
    void parseGraph(uint64_t length)
    {
        size_t end_pos = reader.get_cur_pos() + length;
        uint8_t cur_byte = reader.watch_cur_byte();

        while (reader.get_cur_pos() < end_pos) 
        {
            cur_byte = reader.watch_cur_byte();
            int wire_type = cur_byte & 0x07;
            int field_number = cur_byte >> 3;
            reader.read_byte();

            switch (field_number)
            {
                case 1: // Node, wire_type всегда равен 2
                {
                    uint64_t node_size = reader.read_varint(); // длина узла

                    Node res_node = parseNode(node_size);
                    graph.add_node(res_node);
                    break;
                }

                case 2: // name (graph name)
                {
                    uint64_t str_size = reader.read_varint();
                    if (reader.get_cur_pos() + str_size > end_pos) break;

                    auto bytes = reader.read_bytes(str_size);
                    std::string name = clean_string(bytes);
                    graph.setGraphName(name);
                    break;
                }

                case 5: // initializer
                {
                    uint64_t tensor_size = reader.read_varint();

                    Tensor res_tensor = parseTensor(tensor_size);
                    graph.add_tensor(res_tensor);
                    break;
                }

                case 11: // inputs графа (входы всей сети)
                {
                    uint64_t len = reader.read_varint();
                    if (reader.get_cur_pos() + len > end_pos) break;

                    reader.read_bytes(len);  // пока просто пропускаем
                    break;
                }

                case 12: // outputs графа (выходы всей сети)
                {
                    uint64_t len = reader.read_varint();
                    if (reader.get_cur_pos() + len > end_pos) break;
                    
                    reader.read_bytes(len);  // пока просто пропускаем
                    break;
                }   

                default: // для неизвестных полей
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
    }    

    // вспомогательная функция для парсинга одной вершины
    Node parseNode(uint64_t node_size);

    // вспомогательная функция для парсинга одного тензора
    Tensor parseTensor(uint64_t tensor_size);   
    
    // вспомогательная функция для парсинга атрибута
    void parseAttribute(Node& node, uint64_t attr_len);
    
public:
    ONNXParser(const std::string& filename);
    Graph parse();            
};

inline ONNXParser::ONNXParser(const std::string& filename) 
    : reader(filename) {}


