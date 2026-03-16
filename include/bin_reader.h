#pragma once

#include <cstddef>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>

// класс, который будет считывать ONNX файл и разбирать его
class BinaryReader
{
private:
    std::vector<uint8_t> byte_vector;
    size_t cur_index;
    size_t size;

public:
    // конструктор
    BinaryReader(const std::string& file_name) 
        : cur_index(0), size(0)  
    {
        std::ifstream onnxFile(file_name, std::ios::binary);
        if (!onnxFile) 
        {
            throw std::runtime_error("Ошибка открытия файла .onnx");
        }

        // вычисляем размер данных
        onnxFile.seekg(0, std::ios::end);
        size = onnxFile.tellg();
        onnxFile.seekg(0, std::ios::beg);  

        // Создаём буфер нужного размера
        byte_vector.resize(size);

        // Читаем все байты файла
        if (!onnxFile.read(reinterpret_cast<char*>(byte_vector.data()), size)) 
        {
            throw std::runtime_error("Ошибка чтения файла");
        }
    }

    // функция для просмотра текущего байта 
    uint8_t watch_cur_byte()
    {
        if (cur_index >= size) return 0;
        return byte_vector[cur_index];
    }

    // прочитать байт и сдвинуться
    uint8_t read_byte()
    {
        if (cur_index >= size) throw std::out_of_range("Unexpected EOF");
        uint8_t cur_byte = byte_vector[cur_index];
        cur_index++;
        return cur_byte;
    }


    // прочитать varint
    uint64_t read_varint()
    {
        uint64_t result = 0;
        int shift = 0;
    
        while (cur_index < size)
        {
            uint8_t byte = byte_vector[cur_index++];
            result |= static_cast<uint64_t>(byte & 0x7F) << shift;
            if ((byte & 0x80) == 0) return result;

            shift += 7;
            if (shift >= 64) throw std::runtime_error("Varint too long");
        }

        if (cur_index > 0) cur_index--;
        throw std::runtime_error("Unexpected EOF while reading varint");
    }


    // функция считывания n байтов подряд 
    std::vector<uint8_t> read_bytes(size_t n)  
    {
        if (cur_index + n > size) throw std::out_of_range("Unexpected EOF");
        std::vector<uint8_t> data(byte_vector.begin() + cur_index, 
                              byte_vector.begin() + cur_index + n);
        cur_index += n;
        return data;
    }

    // функция для проверки выхода за границу массива битов
    bool check_eof()
    {
        if (cur_index >= size)
        {
            return true;
        }

        else
        {
            return false;
        }
    }

    // геттер для текущего индекса
    size_t get_cur_pos()
    {
        return cur_index;
    }
};