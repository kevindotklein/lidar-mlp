#ifndef LOG_HPP
#define LOG_HPP

#include <iostream>
#include <string>
#include <fstream>

namespace L {
    void log(const std::string& message, const std::string& filename = "log.txt") {
        std::ofstream file(filename, std::ios::app); 

        if (file.is_open()) {
            file << message;
            file.close();
        } else {
            std::cerr << "Unable to open log file: " << filename << std::endl;
        }
    }
}

#endif