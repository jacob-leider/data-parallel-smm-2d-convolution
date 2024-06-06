#pragma once

#include <sstream>
#include <string>

void bail_if(bool check, std::string &mes);
template <typename... Args> void bail_if(bool check, Args... args) {
    if (check) {
        std::stringstream ss;
        (ss << ... << args);
        throw std::runtime_error(ss.str());
    }
}
