#include "errors.hpp"

void bail_if(bool check, std::string &mes) {
    if (check) {
        throw std::runtime_error(mes);
    }
}
