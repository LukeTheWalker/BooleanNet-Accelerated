#ifndef INPUTPARSER_H
#define INPUTPARSER_H

#include <vector>
#include <string>

// https://stackoverflow.com/a/868894/10210279
class InputParser{
    public:
        InputParser (int &argc, char **argv);
        const std::string& getCmdOption(const std::string &option) const;
        bool cmdOptionExists(const std::string &option) const;
    private:
        std::vector <std::string> tokens;
};

#endif