#include "InputParser.hpp"
#include <algorithm>
#include <iostream>

using namespace std;

InputParser::InputParser (int &argc, char **argv){
    for (int i=1; i < argc; ++i)
        this->tokens.push_back(string(argv[i]));
}

const string& InputParser::getCmdOption(const string &option) const{
    vector<string>::const_iterator itr;
    if (!cmdOptionExists(option)){
        std::cerr << "Error: option " << option << " not found" << endl;
    }
    itr = find(this->tokens.begin(), this->tokens.end(), option);
    if (itr != this->tokens.end() && ++itr != this->tokens.end()){
        return *itr;
    }
    static const string empty_string("");
    return empty_string;
}

bool InputParser::cmdOptionExists(const string &option) const{
    return find(this->tokens.begin(), this->tokens.end(), option) != this->tokens.end();
}