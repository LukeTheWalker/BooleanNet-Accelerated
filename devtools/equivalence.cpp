#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <sstream>
#include <unordered_map>
#include <sstream>

using namespace std;

typedef struct implication_t{
    string gene1;
    string gene2;
    int type;
    float stat;
    float pval;
    bool operator==(const implication_t& other) const {
        return gene1 == other.gene1 && gene2 == other.gene2 && type == other.type;
    }
    friend ostream& operator<<(ostream& os, const implication_t& imp){
        os << imp.gene1 << " " << imp.gene2 << " " << imp.type << " " << imp.stat << " " << imp.pval;
        return os;
    }

} implication;

typedef struct symm_impl_t{
    string gene1;
    string gene2;
    int type;
    float stat[2];
    float pval[2];
    bool operator==(const symm_impl_t& other) const {
        return gene1 == other.gene1 && gene2 == other.gene2 && type == other.type;
    }
    friend ostream& operator<<(ostream& os, const symm_impl_t& imp){
        os << imp.gene1 << " " << imp.gene2 << " " << imp.type << " " << imp.stat[0] << " " << imp.stat[1] << " " << imp.pval[0] << " " << imp.pval[1];
        return os;
    }
} symm_impl;


int get_impl_type_micale (string val1, string val2){
    if (val1 == "low" && val2 == "low"){
        return 0;
    }
    else if (val1 == "low" && val2 == "high"){
        return 1;
    }
    else if (val1 == "high" && val2 == "low"){
        return 2;
    }
    else if (val1 == "high" && val2 == "high"){
        return 3;
    }
    else {
        cerr << "Error: invalid implication type" << endl;
        exit(1);
    }
}

int get_implication_type_luca (string impl){
    if (impl == "low-low"){
        return 0;
    }
    else if (impl == "low-high"){
        return 1;
    }
    else if (impl == "high-low"){
        return 2;
    }
    else if (impl == "high-high"){
        return 3;
    }
    else if (impl == "equivalence"){
        return 4;
    }
    else if (impl == "opposite"){
        return 5;
    }
    else {
        cerr << "Error: invalid implication type" << endl;
        exit(1);
    }
}

void read_implications_micale (string file, vector<implication> &implications, vector<symm_impl> &symm_impls){
    ifstream in(file);

    string line;
    getline(in, line);

    while (in.eof() == false){
        string gene1, gene2, val1, val2, arrow;
        string doubt;
        int type;
        float stat1, stat2, pvalue1, pvalue2;
        in >> gene1 >> val1 >> arrow >> gene2 >> val2;
        in >> doubt;
        if (gene1 == "") break;
        if (doubt == "AND"){
            in >> doubt >> doubt >> doubt >> doubt >> doubt;
            string fused_values;
            in >> fused_values;
            istringstream iss(fused_values);
            vector<string> values;
            string value;
            getline(iss, value, ',');
            stat1 = stof(value);
            getline(iss, value, ',');
            stat2 = stof(value);
            in >> fused_values;
            iss = istringstream(fused_values);
            getline(iss, value, ',');
            pvalue1 = stof(value);
            getline(iss, value, ',');
            pvalue2 = stof(value);
            symm_impl symm;
            if (val2 == "low")
                symm = {gene1, gene2, 4, {stat1, stat2}, {pvalue1, pvalue2}};
            else 
                symm = {gene1, gene2, 5, {stat1, stat2}, {pvalue1, pvalue2}};
            symm_impls.push_back(symm);
        }
        else {
            in  >> pvalue1;
            stat1 = stof(doubt);
            type = get_impl_type_micale(val1, val2);
            implication imp = {gene1, gene2, type, stat1, pvalue1};
            implications.push_back(imp);
            // cout << "Implication added: " << gene1 << " " << gene2 << " " << type << " " << stat << " " << pvalue << endl;
        }
    }
    in.close();
}

void read_implications_luca (string file, vector<implication>&implications, vector<symm_impl> &symm_impls){
    ifstream in(file);
    string line;

    getline(in, line);

    while (in.eof() == false){
        string gene1, gene2;
        string implication_type;
        float stat, pvalue;

        in >> gene1 >> gene2 >> implication_type  >> stat >> pvalue;
        if (gene1 == "") break;
        implication imp = {gene1, gene2, get_implication_type_luca(implication_type), stat, pvalue};
        implications.push_back(imp);
    }
    in.close();
    string symm_file = file.substr(0, file.length() - 4) + "_symm.txt";
    ifstream in_symm(symm_file);

    getline(in_symm, line);
    while (in_symm.eof() == false){        
        string gene1, gene2;
        string implication_type;
        float stat1, stat2, pval1, pval2;

        in_symm >> gene1 >> gene2 >> implication_type >> stat1 >> stat2 >> pval1 >> pval2;
        if (gene1 == "") break;
        symm_impl symm = {gene1, gene2, get_implication_type_luca(implication_type), {stat1, stat2}, {pval1, pval2}};
        symm_impls.push_back(symm);
    }
    in_symm.close();
}


bool within_tolerance(implication imp1, implication imp2, float pvalueTolerance, float statTolerance){
    return abs(imp1.stat - imp2.stat) > statTolerance || abs(imp1.pval - imp2.pval) > pvalueTolerance;
}

bool within_tolerance(symm_impl symm1, symm_impl symm2, float pvalueTolerance, float statTolerance){
    return abs(symm1.stat[0] - symm2.stat[0]) > statTolerance || abs(symm1.stat[1] - symm2.stat[1]) > statTolerance || abs(symm1.pval[0] - symm2.pval[0]) > pvalueTolerance || abs(symm1.pval[1] - symm2.pval[1]) > pvalueTolerance;
}

template <typename T>
struct ImplicationHash {
    std::size_t operator()(const T& imp) const {
        std::stringstream ss;
        ss << imp.gene1 << imp.gene2 << imp.type;
        return std::hash<std::string>()(ss.str());
    }
};

template <typename T>
void check_difference(const std::vector<T>& implications_micale, const std::vector<T>& implcations_luca, float pvalueTolerance, float statTolerance) {
    if (implications_micale.size() != implcations_luca.size()) cerr << "Different sizes found" << endl;
    std::unordered_map<T, int, ImplicationHash<T> > map;
    for (const auto& imp : implications_micale) 
        map[imp] = 1;
    
    cout << "Maps created" << endl;
    for (const auto& imp : implcations_luca) {
        auto it = map.find(imp);
        if (it == map.end() || it->second == 0)
            cerr << "Implication not found: " << imp;
        else if (within_tolerance(imp, it->first, pvalueTolerance, statTolerance)) {
            cerr << "Difference found Micale: " << imp << endl;
            cerr << "Difference found Luca:   " << it->first << endl;
        }
    }
    cout << "Done" << endl;
}

int main(int argc, char** argv) {

    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <pvalueTolerance> <statTolerance> <file_micale> <file_luca>" << endl;
        return 1;
    }

    float pvalueTolerance = stof(argv[1]);
    float statTolerance = stof(argv[2]);
    string file_micale = argv[3];
    string file_luca = argv[4];

    vector<implication> implications_micale;
    vector<implication> implications_luca;

    vector<symm_impl> symm_impls_micale;
    vector<symm_impl> symm_impls_luca;

    read_implications_micale(file_micale, implications_micale, symm_impls_micale);
    read_implications_luca(file_luca, implications_luca, symm_impls_luca);

    cout << "Implications Micale: " << implications_micale.size() << endl;
    cout << "Implications Luca:   " << implications_luca.size() << endl;

    check_difference(implications_micale, implications_luca, 0.001, 0.001);

    cout << "Symmetric Implications Micale: " << symm_impls_micale.size() << endl;
    cout << "Symmetric Implications Luca:   " << symm_impls_luca.size() << endl;

    check_difference(symm_impls_micale, symm_impls_luca, pvalueTolerance, statTolerance);

    return 0;
}