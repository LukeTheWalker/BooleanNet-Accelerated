#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "util.hpp"

using namespace std;

// 0: gene1 low  => gene2 low
// 1: gene1 low  => gene2 high
// 2: gene1 high => gene2 low
// 3: gene1 high => gene2 high
// 4: equivalence (gene1 low <=> gene2 low)  && (gene1 high <=> gene2 high) => 0 && 3
// 5: opposite    (gene1 low <=> gene2 high) && (gene1 high <=> gene2 low)  => 1 && 2

string get_impl_string (int impl){
    switch (impl){
        case 0: return "low-low";
        case 1: return "low-high";
        case 2: return "high-low";
        case 3: return "high-high";
        case 4: return "equivalence";
        case 5: return "opposite";
        default: return "unknown";
    }
}