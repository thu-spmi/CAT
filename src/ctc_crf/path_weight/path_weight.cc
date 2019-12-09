#include <iostream>
#include <fstream>
#include <istream>
#include <ostream>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <string>
#include <iomanip>
#include <fst/fstlib.h>
using namespace fst;

float pathWeight(StdVectorFst *ifst) {
    StdArc::Weight s = StdArc::Weight::One();
    for (StateIterator<StdVectorFst> siter(*ifst); !siter.Done(); siter.Next()){
        // std::cout << ifst->NumArcs(siter.Value()) << std::endl;
        if (ifst->NumArcs(siter.Value()) != 1 && ifst->Final(siter.Value()) == StdArc::Weight::Zero()) {
            std::cerr << "The number of possible path does not equal to 1" << std::endl;
        } ;
        StdArc::Weight w = StdArc::Weight::Zero();
        for (ArcIterator<StdVectorFst> aiter(*ifst, siter.Value()); !aiter.Done(); aiter.Next()) {
            w = Plus(w, aiter.Value().weight);
        }
        // std::cout << "arc weight: " << w.Value() << std::endl;
        if (ifst->Final(siter.Value()) == StdArc::Weight::Zero()) {
            s = Times(s, w);
        } else {
            s = Times(s, ifst->Final(siter.Value()));
        }
    }
    return s.Value();
}

int main(int argc, char* argv[]) {
    std::ifstream fin(argv[1], std::ios::in);
    std::string line;
    std::vector<std::vector<int> > vec;
    std::vector<std::string> utt;
    const char *sep = " ";
    char *p;
    while (getline(fin,line)) {
        vec.push_back(std::vector<int>());
        p = std::strtok(const_cast<char*>(line.c_str()), sep);
        // std::cout << "p: " << p << std::endl;
        // std::cout << line << std::endl;
        utt.push_back(p);

        p = strtok(NULL, sep);
        while (p) {
            vec.back().push_back(atoi(p));
            p = strtok(NULL, sep);
        }
        // std::cout << line << std::endl;
    }

    StdVectorFst *phone_lm = StdVectorFst::Read(argv[2]);
    StdVectorFst *linearFST = new StdVectorFst();
    StdVectorFst *ofst = new StdVectorFst();
    
    for (size_t i = 0; i < vec.size(); i++) {
        linearFST->DeleteStates();
        ofst->DeleteStates();
        int cur_state = linearFST->AddState();
        linearFST->SetStart(cur_state);
        for (size_t j = 0; j < vec[i].size(); j++) {
            int next_state = linearFST->AddState();
            linearFST->AddArc(cur_state, StdArc(vec[i][j], vec[i][j], TropicalWeight::One(), next_state));
            cur_state = next_state;
        }
        linearFST->SetFinal(cur_state, TropicalWeight::One());
        Compose(*linearFST, *phone_lm, ofst);
        std::cout << utt[i] << " ";
        std::cout << std::setprecision(8) << -pathWeight(ofst) << std::endl;
    }
    delete phone_lm;
    delete linearFST;
    delete ofst;

    return 0;
}
