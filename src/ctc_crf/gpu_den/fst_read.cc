// Copyright 2018-2019 Tsinghua University, Author: Hongyu Xiang
// Apache 2.0.
// This file contains functions for reading the denominator WFST.

#include <fst/fstlib.h>
using namespace fst;

extern int DEN_NUM_STATES;
extern int DEN_NUM_ARCS;

void ReadFst(const char * fst_name,
             vector<vector<int> > &alpha_next,
             vector<vector<int> > &beta_next,
             vector<vector<int> > &alpha_ilabel,
             vector<vector<int> > &beta_ilabel,
             vector<vector<float> > &alpha_weight,
             vector<vector<float> > &beta_weight,
             vector<float> &start_weight,
             vector<float> &end_weight,
             int &num_states,
             int &num_arcs) {
    // assume that they are proper initialized
    StdVectorFst *fst = StdVectorFst::Read(fst_name);
    num_states = fst->NumStates(); 
    DEN_NUM_STATES = num_states;

    num_arcs = 0;
    for (StateIterator<StdVectorFst> siter(*fst); !siter.Done(); siter.Next()) {
        num_arcs += fst->NumArcs(siter.Value());
    }
    DEN_NUM_ARCS = num_arcs;

    alpha_next.resize(num_states, vector<int>());
    beta_next.resize(num_states, vector<int>());
    alpha_ilabel.resize(num_states, vector<int>());
    beta_ilabel.resize(num_states, vector<int>());
    alpha_weight.resize(num_states, vector<float>());
    beta_weight.resize(num_states, vector<float>());

    start_weight.resize(num_states, -float(INFINITY));
    end_weight.resize(num_states, -float(INFINITY));

    start_weight[fst->Start()] = 0.;

    for (StateIterator<StdVectorFst> siter(*fst); !siter.Done(); siter.Next()){
        if (fst->Final(siter.Value()) != StdArc::Weight::Zero()) {
            end_weight[siter.Value()] = -fst->Final(siter.Value()).Value();
        }
        int state = siter.Value();

        for (ArcIterator<StdVectorFst> aiter(*fst, siter.Value()); !aiter.Done(); aiter.Next()) {
            beta_next[state].push_back(aiter.Value().nextstate);
            alpha_next[aiter.Value().nextstate].push_back(state);

            beta_ilabel[state].push_back(aiter.Value().ilabel-1);
            alpha_ilabel[aiter.Value().nextstate].push_back(aiter.Value().ilabel-1);

            beta_weight[state].push_back(-aiter.Value().weight.Value());
            alpha_weight[aiter.Value().nextstate].push_back(-aiter.Value().weight.Value());
        }
    }
}
