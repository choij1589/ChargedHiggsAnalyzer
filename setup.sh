#!/bin/sh
if [[ $HOSTNAME == *"tamsa2"* ]]; then
    echo "@@@@ Working on tamsa2"
    export WORKDIR="/data6/Users/choij/ChargedHiggsAnalyzer"

    # Python setup
    source /cvmfs/sft.cern.ch/lcg/releases/LCG_102cuda/Python/3.9.12/x86_64-centos7-gcc8-opt/Python-env.sh
    # ROOT setup
    source /cvmfs/sft.cern.ch/lcg/releases/LCG_102cuda/tbb/2020_U2/x86_64-centos7-gcc8-opt/tbb-env.sh
    source /cvmfs/sft.cern.ch/lcg/releases/LCG_102cuda/ROOT/6.26.04/x86_64-centos7-gcc8-opt/ROOT-env.sh
else
    echo @@@@ Working in local...
    export WORKDIR="$HOME/workspace/ChargedHiggsAnalyzer"
    echo @@@@ WORKDIR=$WORKDIR
    source $HOME/.conda-activate
    conda activate torch
fi

alias splitFile="${WORKDIR}/libCpp/splitFile"
alias copyFile="${WORKDIR}/libCpp/copyFile"
