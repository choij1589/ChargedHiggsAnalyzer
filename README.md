# ChargedHiggsAnalyzer
Author: Jin Choi</br>
Contact: choij@cern.ch</br>
Keywords: Charged Higgs, 2HDM, ParticleNet</br>

---
### Notes
The analyzer shares the same ntuple with SKFlatAnalyzer,
so the classes and functions are almost the translation of C++ SKFlatAnalyzer codes
to python code. There are two purposes for this analyzer,
- Try to use torch\_geometric codes as native as possible(i.e. without TorchScript support)
- Add systematic calculations
---
### Naming convention
#### For directory and file names
- Directory names use upper camel case
- File names use lower camel case
- snake case is depreciated
#### For C++ and python codes
- Types and classes use upper camel case
- Variables and constants use lower camel case, try to use nouns as possible
- functions use lower camel case, try to use verbs as possible
- type annotation should be used as much as possible
---
### TODO
- Running time of the systematic sources is time consuming. Try to find out the ways to optimize the processes.
- Additional skim of SKFlat ntuples?
