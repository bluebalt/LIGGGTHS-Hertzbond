#ifdef PAIR_CLASS
PairStyle(gran/hertzianbond, PairHertzianBond)
#else

#ifndef LMP_PAIR_HERTZIANBOND_H
#define LMP_PAIR_HERTZIANBOND_H

#include <map>
#include <vector>
#include "pointers.h"
#include "lammps.h"
#include "pair.h"
#include "atom.h"        // tagint 사용을 위해 포함

struct BondData {
    bool broken;
    double A;
    double Rbond;
    double J;
    double Sn, St;
    double a_damp;
    double Fn_bond;
    double Ft_bond[3];
    double Mn_bond;
    double Mt_bond[3];
};

namespace LAMMPS_NS {
class PairHertzianBond : public Pair {
public:
    PairHertzianBond(class LAMMPS *lmp);
    virtual ~PairHertzianBond();

    virtual void compute(int eflag, int vflag);
    virtual void settings(int, char **);
    virtual void coeff(int, char **);
    virtual double init_one(int i, int j);
    void allocate();

protected:
    double youngModulus;
    double poissonRatio;
    double bond_stiffness;
    double dampingRatio;
    double bond_normal_strength;
    double bond_shear_strength;
    double binderRatio;
    double **cut;
    double cut_global;
    std::map<std::pair<tagint, tagint>, BondData> bondMap; // tag 기반 키
};
}
#endif
#endif