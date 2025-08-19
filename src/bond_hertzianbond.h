#ifdef BOND_CLASS
BondStyle(hertzbond,BondHertzianBond)
#else

#ifndef LMP_BOND_HERTZIANBOND_H
#define LMP_BOND_HERTZIANBOND_H

#include "bond.h"
#include <map>
#include "atom.h" // for tagint

struct HertzBondState {
    bool broken;
    double A;      // cross-section area
    double Rbond;  // bond radius
    double J;      // polar moment of inertia
    double Fn;     // normal force
    double Ft[3];  // tangential force components
    double Mn;     // normal torque
    double Mt[3];  // tangential torque components
};

namespace LAMMPS_NS {

class BondHertzianBond : public Bond {
public:
    BondHertzianBond(class LAMMPS *);
    ~BondHertzianBond() override;

    void compute(int, int) override;
    void coeff(int, char **) override;
    double equilibrium_distance(int) override;
    void write_restart(FILE *) override;
    void read_restart(FILE *) override;
    void allocate();

protected:
    double *young;    // Young's modulus
    double *poisson;  // Poisson ratio
    double *damp;     // damping ratio
    double *sigmax;   // max normal stress
    double *taumax;   // max shear stress

    std::map<std::pair<tagint, tagint>, HertzBondState> state;
};

}

#endif
#endif
