#include "pair_hertzianbond.h"
#include "atom.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "domain.h"
#include "update.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include "force.h"
#include <cmath>
#include <cstring>
#include "pair.h"
#include "memory.h"
#include "error.h"
#include <cstdlib>
using namespace LAMMPS_NS;

PairHertzianBond::PairHertzianBond(LAMMPS *lmp) : Pair(lmp) {
    youngModulus = 0.0;
    poissonRatio = 0.0;
    bond_stiffness = 0.0;
    dampingRatio = 0.0;
    bond_normal_strength = 5.0e6;
    bond_shear_strength  = 5.0e6;
    // binderRatio = 0.24;
    one_coeff = 1;
    cut_global = 0.0;
}

PairHertzianBond::~PairHertzianBond() {
    memory->destroy(setflag);
    memory->destroy(cut);
    memory->destroy(cutsq);
    bondMap.clear();                     // 결합 데이터 해제
}

void PairHertzianBond::settings(int narg, char **arg) {
    if (narg != 2) {
        error->all(FLERR, "Illegal pair_style hertzianbond command");
    }
    binderRatio = atof(arg[0]);
    cut_global  = atof(arg[1]);
    cutforce    = cut_global;

    if (allocated) {
        for (int i = 1; i <= atom->ntypes; ++i)
            for (int j = 1; j <= atom->ntypes; ++j)
                if (setflag[i][j]) {
                    cut[i][j]   = cut_global;
                    cutsq[i][j] = cut_global * cut_global;
                }
    }
}

void PairHertzianBond::coeff(int narg, char **arg) {
    if (narg < 4) {
        error->all(FLERR, "Incorrect number of coeff arguments for pair_style hertzianbond");
    }
    youngModulus   = atof(arg[0]);
    poissonRatio   = atof(arg[1]);
    bond_stiffness = atof(arg[2]);
    dampingRatio   = atof(arg[3]);
    if (narg >= 6) {
        bond_normal_strength = atof(arg[4]);
        bond_shear_strength  = atof(arg[5]);
    }
    if (!allocated) allocate();
    if (cut_global <= 0.0) {
        error->all(FLERR, "Pair cutoff must be positive for hertzianbond");
    
    }
    
    cutforce = cut_global;

    for (int i = 1; i <= atom->ntypes; ++i)
        for (int j = 1; j <= atom->ntypes; ++j) {
            setflag[i][j] = 1;
            cut[i][j]    = cut_global;
            cutsq[i][j]  = cut_global * cut_global;
        }
}

void PairHertzianBond::allocate() {
    allocated = 1;
    int n = atom->ntypes;
    setflag = memory->create(setflag, n+1, n+1, "pair:setflag");
    cut     = memory->create(cut,     n+1, n+1, "pair:cut");
    cutsq   = memory->create(cutsq,   n+1, n+1, "pair:cutsq");
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= n; ++j)
            setflag[i][j] = 0;
}

double PairHertzianBond::init_one(int i, int j) {
    return cut_global;
}

void PairHertzianBond::compute(int eflag, int vflag) {
    double **x = atom->x;
    double **v = atom->v;
    double **omega = atom->omega;
    double *radius = atom->radius;
    double *rmass   = atom->rmass;
    // int *type = atom->type;
    int nlocal = atom->nlocal;

    double E_eff = youngModulus / (2.0 * (1 - poissonRatio * poissonRatio));

    int *ilist = list->ilist;
    int inum  = list->inum;

    for (int ii = 0; ii < inum; ++ii) {
        int i = ilist[ii];
        double xi = x[i][0], yi = x[i][1], zi = x[i][2];
        double ri = radius[i];
        int *jlist = list->firstneigh[i];
        int jnum   = list->numneigh[i];

        for (int jj = 0; jj < jnum; ++jj) {
            // int j = jlist[jj];
            // if (j >= nlocal) continue;   // 고스트 원자 무시
            // if (j <= i) continue;        // 각 쌍을 한 번만 처리
            int j = jlist[jj] & NEIGHMASK; //코덱스 추가
            bool jlocal = (j < nlocal);    //코덱스 추가

            double xj = x[j][0], yj = x[j][1], zj = x[j][2];
            double rj = radius[j];

            double dx = xj - xi;
            double dy = yj - yi;
            double dz = zj - zi;
            double r_ij = sqrt(dx*dx + dy*dy + dz*dz);
            if (r_ij < 1e-12) r_ij = 1e-12;

            double nx = dx / r_ij;
            double ny = dy / r_ij;
            double nz = dz / r_ij;
            double contactDist = ri + rj;
            double overlap = contactDist - r_ij;
            bool inContact = (overlap > 0.0);

            tagint ti = atom->tag[i];
            tagint tj = atom->tag[j];
            std::pair<tagint, tagint> key =
                (ti < tj ? std::make_pair(ti, tj) : std::make_pair(tj, ti));

            BondData *bond = nullptr;
            auto it = bondMap.find(key);
            if (it == bondMap.end()) {
                if (r_ij < contactDist * (1.0 + binderRatio)) {
                    BondData newbond;
                    newbond.broken = false;
                    double minR = (ri < rj ? ri : rj);
                    newbond.Rbond = 0.5 * minR;
                    newbond.A = M_PI * newbond.Rbond * newbond.Rbond;
                    newbond.J = 0.5 * M_PI * pow(newbond.Rbond, 4);
                    newbond.Sn = bond_stiffness;
                    newbond.St = bond_stiffness;
                    newbond.a_damp = dampingRatio;
                    newbond.Fn_bond = 0.0;
                    newbond.Ft_bond[0] = newbond.Ft_bond[1] = newbond.Ft_bond[2] = 0.0;
                    newbond.Mn_bond = 0.0;
                    newbond.Mt_bond[0] = newbond.Mt_bond[1] = newbond.Mt_bond[2] = 0.0;
                    bondMap[key] = newbond;
                    bond = &bondMap[key];
                }
            } else {
                bond = &(it->second);
            }

            double force_i[3] = {0.0, 0.0, 0.0};
            double torque_i[3] = {0.0, 0.0, 0.0};
            double force_j[3] = {0.0, 0.0, 0.0};
            double torque_j[3] = {0.0, 0.0, 0.0};

            if (inContact) {
                double Reff = (ri * rj) / (ri + rj);
                double kn = (4.0/3.0) * E_eff * sqrt(Reff);
                double Fn_contact = kn * pow(overlap, 1.5);
                double meff = (rmass[i] * rmass[j]) /
                              (rmass[i] + rmass[j]);
                double kn_linear = (overlap > 0.0 ? 1.5 * kn * sqrt(overlap) : kn);
                double Cn = 2.0 * dampingRatio * sqrt(meff * kn_linear);
                double vn = (v[j][0]-v[i][0])*nx + (v[j][1]-v[i][1])*ny + (v[j][2]-v[i][2])*nz;
                double Fdamp_n = - Cn * vn;
                double Fn_total = Fn_contact + Fdamp_n;
                if (Fn_total < 0.0) Fn_total = 0.0;
                force_i[0] += Fn_total * nx;
                force_i[1] += Fn_total * ny;
                force_i[2] += Fn_total * nz;
                force_j[0] -= Fn_total * nx;
                force_j[1] -= Fn_total * ny;
                force_j[2] -= Fn_total * nz;
            }

            if (bond && !bond->broken) {
                double vx_rel = v[j][0] - v[i][0];
                double vy_rel = v[j][1] - v[i][1];
                double vz_rel = v[j][2] - v[i][2];
                double vn_rel = vx_rel*nx + vy_rel*ny + vz_rel*nz;
                double vx_t = vx_rel - vn_rel * nx;
                double vy_t = vy_rel - vn_rel * ny;
                double vz_t = vz_rel - vn_rel * nz;
                double dFtx = - bond->St * bond->A * vx_t * update->dt;
                double dFty = - bond->St * bond->A * vy_t * update->dt;
                double dFtz = - bond->St * bond->A * vz_t * update->dt;
                bond->Ft_bond[0] += dFtx;
                bond->Ft_bond[1] += dFty;
                bond->Ft_bond[2] += dFtz;
                double dFn = - bond->Sn * bond->A * vn_rel * update->dt;
                bond->Fn_bond += dFn;

                double wx_rel = omega[j][0] - omega[i][0];
                double wy_rel = omega[j][1] - omega[i][1];
                double wz_rel = omega[j][2] - omega[i][2];
                double w_n_rel = wx_rel*nx + wy_rel*ny + wz_rel*nz;
                double wx_par = w_n_rel * nx;
                double wy_par = w_n_rel * ny;
                double wz_par = w_n_rel * nz;
                double wx_perp = wx_rel - wx_par;
                double wy_perp = wy_rel - wy_par;
                double wz_perp = wz_rel - wz_par;
                bond->Mn_bond += - w_n_rel * bond->Sn * (bond->J / 2.0) * update->dt;
                bond->Mt_bond[0] += - wx_perp * bond->St * bond->J * update->dt;
                bond->Mt_bond[1] += - wy_perp * bond->St * bond->J * update->dt;
                bond->Mt_bond[2] += - wz_perp * bond->St * bond->J * update->dt;

                bond->Fn_bond    *= (1.0 - bond->a_damp * update->dt);
                bond->Ft_bond[0] *= (1.0 - bond->a_damp * update->dt);
                bond->Ft_bond[1] *= (1.0 - bond->a_damp * update->dt);
                bond->Ft_bond[2] *= (1.0 - bond->a_damp * update->dt);
                bond->Mn_bond    *= (1.0 - bond->a_damp * update->dt);
                bond->Mt_bond[0] *= (1.0 - bond->a_damp * update->dt);
                bond->Mt_bond[1] *= (1.0 - bond->a_damp * update->dt);
                bond->Mt_bond[2] *= (1.0 - bond->a_damp * update->dt);

                double Mt_mag = sqrt(bond->Mt_bond[0]*bond->Mt_bond[0]
                                   + bond->Mt_bond[1]*bond->Mt_bond[1]
                                   + bond->Mt_bond[2]*bond->Mt_bond[2]);
                double sigma_n = bond->Fn_bond / bond->A +
                                 2.0 * Mt_mag * bond->Rbond / bond->J;
                double Ft_mag = sqrt(bond->Ft_bond[0]*bond->Ft_bond[0]
                                   + bond->Ft_bond[1]*bond->Ft_bond[1]
                                   + bond->Ft_bond[2]*bond->Ft_bond[2]);
                double tau = Ft_mag / bond->A +
                             fabs(bond->Mn_bond) * bond->Rbond / bond->J;
                if (sigma_n > bond_normal_strength ||
                    tau > bond_shear_strength ||
                    r_ij > contactDist * (1.0 + binderRatio)) {
                    bondMap.erase(key);   // 파괴된 결합 제거
                    bond = nullptr;
                }
            }

            if (bond) {
                force_i[0] += bond->Fn_bond * nx;
                force_i[1] += bond->Fn_bond * ny;
                force_i[2] += bond->Fn_bond * nz;
                force_j[0] -= bond->Fn_bond * nx;
                force_j[1] -= bond->Fn_bond * ny;
                force_j[2] -= bond->Fn_bond * nz;

                force_i[0] += bond->Ft_bond[0];
                force_i[1] += bond->Ft_bond[1];
                force_i[2] += bond->Ft_bond[2];
                force_j[0] -= bond->Ft_bond[0];
                force_j[1] -= bond->Ft_bond[1];
                force_j[2] -= bond->Ft_bond[2];

                torque_i[0] += bond->Mn_bond * nx + bond->Mt_bond[0];
                torque_i[1] += bond->Mn_bond * ny + bond->Mt_bond[1];
                torque_i[2] += bond->Mn_bond * nz + bond->Mt_bond[2];
                torque_j[0] -= bond->Mn_bond * nx + bond->Mt_bond[0];
                torque_j[1] -= bond->Mn_bond * ny + bond->Mt_bond[1];
                torque_j[2] -= bond->Mn_bond * nz + bond->Mt_bond[2];
            }

            atom->f[i][0] += force_i[0];
            atom->f[i][1] += force_i[1];
            atom->f[i][2] += force_i[2];
            atom->torque[i][0] += torque_i[0];
            atom->torque[i][1] += torque_i[1];
            atom->torque[i][2] += torque_i[2];

             if (jlocal) {
                atom->f[j][0] += force_j[0];
                atom->f[j][1] += force_j[1];
                atom->f[j][2] += force_j[2];
                atom->torque[j][0] += torque_j[0];
                atom->torque[j][1] += torque_j[1];
                atom->torque[j][2] += torque_j[2];
             }
        }
    }
}