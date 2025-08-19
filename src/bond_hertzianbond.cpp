#define _USE_MATH_DEFINES
#include <cmath>
#include "bond_hertzianbond.h"
#include "atom.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "update.h"
#include "memory.h"
#include "force.h"
#include "error.h"
// young = poisson = damp = sigmax = taumax = nullptr;
using namespace LAMMPS_NS;

BondHertzianBond::BondHertzianBond(LAMMPS *lmp) : Bond(lmp) {
    young = poisson = damp = sigmax = taumax = nullptr;
}

BondHertzianBond::~BondHertzianBond() {
    if (allocated) {
        memory->destroy(setflag);
        memory->destroy(young);
        memory->destroy(poisson);
        memory->destroy(damp);
        memory->destroy(sigmax);
        memory->destroy(taumax);
    }
}

void BondHertzianBond::allocate() {
    allocated = 1;
    int n = atom->nbondtypes;
    memory->create(young,n+1,"bond:young");
    memory->create(poisson,n+1,"bond:poisson");
    memory->create(damp,n+1,"bond:damp");
    memory->create(sigmax,n+1,"bond:sigmax");
    memory->create(taumax,n+1,"bond:taumax");
    memory->create(setflag,n+1,"bond:setflag");
    for (int i=1;i<=n;i++) setflag[i]=0;
}

void BondHertzianBond::coeff(int narg, char **arg) {
    if (narg != 6) error->all(FLERR,"Incorrect args for bond coefficients");
    if (!allocated) allocate();

    int ilo,ihi;
    force->bounds(arg[0],atom->nbondtypes,ilo,ihi);

    double young_one   = force->numeric(FLERR,arg[1]);
    double poisson_one = force->numeric(FLERR,arg[2]);
    double damp_one    = force->numeric(FLERR,arg[3]);
    double sig_one     = force->numeric(FLERR,arg[4]);
    double tau_one     = force->numeric(FLERR,arg[5]);

    int count=0;
    for (int i=ilo;i<=ihi;i++) {
        young[i]=young_one;
        poisson[i]=poisson_one;
        damp[i]=damp_one;
        sigmax[i]=sig_one;
        taumax[i]=tau_one;
        setflag[i]=1;
        count++;
    }
    if (count==0) error->all(FLERR,"Incorrect args for bond coefficients");
}

double BondHertzianBond::equilibrium_distance(int i) {
    return 0.0; // no preferred length
}

void BondHertzianBond::write_restart(FILE *fp) {
    fwrite(&young[1],sizeof(double),atom->nbondtypes,fp);
    fwrite(&poisson[1],sizeof(double),atom->nbondtypes,fp);
    fwrite(&damp[1],sizeof(double),atom->nbondtypes,fp);
    fwrite(&sigmax[1],sizeof(double),atom->nbondtypes,fp);
    fwrite(&taumax[1],sizeof(double),atom->nbondtypes,fp);
}

void BondHertzianBond::read_restart(FILE *fp) {
    allocate();
    if (comm->me==0) {
        fread(&young[1],sizeof(double),atom->nbondtypes,fp);
        fread(&poisson[1],sizeof(double),atom->nbondtypes,fp);
        fread(&damp[1],sizeof(double),atom->nbondtypes,fp);
        fread(&sigmax[1],sizeof(double),atom->nbondtypes,fp);
        fread(&taumax[1],sizeof(double),atom->nbondtypes,fp);
    }
    MPI_Bcast(&young[1],atom->nbondtypes,MPI_DOUBLE,0,world);
    MPI_Bcast(&poisson[1],atom->nbondtypes,MPI_DOUBLE,0,world);
    MPI_Bcast(&damp[1],atom->nbondtypes,MPI_DOUBLE,0,world);
    MPI_Bcast(&sigmax[1],atom->nbondtypes,MPI_DOUBLE,0,world);
    MPI_Bcast(&taumax[1],atom->nbondtypes,MPI_DOUBLE,0,world);
}

void BondHertzianBond::compute(int eflag, int vflag) {
    double **x = atom->x;
    double **v = atom->v;
    double **omega = atom->omega;
    double *radius = atom->radius;
    double **f = atom->f;
    double **torque = atom->torque;
    int **bondlist = neighbor->bondlist;
    int nbondlist = neighbor->nbondlist;
    int nlocal = atom->nlocal;
    int newton_bond = force->newton_bond;

    if (eflag || vflag) ev_setup(eflag,vflag); else evflag = 0;

    for (int n=0;n<nbondlist;n++) {
        int i1 = bondlist[n][0];
        int i2 = bondlist[n][1];
        int type = bondlist[n][2];

        double delx = x[i2][0]-x[i1][0];
        double dely = x[i2][1]-x[i1][1];
        double delz = x[i2][2]-x[i1][2];
        double rsq = delx*delx + dely*dely + delz*delz;
        double r = sqrt(rsq);
        if (r < 1e-12) r = 1e-12;
        double nx = delx/r;
        double ny = dely/r;
        double nz = delz/r;

        tagint ti = atom->tag[i1];
        tagint tj = atom->tag[i2];
        std::pair<tagint,tagint> key =
            (ti < tj ? std::make_pair(ti,tj) : std::make_pair(tj,ti));
        HertzBondState &bs = state[key];
        if (bs.A == 0.0) {
            bs.broken = false;
            double minR = radius[i1] < radius[i2] ? radius[i1] : radius[i2];
            bs.Rbond = 0.5*minR;
            bs.A = M_PI*bs.Rbond*bs.Rbond;
            bs.J = 0.5*M_PI*pow(bs.Rbond,4);
            bs.Fn = bs.Mn = 0.0;
            bs.Ft[0] = bs.Ft[1] = bs.Ft[2] = 0.0;
            bs.Mt[0] = bs.Mt[1] = bs.Mt[2] = 0.0;
        }
        if (bs.broken) continue;

        double vx_rel = v[i2][0]-v[i1][0];
        double vy_rel = v[i2][1]-v[i1][1];
        double vz_rel = v[i2][2]-v[i1][2];
        double vn_rel = vx_rel*nx + vy_rel*ny + vz_rel*nz;
        double vx_t = vx_rel - vn_rel*nx;
        double vy_t = vy_rel - vn_rel*ny;
        double vz_t = vz_rel - vn_rel*nz;

        double reff = (radius[i1]*radius[i2])/(radius[i1]+radius[i2]);
        double overlap = (radius[i1]+radius[i2]) - r;
        if (overlap <= 0.0) overlap = bs.Rbond;
        double sqrtval = sqrt(reff*overlap);
        double E = young[type];
        double nu = poisson[type];
        double Yeff = E/(2.0*(1.0 - nu*nu));
        double G = E/(2.0*(1.0 + nu));
        double Geff = G/(2.0*(2.0 - nu));
        double kn = (4.0/3.0)*Yeff*sqrtval;
        double kt = 8.0*Geff*sqrtval;

        bs.Ft[0] += -kt*bs.A*vx_t*update->dt;
        bs.Ft[1] += -kt*bs.A*vy_t*update->dt;
        bs.Ft[2] += -kt*bs.A*vz_t*update->dt;
        bs.Fn    += -kn*bs.A*vn_rel*update->dt;

        double wx_rel = omega[i2][0]-omega[i1][0];
        double wy_rel = omega[i2][1]-omega[i1][1];
        double wz_rel = omega[i2][2]-omega[i1][2];
        double w_n_rel = wx_rel*nx + wy_rel*ny + wz_rel*nz;
        double wx_par = w_n_rel*nx;
        double wy_par = w_n_rel*ny;
        double wz_par = w_n_rel*nz;
        double wx_perp = wx_rel - wx_par;
        double wy_perp = wy_rel - wy_par;
        double wz_perp = wz_rel - wz_par;
        bs.Mn += -w_n_rel*kn*(bs.J/2.0)*update->dt;
        bs.Mt[0] += -wx_perp*kt*bs.J*update->dt;
        bs.Mt[1] += -wy_perp*kt*bs.J*update->dt;
        bs.Mt[2] += -wz_perp*kt*bs.J*update->dt;

        double dampfac = 1.0 - damp[type]*update->dt;
        bs.Fn *= dampfac;
        bs.Ft[0]*=dampfac; bs.Ft[1]*=dampfac; bs.Ft[2]*=dampfac;
        bs.Mn *= dampfac;
        bs.Mt[0]*=dampfac; bs.Mt[1]*=dampfac; bs.Mt[2]*=dampfac;

        double Mt_mag = sqrt(bs.Mt[0]*bs.Mt[0]+bs.Mt[1]*bs.Mt[1]+bs.Mt[2]*bs.Mt[2]);
        double sigma_n = bs.Fn/bs.A + 2.0*Mt_mag*bs.Rbond/bs.J;
        double Ft_mag = sqrt(bs.Ft[0]*bs.Ft[0]+bs.Ft[1]*bs.Ft[1]+bs.Ft[2]*bs.Ft[2]);
        double tau = Ft_mag/bs.A + fabs(bs.Mn)*bs.Rbond/bs.J;
        if (sigma_n > sigmax[type] || tau > taumax[type]) {
            bs.broken = true;
            continue;
        }

        double fx = bs.Fn*nx + bs.Ft[0];
        double fy = bs.Fn*ny + bs.Ft[1];
        double fz = bs.Fn*nz + bs.Ft[2];
        double tx = bs.Mn*nx + bs.Mt[0];
        double ty = bs.Mn*ny + bs.Mt[1];
        double tz = bs.Mn*nz + bs.Mt[2];

        if (newton_bond || i1 < nlocal) {
            f[i1][0] += fx; f[i1][1] += fy; f[i1][2] += fz;
            torque[i1][0] += tx; torque[i1][1] += ty; torque[i1][2] += tz;
        }
        if (newton_bond || i2 < nlocal) {
            f[i2][0] -= fx; f[i2][1] -= fy; f[i2][2] -= fz;
            torque[i2][0] -= tx; torque[i2][1] -= ty; torque[i2][2] -= tz;
        }

        if (evflag) ev_tally(i1,i2,nlocal,newton_bond,0.0,bs.Fn,delx,dely,delz);
    }
}


