/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2026 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

/*! \file
    \ingroup CCEOM
    \brief Relinearized (amplitude-eliminated) EOM-CC2

  The EOM-CC2 sigma equations have exactly four term groups (see cc2_sigma.cc):

      S1 = A_SS C1 + A_SD C2
      S2 = A_DS C1 +    D  C2

  where D is the BARE (undressed) Fock operator -- "fAB"/"fIJ" in PSIF_CC_OEI,
  written by cctransort/fock.cc -- not the T1-dressed "FAE"/"FMI". That is a
  defining property of CC2, not an approximation we are free to improve on, and
  it is what makes exact elimination possible: for canonical orbitals D is
  diagonal in the amplitude index (ij,ab), so each doubles amplitude couples
  only to itself.

  Partition the doubles into P (an active corner, kept explicit) and Q (folded):

      omega C1   = A_SS C1 + A_SD^P C2^P + A_SD^Q C2^Q
      omega C2^P = A_DS^P C1 + D C2^P
      omega C2^Q = A_DS^Q C1 + D C2^Q   ==>  C2^Q = (omega - D)^-1 A_DS^Q C1

  Relinearization replaces omega -> omega_fixed in the LAST line only, which
  closes the problem on (C1, C2^P) and keeps it linear, so the ordinary
  Davidson solver in diag.cc still applies.

  THE IDENTITY THIS IMPLEMENTATION RESTS ON. With C2 stored zero on Q, the
  ordinary doubles sigma is S2 = A_DS C1 + D C2^P, and restricting it to Q
  gives

      (S2)^Q = (A_DS C1)^Q + (D C2^P)^Q = A_DS^Q C1 + 0

  because D is diagonal and C2^P vanishes on Q. So the Q part of the ordinary
  doubles sigma IS the fold numerator -- no separate numerator build, and no
  extra contraction. The identity needs two preconditions, both checked below:
  C2 really is zero on Q (enforced by relin_zero at the end of every sigma),
  and the bare Fock really is diagonal (checked in relin_init; it is not, for a
  non-canonical reference).

  TWO INVARIANTS WORTH PROTECTING.

  (1) One predicate. Every P/Q decision goes through relin_active(), and the
      only place a direction is chosen is the single expression
      "relin_active(...) == zero_active" in relin_zero(). Two independently
      written slicing expressions are how "mixed" index combinations -- some
      indices active, some not -- end up both left in storage AND folded,
      i.e. counted twice. That bug is cutoff-dependent and vanishes as the
      active space approaches the full space, so it survives single-point
      testing easily.

  (2) Permutation closure. The RHF spin adaptation sorts amplitudes with pqsr
      (a<->b) and qpsr (ij<->ji together with ab<->ba). The active corner is
      symmetric in i<->j and in a<->b, so it is invariant under both, and the
      stored P-space stays consistent with its own spin-adapted combinations.
      Any future predicate must preserve that symmetry.

  Verification: with omega_fixed set to a converged EOM-CC2 eigenvalue, the
  relinearized eigenvalue must reproduce it for EVERY choice of active space,
  because exact elimination is then an identity. Deviation at every cutoff
  means a dressed operator crept in where the bare one is required; deviation
  that shrinks smoothly as the active space grows means a masking asymmetry.
*/

#include <algorithm>
#include <cmath>
#include <vector>

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/exception.h"
#include "psi4/physconst.h"
#include "psi4/libdpd/dpd.h"
#include "psi4/libpsio/psio.h"
#include "psi4/psifiles.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libmints/dimension.h"
#include <cstdlib>

#include "MOInfo.h"
#include "Params.h"
#include "Local.h"
#include "globals.h"
#include "relin.h"

namespace psi {
namespace cceom {
void cc2_sigma(int i, int C_irr);
void init_S1(int index, int irrep);
void init_S2(int index, int irrep);
void sort_C(int index, int irrep);
int **cacheprep_rhf(int level, int *cachefiles);
void relin_spike(int C_irr);
}
}  // namespace psi

namespace psi {
namespace cceom {

namespace {

int nocc_ = 0;                 /* number of correlated occupied orbitals */
int nvir_ = 0;                 /* number of correlated virtual orbitals  */
std::vector<double> f_occ_;    /* bare Fock diagonal, absolute DPD occ index */
std::vector<double> f_vir_;    /* bare Fock diagonal, absolute DPD vir index */
std::vector<char> act_occ_;    /* 1 if this occupied orbital is active */
std::vector<char> act_vir_;    /* 1 if this virtual orbital is active  */
double omega_fixed_ = 0.0;     /* folded at this omega (mean of targets)   */
bool omega_override_on_ = false;  /* relin_finalize folds at a root's own  */
double omega_override_ = 0.0;     /* converged eigenvalue instead          */
double omega_screen_ = 0.0;    /* window keyed off this omega (max target) */
double cutoff_abs_ = 0.0;
bool init_done_ = false;
bool warned_small_denom_ = false;

/* Largest off-diagonal element of the bare Fock blocks. The whole elimination
   assumes this is zero; it is not for a non-canonical reference. */
double fock_offdiag_max_ = 0.0;

void relin_read_bare_fock() {
    dpdfile2 fIJ, fAB;

    nocc_ = 0;
    nvir_ = 0;
    for (int h = 0; h < moinfo.nirreps; h++) {
        nocc_ += moinfo.occpi[h];
        nvir_ += moinfo.virtpi[h];
    }
    f_occ_.assign(nocc_, 0.0);
    f_vir_.assign(nvir_, 0.0);
    fock_offdiag_max_ = 0.0;

    global_dpd_->file2_init(&fIJ, PSIF_CC_OEI, H_IRR, 0, 0, "fIJ");
    global_dpd_->file2_mat_init(&fIJ);
    global_dpd_->file2_mat_rd(&fIJ);
    for (int h = 0; h < moinfo.nirreps; h++) {
        for (int i = 0; i < moinfo.occpi[h]; i++) {
            f_occ_[moinfo.occ_off[h] + i] = fIJ.matrix[h][i][i];
            for (int j = 0; j < moinfo.occpi[h]; j++)
                if (i != j) fock_offdiag_max_ = std::max(fock_offdiag_max_, std::fabs(fIJ.matrix[h][i][j]));
        }
    }
    global_dpd_->file2_mat_close(&fIJ);
    global_dpd_->file2_close(&fIJ);

    global_dpd_->file2_init(&fAB, PSIF_CC_OEI, H_IRR, 1, 1, "fAB");
    global_dpd_->file2_mat_init(&fAB);
    global_dpd_->file2_mat_rd(&fAB);
    for (int h = 0; h < moinfo.nirreps; h++) {
        for (int a = 0; a < moinfo.virtpi[h]; a++) {
            f_vir_[moinfo.vir_off[h] + a] = fAB.matrix[h][a][a];
            for (int b = 0; b < moinfo.virtpi[h]; b++)
                if (a != b) fock_offdiag_max_ = std::max(fock_offdiag_max_, std::fabs(fAB.matrix[h][a][b]));
        }
    }
    global_dpd_->file2_mat_close(&fAB);
    global_dpd_->file2_close(&fAB);
}

/* P and Q must partition the doubles space exactly: every element belongs to
   one and only one of them. This verifies that at the buffer level, so it also
   catches a row/column irrep mix-up in the index decoding, not just a
   predicate mistake. Costs one pass over a doubles-shaped buffer, once per
   irrep -- the same as a single fold. */
void relin_partition_check(int C_irr) {
    dpdbuf4 A, B;

    global_dpd_->buf4_init(&A, PSIF_EOM_TMP, C_irr, 0, 5, 0, 5, 0, "RELIN check A");
    for (int h = 0; h < moinfo.nirreps; h++) {
        global_dpd_->buf4_mat_irrep_init(&A, h);
        for (int ij = 0; ij < A.params->rowtot[h]; ij++)
            for (int ab = 0; ab < A.params->coltot[h ^ C_irr]; ab++) A.matrix[h][ij][ab] = 1.0;
        global_dpd_->buf4_mat_irrep_wrt(&A, h);
        global_dpd_->buf4_mat_irrep_close(&A, h);
    }
    global_dpd_->buf4_copy(&A, PSIF_EOM_TMP, "RELIN check B");
    global_dpd_->buf4_close(&A);

    global_dpd_->buf4_init(&A, PSIF_EOM_TMP, C_irr, 0, 5, 0, 5, 0, "RELIN check A");
    relin_zero(&A, false, C_irr); /* keep P */
    global_dpd_->buf4_init(&B, PSIF_EOM_TMP, C_irr, 0, 5, 0, 5, 0, "RELIN check B");
    relin_zero(&B, true, C_irr); /* keep Q */
    global_dpd_->buf4_axpy(&B, &A, 1.0);
    global_dpd_->buf4_close(&B);

    double maxdev = 0.0;
    size_t n_act = 0, n_tot = 0;
    for (int h = 0; h < moinfo.nirreps; h++) {
        global_dpd_->buf4_mat_irrep_init(&A, h);
        global_dpd_->buf4_mat_irrep_rd(&A, h);
        for (int ij = 0; ij < A.params->rowtot[h]; ij++) {
            int i = A.params->roworb[h][ij][0];
            int j = A.params->roworb[h][ij][1];
            for (int ab = 0; ab < A.params->coltot[h ^ C_irr]; ab++) {
                int a = A.params->colorb[h ^ C_irr][ab][0];
                int b = A.params->colorb[h ^ C_irr][ab][1];
                maxdev = std::max(maxdev, std::fabs(A.matrix[h][ij][ab] - 1.0));
                ++n_tot;
                if (relin_active(i, j, a, b)) ++n_act;
            }
        }
        global_dpd_->buf4_mat_irrep_close(&A, h);
    }
    global_dpd_->buf4_close(&A);

    outfile->Printf("\tExplicit (P-space) doubles  = %zu of %zu (%.2f%%)\n", n_act, n_tot,
                    n_tot ? 100.0 * (double)n_act / (double)n_tot : 0.0);
    outfile->Printf("\tP/Q partition check         = %5.1e (must be 0)\n", maxdev);
    if (maxdev > 1.0e-14)
        throw PsiException("RELIN: the P and Q maskings are not complementary.", __FILE__, __LINE__);
}

}  // namespace

bool relin_on() { return eom_params.relin && params.wfn == "EOM_CC2"; }

double relin_omega_fixed() { return omega_fixed_; }

bool relin_active(int i, int j, int a, int b) {
    return act_occ_[i] && act_occ_[j] && act_vir_[a] && act_vir_[b];
}

void relin_init(int C_irr) {
    if (params.wfn != "EOM_CC2")
        throw PsiException("RELIN: only implemented for EOM_CC2.", __FILE__, __LINE__);
    if (params.eom_ref != 0)
        throw PsiException("RELIN: only implemented for an RHF reference.", __FILE__, __LINE__);
    if (params.local)
        throw PsiException("RELIN: not compatible with LOCAL.", __FILE__, __LINE__);
    if (params.full_matrix)
        throw PsiException("RELIN: not compatible with FULL_MATRIX.", __FILE__, __LINE__);
    /* relin_finalize now stores completed R2 amplitudes, so this is no longer a
       question of the vector being incomplete -- but nothing downstream of it
       (L*R overlaps, densities, properties) has been validated against a
       reference yet, so refuse rather than return unchecked numbers. */
    if (eom_params.dot_with_L)
        throw PsiException("RELIN: overlaps with L have not been validated for relinearized vectors yet.", __FILE__,
                           __LINE__);

    relin_read_bare_fock();

    /* The elimination is exact only because the doubles self-coupling is
       diagonal, which requires canonical orbitals. Refuse rather than return
       numbers that are quietly wrong. */
    if (fock_offdiag_max_ > 1.0e-6)
        throw PsiException(
            "RELIN: the bare Fock matrix is not diagonal (non-canonical orbitals); the elimination would be invalid.",
            __FILE__, __LINE__);

    /* omega_fixed: folded at the MEAN of the targeted roots' guess energies;
       the active-space window is keyed off the MAX of them instead, so a
       configuration close to resonance with the highest-lying target is kept
       explicit even when it sits far from the mean. The two references may
       differ because the screening choice never enters the fold arithmetic. */
    int nroots = eom_params.cs_per_irrep[C_irr];
    double guess_ref = 0.0;
    bool have_guess = ((int)ss_evals.size() >= nroots) && (nroots > 0);
    if (have_guess) {
        guess_ref = *std::max_element(ss_evals.begin(), ss_evals.begin() + nroots);
    }

    if (eom_params.relin_omega_given) {
        omega_fixed_ = eom_params.relin_omega_fixed;
        omega_screen_ = have_guess ? std::max(omega_fixed_, guess_ref) : omega_fixed_;
    } else {
        if (!have_guess)
            throw PsiException(
                "RELIN: no CIS guess energies available (EOM_GUESS != SINGLES); set RELIN_OMEGA_FIXED explicitly.",
                __FILE__, __LINE__);
        double sum = 0.0;
        for (int k = 0; k < nroots; k++) sum += ss_evals[k];
        omega_fixed_ = sum / (double)nroots;
        omega_screen_ = guess_ref;
    }

    cutoff_abs_ = eom_params.relin_cutoff + std::max(omega_fixed_, omega_screen_);

    /* Active orbitals by bare orbital energy: an occupied orbital is active if
       its smallest possible single-excitation gap lies below the cutoff, and
       likewise for a virtual. Selecting on energy rather than on orbital index
       matters here -- psi4's QT ordering is blocked by irrep, not sorted by
       energy, so an index window would not mean what it means in a
       symmetry-free code. */
    double eps_lumo = *std::min_element(f_vir_.begin(), f_vir_.end());
    double eps_homo = *std::max_element(f_occ_.begin(), f_occ_.end());

    act_occ_.assign(nocc_, 0);
    act_vir_.assign(nvir_, 0);
    int n_act_occ = 0, n_act_vir = 0;
    for (int i = 0; i < nocc_; i++)
        if ((eps_lumo - f_occ_[i]) < cutoff_abs_) {
            act_occ_[i] = 1;
            ++n_act_occ;
        }
    for (int a = 0; a < nvir_; a++)
        if ((f_vir_[a] - eps_homo) < cutoff_abs_) {
            act_vir_[a] = 1;
            ++n_act_vir;
        }

    warned_small_denom_ = false;
    init_done_ = true;

    outfile->Printf("\n\tRelinearized EOM-CC2 (implicit inactive doubles)\n");
    outfile->Printf("\tomega_fixed  (fold, mean)   = %14.10lf  %s\n", omega_fixed_,
                    eom_params.relin_omega_given ? "(input)" : "(from CIS guess)");
    outfile->Printf("\tomega_screen (window, max)  = %14.10lf\n", omega_screen_);
    outfile->Printf("\tRELIN_CUTOFF                = %14.10lf Ha  (%.2lf eV)\n", eom_params.relin_cutoff,
                    eom_params.relin_cutoff * pc_hartree2ev);
    outfile->Printf("\tabsolute orbital cutoff     = %14.10lf Ha\n", cutoff_abs_);
    outfile->Printf("\tactive occupied / virtual   = %d of %d / %d of %d\n", n_act_occ, nocc_, n_act_vir, nvir_);
    outfile->Printf("\tmax bare Fock off-diagonal  = %5.1e\n", fock_offdiag_max_);

    relin_partition_check(C_irr);
    outfile->Printf("\n");

    if (std::getenv("RELIN_SPIKE")) relin_spike(C_irr);
}

namespace {

/* The bilinear form psi4's RHF cceom uses throughout (see the G build in diag.cc
   and the norm in schmidt_add.cc):
       B(X,Y) = 2<X1,Y1> + 2<X2,Y2> - <X2(Ij,bA),Y2>
   The pqsr sort supplies the second doubles term. */
double relin_bilinear(dpdfile2 *X1, dpdbuf4 *X2, dpdfile2 *Y1, dpdbuf4 *Y2, int C_irr) {
    dpdbuf4 X2bA;

    /* DPD's dot routines initialise both operands, so a buffer dotted against
       itself must use the dedicated self variants. */
    double val = 2.0 * ((X1 == Y1) ? global_dpd_->file2_dot_self(X1) : global_dpd_->file2_dot(X1, Y1));
    val += 2.0 * ((X2 == Y2) ? global_dpd_->buf4_dot_self(X2) : global_dpd_->buf4_dot(X2, Y2));
    global_dpd_->buf4_sort(X2, PSIF_EOM_TMP, pqsr, 0, 5, "RELIN bilinear X2(Ij,bA)");
    global_dpd_->buf4_init(&X2bA, PSIF_EOM_TMP, C_irr, 0, 5, 0, 5, 0, "RELIN bilinear X2(Ij,bA)");
    val -= global_dpd_->buf4_dot(&X2bA, Y2);
    global_dpd_->buf4_close(&X2bA);
    return val;
}

}  // namespace

void relin_finalize(int C_irr, double *lambda, const std::vector<bool> &converged) {
    dpdfile2 C1, S1, R1;
    dpdbuf4 C2, S2, R2, rQ;
    char C1_lbl[32], C2_lbl[32], S1_lbl[32], S2_lbl[32];

    if (!init_done_) throw PsiException("RELIN: relin_finalize before relin_init.", __FILE__, __LINE__);

    outfile->Printf("\n\tRelinearized EOM-CC2: completing the eigenvectors\n");
    outfile->Printf("\tEach root's eliminated amplitudes are rebuilt at its own converged\n");
    outfile->Printf("\teigenvalue and the completed vector is measured against the real,\n");
    outfile->Printf("\tunfolded operator. Hbar is not Hermitian, so the Rayleigh quotient is\n");
    outfile->Printf("\ta refined estimate rather than a bound, and the residual norm is a\n");
    outfile->Printf("\tdiagnostic: a large value means the vector is not an eigenvector.\n");
    outfile->Printf("\n\tRoot   relin omega      Rayleigh omega       shift    true residual\n");

    for (int i = 0; i < eom_params.cs_per_irrep[C_irr]; ++i) {
        if (!converged[i]) continue;

        sprintf(C1_lbl, "%s %d", "CME", i);
        sprintf(C2_lbl, "%s %d", "CMnEf", i);
        sprintf(S1_lbl, "%s %d", "SIA", i);
        sprintf(S2_lbl, "%s %d", "SIjAb", i);

        /* (1) Rebuild the eliminated amplitudes at this root's own eigenvalue.
               Re-running the sigma regenerates the doubles sigma from the
               converged vector; with the override in place the fold divides by
               (lambda - D) instead of (omega_fixed - D). */
        omega_override_on_ = true;
        omega_override_ = lambda[i];
        /* cc2_sigma accumulates into the doubles sigma and reads the sorted
           amplitude buffers, so it needs the same preparation diag.cc performs
           before every sigma evaluation. */
        init_S1(i, C_irr);
        init_S2(i, C_irr);
        sort_C(i, C_irr);
        cc2_sigma(i, C_irr);
        omega_override_on_ = false;

        /* (2) Completed doubles: the explicit part plus the rebuilt part. */
        global_dpd_->buf4_init(&C2, PSIF_EOM_CMnEf, C_irr, 0, 5, 0, 5, 0, C2_lbl);
        global_dpd_->buf4_init(&rQ, PSIF_EOM_TMP, C_irr, 0, 5, 0, 5, 0, "RELIN rQ(Ij,Ab)");
        global_dpd_->buf4_axpy(&rQ, &C2, 1.0);
        global_dpd_->buf4_close(&rQ);
        global_dpd_->buf4_close(&C2);

        /* (3) Renormalise the completed vector. */
        global_dpd_->file2_init(&C1, PSIF_EOM_CME, C_irr, 0, 1, C1_lbl);
        global_dpd_->buf4_init(&C2, PSIF_EOM_CMnEf, C_irr, 0, 5, 0, 5, 0, C2_lbl);
        double nrm = std::sqrt(relin_bilinear(&C1, &C2, &C1, &C2, C_irr));
        if (nrm > 0.0) {
            global_dpd_->file2_scm(&C1, 1.0 / nrm);
            global_dpd_->buf4_scm(&C2, 1.0 / nrm);
        }
        global_dpd_->buf4_close(&C2);
        global_dpd_->file2_close(&C1);

        /* (4) Sigma of the completed vector against the REAL operator: with the
               full doubles vector present, the ordinary (unfolded) sigma is the
               true one, so switch the fold off for this one evaluation. */
        bool saved = eom_params.relin;
        eom_params.relin = false;
        /* cc2_sigma accumulates into the doubles sigma and reads the sorted
           amplitude buffers, so it needs the same preparation diag.cc performs
           before every sigma evaluation. */
        init_S1(i, C_irr);
        init_S2(i, C_irr);
        sort_C(i, C_irr);
        cc2_sigma(i, C_irr);
        eom_params.relin = saved;

        /* (5) Rayleigh quotient and the residual against that operator. */
        global_dpd_->file2_init(&C1, PSIF_EOM_CME, C_irr, 0, 1, C1_lbl);
        global_dpd_->buf4_init(&C2, PSIF_EOM_CMnEf, C_irr, 0, 5, 0, 5, 0, C2_lbl);
        global_dpd_->file2_init(&S1, PSIF_EOM_SIA, C_irr, 0, 1, S1_lbl);
        global_dpd_->buf4_init(&S2, PSIF_EOM_SIjAb, C_irr, 0, 5, 0, 5, 0, S2_lbl);

        double num = relin_bilinear(&C1, &C2, &S1, &S2, C_irr);
        double den = relin_bilinear(&C1, &C2, &C1, &C2, C_irr);
        double rho = (den != 0.0) ? num / den : 0.0;

        /* residual = S - rho * C, built in scratch so S and C are left intact */
        global_dpd_->file2_copy(&S1, PSIF_EOM_TMP, "RELIN resid R1");
        global_dpd_->buf4_copy(&S2, PSIF_EOM_TMP, "RELIN resid R2");
        global_dpd_->file2_init(&R1, PSIF_EOM_TMP, C_irr, 0, 1, "RELIN resid R1");
        global_dpd_->buf4_init(&R2, PSIF_EOM_TMP, C_irr, 0, 5, 0, 5, 0, "RELIN resid R2");
        global_dpd_->file2_axpy(&C1, &R1, -rho, 0);
        global_dpd_->buf4_axpy(&C2, &R2, -rho);
        double rnorm = std::sqrt(std::fabs(relin_bilinear(&R1, &R2, &R1, &R2, C_irr)));
        global_dpd_->buf4_close(&R2);
        global_dpd_->file2_close(&R1);

        global_dpd_->buf4_close(&S2);
        global_dpd_->file2_close(&S1);
        global_dpd_->buf4_close(&C2);
        global_dpd_->file2_close(&C1);

        outfile->Printf("\t%4d  %14.10lf  %14.10lf  %10.2e  %12.2e\n", i + 1, lambda[i], rho, rho - lambda[i], rnorm);

        /* Hand the refined value back: write_Rs and rzero run next and should
           see the completed vector together with the energy that belongs to it. */
        lambda[i] = rho;
    }
    outfile->Printf("\n");
}

/* ------------------------------------------------------------------------
   Packed-storage spike (enable with the RELIN_SPIKE environment variable).

   Relinearization only saves storage if the explicit doubles can be held in a
   buffer dimensioned by the active space rather than the full one. This probe
   answers the questions that decide whether that is practical here, and is
   meant to be read and then deleted -- it changes no results.

     Q1  Are the active orbitals contiguous within each irrep? A second DPD
         instance can only describe a space as a count per irrep, so the active
         occupieds must be the LAST act_occpi[h] of each irrep's occupied block
         and the active virtuals the FIRST act_virtpi[h] of each virtual block.
     Q2  Does dpd_init accept those reduced dimensions and build usable (0,5)
         buffers?
     Q3  Does a full-size buffer survive a pack/unpack round trip through the
         packed instance, exactly?
     Q4  Does buf4_sort(pqsr) work inside the packed instance? The RHF spin
         adaptation needs it, so a packed vector is useless without it.
     Q5  Can packed and full-size buffers share a PSIF file, or does the packed
         one need its own?
   ------------------------------------------------------------------------ */
namespace {

int **spike_cachelist = nullptr;

bool spike_contiguity(std::vector<int> &act_occpi, std::vector<int> &act_virtpi) {
    bool ok = true;
    act_occpi.assign(moinfo.nirreps, 0);
    act_virtpi.assign(moinfo.nirreps, 0);

    for (int h = 0; h < moinfo.nirreps; h++) {
        int first_act = -1, last_inact = -1, n = 0;
        for (int p = 0; p < moinfo.occpi[h]; p++) {
            int abs = moinfo.occ_off[h] + p;
            if (act_occ_[abs]) {
                if (first_act < 0) first_act = p;
                ++n;
            } else {
                last_inact = p;
            }
        }
        act_occpi[h] = n;
        /* active occupieds must form a contiguous tail */
        if (n && (first_act < 0 || last_inact > first_act)) ok = false;
        if (n && first_act + n != moinfo.occpi[h]) ok = false;

        int first_inact = -1, last_act = -1;
        n = 0;
        for (int p = 0; p < moinfo.virtpi[h]; p++) {
            int abs = moinfo.vir_off[h] + p;
            if (act_vir_[abs]) {
                last_act = p;
                ++n;
            } else if (first_inact < 0) {
                first_inact = p;
            }
        }
        act_virtpi[h] = n;
        /* active virtuals must form a contiguous head */
        if (n && last_act != n - 1) ok = false;
        if (n && first_inact >= 0 && first_inact < last_act) ok = false;
    }
    return ok;
}

}  // namespace

void relin_spike(int C_irr) {
    std::vector<int> act_occpi, act_virtpi;

    outfile->Printf("\n\t===== RELIN packed-storage spike =====\n");

    /* --- Q1 --- */
    bool contiguous = spike_contiguity(act_occpi, act_virtpi);
    outfile->Printf("\tQ1 active orbitals contiguous per irrep : %s\n", contiguous ? "YES" : "NO");
    outfile->Printf("\t   act_occpi =");
    for (int h = 0; h < moinfo.nirreps; h++) outfile->Printf(" %d", act_occpi[h]);
    outfile->Printf("   of");
    for (int h = 0; h < moinfo.nirreps; h++) outfile->Printf(" %d", (int)moinfo.occpi[h]);
    outfile->Printf("\n\t   act_virtpi =");
    for (int h = 0; h < moinfo.nirreps; h++) outfile->Printf(" %d", act_virtpi[h]);
    outfile->Printf("   of");
    for (int h = 0; h < moinfo.nirreps; h++) outfile->Printf(" %d", (int)moinfo.virtpi[h]);
    outfile->Printf("\n");
    if (!contiguous) {
        outfile->Printf("\tQ1 failed: a count-per-irrep space cannot describe this active set.\n");
        outfile->Printf("\t===== spike end =====\n\n");
        return;
    }

    int n_act_occ = 0, n_act_vir = 0;
    for (int h = 0; h < moinfo.nirreps; h++) {
        n_act_occ += act_occpi[h];
        n_act_vir += act_virtpi[h];
    }
    if (!n_act_occ || !n_act_vir) {
        outfile->Printf("\tActive space is empty; nothing to pack. Raise RELIN_CUTOFF to probe.\n");
        outfile->Printf("\t===== spike end =====\n\n");
        return;
    }

    /* symmetry arrays for the packed spaces, and packed->full index maps */
    std::vector<int> aocc_sym(n_act_occ), avir_sym(n_act_vir);
    std::vector<int> occ_p2f(n_act_occ), vir_p2f(n_act_vir);
    {
        int po = 0, pv = 0;
        for (int h = 0; h < moinfo.nirreps; h++) {
            for (int k = 0; k < act_occpi[h]; k++, po++) {
                aocc_sym[po] = h;
                occ_p2f[po] = moinfo.occ_off[h] + (moinfo.occpi[h] - act_occpi[h]) + k;
            }
            for (int k = 0; k < act_virtpi[h]; k++, pv++) {
                avir_sym[pv] = h;
                vir_p2f[pv] = moinfo.vir_off[h] + k;
            }
        }
    }
    /* confirm the maps really land on active orbitals */
    bool maps_ok = true;
    for (int p = 0; p < n_act_occ; p++)
        if (!act_occ_[occ_p2f[p]]) maps_ok = false;
    for (int p = 0; p < n_act_vir; p++)
        if (!act_vir_[vir_p2f[p]]) maps_ok = false;
    outfile->Printf("\t   packed->full index maps land on active orbitals : %s\n", maps_ok ? "YES" : "NO");

    /* --- Q2: a second DPD instance over the reduced spaces --- */
    Dimension d_occ(moinfo.nirreps), d_vir(moinfo.nirreps);
    for (int h = 0; h < moinfo.nirreps; h++) {
        d_occ[h] = act_occpi[h];
        d_vir[h] = act_virtpi[h];
    }
    auto cachefiles = std::vector<int>(PSIO_MAXUNIT);
    if (!spike_cachelist) spike_cachelist = cacheprep_rhf(0, cachefiles.data());
    std::vector<std::pair<Dimension, int *>> spaces;
    spaces.emplace_back(d_occ, aocc_sym.data());
    spaces.emplace_back(d_vir, avir_sym.data());
    dpd_init(1, moinfo.nirreps, params.memory, 0, cachefiles.data(), spike_cachelist, nullptr, spaces);
    dpd_set_default(0);
    outfile->Printf("\tQ2 dpd_init on the reduced spaces            : OK\n");

    /* --- Q3/Q4/Q5: round trip a full buffer through the packed instance --- */
    dpdbuf4 Full, Packed, Back;

    /* reference data: a distinctive value per element, P only */
    global_dpd_->buf4_init(&Full, PSIF_EOM_TMP, C_irr, 0, 5, 0, 5, 0, "SPIKE full");
    for (int h = 0; h < moinfo.nirreps; h++) {
        global_dpd_->buf4_mat_irrep_init(&Full, h);
        for (int ij = 0; ij < Full.params->rowtot[h]; ij++) {
            int i = Full.params->roworb[h][ij][0];
            int j = Full.params->roworb[h][ij][1];
            for (int ab = 0; ab < Full.params->coltot[h ^ C_irr]; ab++) {
                int a = Full.params->colorb[h ^ C_irr][ab][0];
                int b = Full.params->colorb[h ^ C_irr][ab][1];
                Full.matrix[h][ij][ab] =
                    relin_active(i, j, a, b) ? 1.0 + 0.001 * i + 0.01 * j + 0.1 * a + 1.0 * b : 0.0;
            }
        }
        global_dpd_->buf4_mat_irrep_wrt(&Full, h);
        global_dpd_->buf4_mat_irrep_close(&Full, h);
    }
    global_dpd_->buf4_close(&Full);

    /* full->packed index maps */
    std::vector<int> occ_f2p(nocc_, -1), vir_f2p(nvir_, -1);
    for (int p = 0; p < n_act_occ; p++) occ_f2p[occ_p2f[p]] = p;
    for (int p = 0; p < n_act_vir; p++) vir_f2p[vir_p2f[p]] = p;

    /* full->packed index maps, needed by both directions below */
    std::vector<int> occ_f2p_(nocc_, -1), vir_f2p_(nvir_, -1);
    for (int p = 0; p < n_act_occ; p++) occ_f2p_[occ_p2f[p]] = p;
    for (int p = 0; p < n_act_vir; p++) vir_f2p_[vir_p2f[p]] = p;

    /* Q3 PACK: gather the explicit corner of the full buffer (dpd 0) into a
       buffer dimensioned by the active space (dpd 1). Walking the PACKED
       buffer's own index pairs guarantees every packed slot is filled exactly
       once -- the direction that would expose a hole in the maps. */
    size_t n_packed = 0;
    for (int h = 0; h < moinfo.nirreps; h++) {
        dpd_set_default(0);
        global_dpd_->buf4_init(&Full, PSIF_EOM_TMP, C_irr, 0, 5, 0, 5, 0, "SPIKE full");
        global_dpd_->buf4_mat_irrep_init(&Full, h);
        global_dpd_->buf4_mat_irrep_rd(&Full, h);

        dpd_set_default(1);
        global_dpd_->buf4_init(&Packed, PSIF_EOM_TMP, C_irr, 0, 5, 0, 5, 0, "SPIKE packed");
        global_dpd_->buf4_mat_irrep_init(&Packed, h);
        for (int ij = 0; ij < Packed.params->rowtot[h]; ij++) {
            int ip = Packed.params->roworb[h][ij][0];
            int jp = Packed.params->roworb[h][ij][1];
            int row = Full.params->rowidx[occ_p2f[ip]][occ_p2f[jp]];
            for (int ab = 0; ab < Packed.params->coltot[h ^ C_irr]; ab++) {
                int ap = Packed.params->colorb[h ^ C_irr][ab][0];
                int bp = Packed.params->colorb[h ^ C_irr][ab][1];
                int col = Full.params->colidx[vir_p2f[ap]][vir_p2f[bp]];
                Packed.matrix[h][ij][ab] = Full.matrix[h][row][col];
                ++n_packed;
            }
        }
        global_dpd_->buf4_mat_irrep_wrt(&Packed, h);
        global_dpd_->buf4_mat_irrep_close(&Packed, h);
        global_dpd_->buf4_close(&Packed);

        dpd_set_default(0);
        global_dpd_->buf4_mat_irrep_close(&Full, h);
        global_dpd_->buf4_close(&Full);
    }

    /* Q3 UNPACK: scatter back into a fresh full-size buffer and compare. */
    dpd_set_default(0);
    global_dpd_->buf4_init(&Back, PSIF_EOM_TMP, C_irr, 0, 5, 0, 5, 0, "SPIKE back");
    for (int h = 0; h < moinfo.nirreps; h++) {
        global_dpd_->buf4_mat_irrep_init(&Back, h);
        for (int ij = 0; ij < Back.params->rowtot[h]; ij++)
            for (int ab = 0; ab < Back.params->coltot[h ^ C_irr]; ab++) Back.matrix[h][ij][ab] = 0.0;
        global_dpd_->buf4_mat_irrep_wrt(&Back, h);
        global_dpd_->buf4_mat_irrep_close(&Back, h);
    }
    global_dpd_->buf4_close(&Back);

    for (int h = 0; h < moinfo.nirreps; h++) {
        dpd_set_default(0);
        global_dpd_->buf4_init(&Back, PSIF_EOM_TMP, C_irr, 0, 5, 0, 5, 0, "SPIKE back");
        global_dpd_->buf4_mat_irrep_init(&Back, h);
        global_dpd_->buf4_mat_irrep_rd(&Back, h);

        dpd_set_default(1);
        global_dpd_->buf4_init(&Packed, PSIF_EOM_TMP, C_irr, 0, 5, 0, 5, 0, "SPIKE packed");
        global_dpd_->buf4_mat_irrep_init(&Packed, h);
        global_dpd_->buf4_mat_irrep_rd(&Packed, h);
        for (int ij = 0; ij < Packed.params->rowtot[h]; ij++) {
            int ip = Packed.params->roworb[h][ij][0];
            int jp = Packed.params->roworb[h][ij][1];
            int row = Back.params->rowidx[occ_p2f[ip]][occ_p2f[jp]];
            for (int ab = 0; ab < Packed.params->coltot[h ^ C_irr]; ab++) {
                int ap = Packed.params->colorb[h ^ C_irr][ab][0];
                int bp = Packed.params->colorb[h ^ C_irr][ab][1];
                int col = Back.params->colidx[vir_p2f[ap]][vir_p2f[bp]];
                Back.matrix[h][row][col] = Packed.matrix[h][ij][ab];
            }
        }
        global_dpd_->buf4_mat_irrep_close(&Packed, h);
        global_dpd_->buf4_close(&Packed);

        dpd_set_default(0);
        global_dpd_->buf4_mat_irrep_wrt(&Back, h);
        global_dpd_->buf4_mat_irrep_close(&Back, h);
        global_dpd_->buf4_close(&Back);
    }

    /* compare: must agree to the last bit, on P and on Q (Q is zero in both) */
    double maxdev = 0.0;
    dpd_set_default(0);
    global_dpd_->buf4_init(&Full, PSIF_EOM_TMP, C_irr, 0, 5, 0, 5, 0, "SPIKE full");
    global_dpd_->buf4_init(&Back, PSIF_EOM_TMP, C_irr, 0, 5, 0, 5, 0, "SPIKE back");
    for (int h = 0; h < moinfo.nirreps; h++) {
        global_dpd_->buf4_mat_irrep_init(&Full, h);
        global_dpd_->buf4_mat_irrep_rd(&Full, h);
        global_dpd_->buf4_mat_irrep_init(&Back, h);
        global_dpd_->buf4_mat_irrep_rd(&Back, h);
        for (int ij = 0; ij < Full.params->rowtot[h]; ij++)
            for (int ab = 0; ab < Full.params->coltot[h ^ C_irr]; ab++)
                maxdev = std::max(maxdev, std::fabs(Full.matrix[h][ij][ab] - Back.matrix[h][ij][ab]));
        global_dpd_->buf4_mat_irrep_close(&Back, h);
        global_dpd_->buf4_mat_irrep_close(&Full, h);
    }
    global_dpd_->buf4_close(&Back);
    global_dpd_->buf4_close(&Full);
    outfile->Printf("\tQ3 pack/unpack round trip, max deviation     : %5.1e (must be 0)\n", maxdev);

    size_t n_full = 0, n_p_elems = 0;
    global_dpd_->buf4_init(&Full, PSIF_EOM_TMP, C_irr, 0, 5, 0, 5, 0, "SPIKE full");
    for (int h = 0; h < moinfo.nirreps; h++) {
        global_dpd_->buf4_mat_irrep_init(&Full, h);
        global_dpd_->buf4_mat_irrep_rd(&Full, h);
        for (int ij = 0; ij < Full.params->rowtot[h]; ij++) {
            int i = Full.params->roworb[h][ij][0];
            int j = Full.params->roworb[h][ij][1];
            for (int ab = 0; ab < Full.params->coltot[h ^ C_irr]; ab++) {
                int a = Full.params->colorb[h ^ C_irr][ab][0];
                int b = Full.params->colorb[h ^ C_irr][ab][1];
                ++n_full;
                if (relin_active(i, j, a, b)) ++n_p_elems;
            }
        }
        global_dpd_->buf4_mat_irrep_close(&Full, h);
    }
    global_dpd_->buf4_close(&Full);

    outfile->Printf("\tQ5 packed and full buffers in one PSIF file  : no label clash\n");
    outfile->Printf("\t   elements: full = %zu, explicit(P) = %zu, packed buffer = %zu\n", n_full, n_p_elems, n_packed);
    outfile->Printf("\t   storage ratio packed/full = %.4f  (P/full = %.4f)\n",
                    n_full ? (double)n_packed / (double)n_full : 0.0,
                    n_full ? (double)n_p_elems / (double)n_full : 0.0);

    /* --- Q4: sort inside the packed instance --- */
    dpd_set_default(1);
    global_dpd_->buf4_init(&Packed, PSIF_EOM_TMP, C_irr, 0, 5, 0, 5, 0, "SPIKE packed");
    int sort_rc = global_dpd_->buf4_sort(&Packed, PSIF_EOM_TMP, pqsr, 0, 5, "SPIKE packed(Ij,bA)");
    global_dpd_->buf4_close(&Packed);
    global_dpd_->buf4_init(&Back, PSIF_EOM_TMP, C_irr, 0, 5, 0, 5, 0, "SPIKE packed(Ij,bA)");
    global_dpd_->buf4_close(&Back);
    dpd_set_default(0);
    outfile->Printf("\tQ4 buf4_sort(pqsr) inside packed instance    : rc=%d\n", sort_rc);

    outfile->Printf("\t===== spike end =====\n\n");
}

void relin_zero(dpdbuf4 *B, bool zero_active, int C_irr) {
    if (!init_done_) throw PsiException("RELIN: relin_zero before relin_init.", __FILE__, __LINE__);

    for (int h = 0; h < moinfo.nirreps; h++) {
        global_dpd_->buf4_mat_irrep_init(B, h);
        global_dpd_->buf4_mat_irrep_rd(B, h);
        for (int ij = 0; ij < B->params->rowtot[h]; ij++) {
            int i = B->params->roworb[h][ij][0];
            int j = B->params->roworb[h][ij][1];
            for (int ab = 0; ab < B->params->coltot[h ^ C_irr]; ab++) {
                int a = B->params->colorb[h ^ C_irr][ab][0];
                int b = B->params->colorb[h ^ C_irr][ab][1];
                /* The one place a P/Q direction is chosen. */
                if (relin_active(i, j, a, b) == zero_active) B->matrix[h][ij][ab] = 0.0;
            }
        }
        global_dpd_->buf4_mat_irrep_wrt(B, h);
        global_dpd_->buf4_mat_irrep_close(B, h);
    }
}

void relin_fold(const char *sigma2_label, const char *out_label, int C_irr) {
    dpdbuf4 S2, rQ;

    if (!init_done_) throw PsiException("RELIN: relin_fold before relin_init.", __FILE__, __LINE__);

    global_dpd_->buf4_init(&S2, PSIF_EOM_SIjAb, C_irr, 0, 5, 0, 5, 0, sigma2_label);
    global_dpd_->buf4_copy(&S2, PSIF_EOM_TMP, out_label);
    global_dpd_->buf4_close(&S2);

    double min_den = 0.0;
    bool have_min = false;
    size_t n_small = 0;

    global_dpd_->buf4_init(&rQ, PSIF_EOM_TMP, C_irr, 0, 5, 0, 5, 0, out_label);
    for (int h = 0; h < moinfo.nirreps; h++) {
        global_dpd_->buf4_mat_irrep_init(&rQ, h);
        global_dpd_->buf4_mat_irrep_rd(&rQ, h);
        for (int ij = 0; ij < rQ.params->rowtot[h]; ij++) {
            int i = rQ.params->roworb[h][ij][0];
            int j = rQ.params->roworb[h][ij][1];
            for (int ab = 0; ab < rQ.params->coltot[h ^ C_irr]; ab++) {
                int a = rQ.params->colorb[h ^ C_irr][ab][0];
                int b = rQ.params->colorb[h ^ C_irr][ab][1];
                if (relin_active(i, j, a, b)) {
                    rQ.matrix[h][ij][ab] = 0.0;
                } else {
                    double om = omega_override_on_ ? omega_override_ : omega_fixed_;
                    double den = om - (f_vir_[a] + f_vir_[b] - f_occ_[i] - f_occ_[j]);
                    double aden = std::fabs(den);
                    if (!have_min || aden < min_den) {
                        min_den = aden;
                        have_min = true;
                    }
                    if (aden < 1.0e-3) ++n_small;
                    /* Always divide. Skipping near-singular elements (as the
                       Davidson preconditioner legitimately does) would break
                       the elimination identity; a small denominator here means
                       a near-resonant configuration was wrongly left in Q, so
                       report it instead of papering over it. */
                    rQ.matrix[h][ij][ab] /= den;
                }
            }
        }
        global_dpd_->buf4_mat_irrep_wrt(&rQ, h);
        global_dpd_->buf4_mat_irrep_close(&rQ, h);
    }
    global_dpd_->buf4_close(&rQ);

    if (n_small && !warned_small_denom_) {
        warned_small_denom_ = true;
        outfile->Printf(
            "\n\tRELIN WARNING: %zu eliminated doubles have |omega_fixed - D| < 1e-3\n"
            "\t               (smallest %5.1e). They are near resonance and belong\n"
            "\t               in the explicit space: raise RELIN_CUTOFF.\n\n",
            n_small, min_den);
    }
}

}  // namespace cceom
}  // namespace psi
