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
    relin_packed_setup(C_irr);
    relin_packed_check(C_irr);
    outfile->Printf("\n");

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
   Packed storage for the explicit doubles.

   Masking leaves the eliminated amplitudes as explicit zeros in a full-size
   buffer, so nothing is actually saved. The Davidson subspace is where that
   costs real space: VECS_PER_ROOT vectors per root for C2 and as many again
   for S2, each a full (ij,ab) buffer. Holding those in a buffer dimensioned by
   the ACTIVE space instead shrinks both the footprint and the I/O of every dot
   and axpy over the subspace by the active fraction.

   This works because a DPD instance is described by a count of orbitals per
   irrep, and the active orbitals are contiguous within each irrep: the active
   occupieds are a tail of each irrep's occupied block and the active virtuals a
   head of each virtual block (psi4's QT ordering runs space by space and then
   irrep by irrep, preserving energy order inside an irrep). Selecting the
   active space by orbital energy rather than orbital index is what makes that
   true -- an arbitrary index subset could not be described this way at all.

   A second instance is needed because the singles space stays full while the
   doubles space shrinks, and one instance cannot hold both dimensions. The
   instance is rebuilt per irrep, since the active window depends on that
   irrep's omega_screen. dpd_set_default is global state and a buf4_init under
   the wrong instance silently gets wrong offsets, so every switch goes through
   the RAII guard below and never a bare call.
   ------------------------------------------------------------------------ */
namespace {

constexpr int RELIN_DPD = 1; /* dpd_list has exactly two slots; cceom uses 0 */

bool packed_ready_ = false;
std::vector<int> act_occpi_, act_virtpi_;
std::vector<int> aocc_sym_, avir_sym_;
std::vector<int> occ_p2f_, vir_p2f_; /* packed index -> full absolute index */
int n_act_occ_ = 0, n_act_vir_ = 0;
/* DPD keeps pointers to these, so they must outlive the instance */
std::vector<int> packed_cachefiles_;
int **packed_cachelist_ = nullptr;

/* Switch the default DPD instance for a scope and always switch back. */
struct PackedScope {
    explicit PackedScope(int num) {
        if (!dpd_list[num]) throw PsiException("RELIN: packed DPD instance is not open.", __FILE__, __LINE__);
        dpd_set_default(num);
    }
    ~PackedScope() { dpd_set_default(0); }
    PackedScope(const PackedScope &) = delete;
    PackedScope &operator=(const PackedScope &) = delete;
};

/* Active orbitals must be contiguous within each irrep for a per-irrep count to
   describe them. Fills act_occpi_/act_virtpi_ and reports whether it holds. */
bool packed_contiguity() {
    bool ok = true;
    act_occpi_.assign(moinfo.nirreps, 0);
    act_virtpi_.assign(moinfo.nirreps, 0);

    for (int h = 0; h < moinfo.nirreps; h++) {
        int n = 0, first = -1;
        for (int p = 0; p < moinfo.occpi[h]; p++)
            if (act_occ_[moinfo.occ_off[h] + p]) {
                if (first < 0) first = p;
                ++n;
            }
        act_occpi_[h] = n;
        if (n && first + n != moinfo.occpi[h]) ok = false; /* must be a tail */

        n = 0;
        int last = -1;
        for (int p = 0; p < moinfo.virtpi[h]; p++)
            if (act_vir_[moinfo.vir_off[h] + p]) {
                last = p;
                ++n;
            }
        act_virtpi_[h] = n;
        if (n && last != n - 1) ok = false; /* must be a head */
    }
    return ok;
}

}  // namespace

bool relin_packed_ready() { return packed_ready_; }

void relin_packed_setup(int C_irr) {
    packed_ready_ = false;

    if (!packed_contiguity())
        throw PsiException("RELIN: the active orbitals are not contiguous within an irrep, so they cannot be "
                           "described by a reduced DPD space.",
                           __FILE__, __LINE__);

    n_act_occ_ = 0;
    n_act_vir_ = 0;
    for (int h = 0; h < moinfo.nirreps; h++) {
        n_act_occ_ += act_occpi_[h];
        n_act_vir_ += act_virtpi_[h];
    }

    aocc_sym_.assign(n_act_occ_, 0);
    avir_sym_.assign(n_act_vir_, 0);
    occ_p2f_.assign(n_act_occ_, -1);
    vir_p2f_.assign(n_act_vir_, -1);
    {
        int po = 0, pv = 0;
        for (int h = 0; h < moinfo.nirreps; h++) {
            for (int k = 0; k < act_occpi_[h]; k++, po++) {
                aocc_sym_[po] = h;
                /* active occupieds are the tail of the irrep's occupied block */
                occ_p2f_[po] = moinfo.occ_off[h] + (moinfo.occpi[h] - act_occpi_[h]) + k;
            }
            for (int k = 0; k < act_virtpi_[h]; k++, pv++) {
                avir_sym_[pv] = h;
                /* active virtuals are the head of the irrep's virtual block */
                vir_p2f_[pv] = moinfo.vir_off[h] + k;
            }
        }
    }
    /* the maps must land on orbitals the predicate calls active */
    for (int p = 0; p < n_act_occ_; p++)
        if (!act_occ_[occ_p2f_[p]])
            throw PsiException("RELIN: packed occupied index map is inconsistent.", __FILE__, __LINE__);
    for (int p = 0; p < n_act_vir_; p++)
        if (!act_vir_[vir_p2f_[p]])
            throw PsiException("RELIN: packed virtual index map is inconsistent.", __FILE__, __LINE__);

    if (!n_act_occ_ || !n_act_vir_) {
        /* No explicit doubles at all: there is nothing to store, so skip the
           packed instance rather than building a zero-dimensional one. */
        outfile->Printf("\tpacked doubles storage      = not used (no explicit doubles)\n");
        return;
    }

    Dimension d_occ(moinfo.nirreps), d_vir(moinfo.nirreps);
    for (int h = 0; h < moinfo.nirreps; h++) {
        d_occ[h] = act_occpi_[h];
        d_vir[h] = act_virtpi_[h];
    }

    if (packed_cachefiles_.empty()) packed_cachefiles_.assign(PSIO_MAXUNIT, 0);
    if (!packed_cachelist_) packed_cachelist_ = cacheprep_rhf(0, packed_cachefiles_.data());

    std::vector<std::pair<Dimension, int *>> spaces;
    spaces.emplace_back(d_occ, aocc_sym_.data());
    spaces.emplace_back(d_vir, avir_sym_.data());

    /* The active window is per irrep, so the instance is rebuilt per irrep;
       dpd_init refuses to overwrite a live slot. */
    if (dpd_list[RELIN_DPD]) dpd_close(RELIN_DPD);
    dpd_init(RELIN_DPD, moinfo.nirreps, params.memory, 0, packed_cachefiles_.data(), packed_cachelist_, nullptr,
             spaces);
    dpd_set_default(0); /* dpd_init makes the new instance current; undo that */

    packed_ready_ = true;

    outfile->Printf("\tpacked doubles storage      = active occ");
    for (int h = 0; h < moinfo.nirreps; h++) outfile->Printf(" %d", act_occpi_[h]);
    outfile->Printf(" / vir");
    for (int h = 0; h < moinfo.nirreps; h++) outfile->Printf(" %d", act_virtpi_[h]);
    outfile->Printf("\n");
}

void relin_packed_teardown() {
    if (dpd_list[RELIN_DPD]) dpd_close(RELIN_DPD);
    dpd_set_default(0);
    packed_ready_ = false;
}

void relin_packed_init(int packed_file, const char *packed_label, int C_irr) {
    dpdbuf4 P;
    if (!packed_ready_) throw PsiException("RELIN: packed buffer requested before setup.", __FILE__, __LINE__);

    PackedScope scope(RELIN_DPD);
    global_dpd_->buf4_init(&P, packed_file, C_irr, 0, 5, 0, 5, 0, packed_label);
    global_dpd_->buf4_scm(&P, 0.0);
    global_dpd_->buf4_close(&P);
}

/* Gather the explicit corner of a full-size buffer into a packed one. Iterating
   the PACKED buffer's own index pairs guarantees every packed slot is written
   exactly once, which is the direction that exposes a gap in the maps. */
void relin_pack(int full_file, const char *full_label, int packed_file, const char *packed_label, int C_irr) {
    dpdbuf4 Full, P;
    if (!packed_ready_) throw PsiException("RELIN: relin_pack before setup.", __FILE__, __LINE__);

    for (int h = 0; h < moinfo.nirreps; h++) {
        global_dpd_->buf4_init(&Full, full_file, C_irr, 0, 5, 0, 5, 0, full_label);
        global_dpd_->buf4_mat_irrep_init(&Full, h);
        global_dpd_->buf4_mat_irrep_rd(&Full, h);
        {
            PackedScope scope(RELIN_DPD);
            global_dpd_->buf4_init(&P, packed_file, C_irr, 0, 5, 0, 5, 0, packed_label);
            global_dpd_->buf4_mat_irrep_init(&P, h);
            for (int ij = 0; ij < P.params->rowtot[h]; ij++) {
                int row = Full.params->rowidx[occ_p2f_[P.params->roworb[h][ij][0]]]
                                             [occ_p2f_[P.params->roworb[h][ij][1]]];
                for (int ab = 0; ab < P.params->coltot[h ^ C_irr]; ab++) {
                    int col = Full.params->colidx[vir_p2f_[P.params->colorb[h ^ C_irr][ab][0]]]
                                                 [vir_p2f_[P.params->colorb[h ^ C_irr][ab][1]]];
                    P.matrix[h][ij][ab] = Full.matrix[h][row][col];
                }
            }
            global_dpd_->buf4_mat_irrep_wrt(&P, h);
            global_dpd_->buf4_mat_irrep_close(&P, h);
            global_dpd_->buf4_close(&P);
        }
        global_dpd_->buf4_mat_irrep_close(&Full, h);
        global_dpd_->buf4_close(&Full);
    }
}

/* Scatter a packed buffer back into a full-size one, zero outside the corner. */
void relin_unpack(int packed_file, const char *packed_label, int full_file, const char *full_label, int C_irr) {
    dpdbuf4 Full, P;
    if (!packed_ready_) throw PsiException("RELIN: relin_unpack before setup.", __FILE__, __LINE__);

    for (int h = 0; h < moinfo.nirreps; h++) {
        global_dpd_->buf4_init(&Full, full_file, C_irr, 0, 5, 0, 5, 0, full_label);
        global_dpd_->buf4_mat_irrep_init(&Full, h);
        for (int ij = 0; ij < Full.params->rowtot[h]; ij++)
            for (int ab = 0; ab < Full.params->coltot[h ^ C_irr]; ab++) Full.matrix[h][ij][ab] = 0.0;
        {
            PackedScope scope(RELIN_DPD);
            global_dpd_->buf4_init(&P, packed_file, C_irr, 0, 5, 0, 5, 0, packed_label);
            global_dpd_->buf4_mat_irrep_init(&P, h);
            global_dpd_->buf4_mat_irrep_rd(&P, h);
            for (int ij = 0; ij < P.params->rowtot[h]; ij++) {
                int row = Full.params->rowidx[occ_p2f_[P.params->roworb[h][ij][0]]]
                                             [occ_p2f_[P.params->roworb[h][ij][1]]];
                for (int ab = 0; ab < P.params->coltot[h ^ C_irr]; ab++) {
                    int col = Full.params->colidx[vir_p2f_[P.params->colorb[h ^ C_irr][ab][0]]]
                                                 [vir_p2f_[P.params->colorb[h ^ C_irr][ab][1]]];
                    Full.matrix[h][row][col] = P.matrix[h][ij][ab];
                }
            }
            global_dpd_->buf4_mat_irrep_close(&P, h);
            global_dpd_->buf4_close(&P);
        }
        global_dpd_->buf4_mat_irrep_wrt(&Full, h);
        global_dpd_->buf4_mat_irrep_close(&Full, h);
        global_dpd_->buf4_close(&Full);
    }
}

/* A full buffer restricted to P, packed and unpacked again, must come back
   bit for bit. Catches an index-map gap or an instance mix-up, both of which
   otherwise produce plausible numbers rather than a failure. */
void relin_packed_check(int C_irr) {
    dpdbuf4 A, B;
    if (!packed_ready_) return;

    global_dpd_->buf4_init(&A, PSIF_EOM_TMP, C_irr, 0, 5, 0, 5, 0, "RELIN packcheck full");
    for (int h = 0; h < moinfo.nirreps; h++) {
        global_dpd_->buf4_mat_irrep_init(&A, h);
        for (int ij = 0; ij < A.params->rowtot[h]; ij++) {
            int i = A.params->roworb[h][ij][0];
            int j = A.params->roworb[h][ij][1];
            for (int ab = 0; ab < A.params->coltot[h ^ C_irr]; ab++) {
                int a = A.params->colorb[h ^ C_irr][ab][0];
                int b = A.params->colorb[h ^ C_irr][ab][1];
                A.matrix[h][ij][ab] = relin_active(i, j, a, b) ? 1.0 + 0.001 * i + 0.01 * j + 0.1 * a + b : 0.0;
            }
        }
        global_dpd_->buf4_mat_irrep_wrt(&A, h);
        global_dpd_->buf4_mat_irrep_close(&A, h);
    }
    global_dpd_->buf4_close(&A);

    relin_packed_init(PSIF_EOM_TMP, "RELIN packcheck packed", C_irr);
    relin_pack(PSIF_EOM_TMP, "RELIN packcheck full", PSIF_EOM_TMP, "RELIN packcheck packed", C_irr);
    relin_unpack(PSIF_EOM_TMP, "RELIN packcheck packed", PSIF_EOM_TMP, "RELIN packcheck back", C_irr);

    double maxdev = 0.0;
    global_dpd_->buf4_init(&A, PSIF_EOM_TMP, C_irr, 0, 5, 0, 5, 0, "RELIN packcheck full");
    global_dpd_->buf4_init(&B, PSIF_EOM_TMP, C_irr, 0, 5, 0, 5, 0, "RELIN packcheck back");
    for (int h = 0; h < moinfo.nirreps; h++) {
        global_dpd_->buf4_mat_irrep_init(&A, h);
        global_dpd_->buf4_mat_irrep_rd(&A, h);
        global_dpd_->buf4_mat_irrep_init(&B, h);
        global_dpd_->buf4_mat_irrep_rd(&B, h);
        for (int ij = 0; ij < A.params->rowtot[h]; ij++)
            for (int ab = 0; ab < A.params->coltot[h ^ C_irr]; ab++)
                maxdev = std::max(maxdev, std::fabs(A.matrix[h][ij][ab] - B.matrix[h][ij][ab]));
        global_dpd_->buf4_mat_irrep_close(&B, h);
        global_dpd_->buf4_mat_irrep_close(&A, h);
    }
    global_dpd_->buf4_close(&B);
    global_dpd_->buf4_close(&A);

    outfile->Printf("\tpacked round-trip check     = %5.1e (must be 0)\n", maxdev);
    if (maxdev > 0.0)
        throw PsiException("RELIN: packed/unpacked amplitudes do not round trip exactly.", __FILE__, __LINE__);
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
