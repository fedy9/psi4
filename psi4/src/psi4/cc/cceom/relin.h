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
    \brief Relinearized (amplitude-eliminated) EOM-CC2: public interface
*/

#ifndef _psi_src_bin_cceom_relin_h
#define _psi_src_bin_cceom_relin_h

#include <vector>

#include "psi4/libdpd/dpd.h"

namespace psi {
namespace cceom {

/* Is the relinearized treatment active for this calculation? */
bool relin_on();

/* Build the active-space window and omega_fixed for transition irrep C_irr.
   Must be called after diagSS() (which supplies the CIS guess energies) and
   before the first sigma evaluation of that irrep. */
void relin_init(int C_irr);

/* THE active-space predicate. Every P/Q decision in the code must go through
   this one function -- see the comment block at the top of relin.cc. */
bool relin_active(int i, int j, int a, int b);

/* Zero part of a doubles-shaped (0,5) buffer:
     zero_active == false  ->  keep P (zero everything outside the active corner)
     zero_active == true   ->  keep Q (zero the active corner itself)
   The two directions are exact logical negations of one another, by construction. */
void relin_zero(dpdbuf4 *B, bool zero_active, int C_irr);

/* Fold the Q-space doubles: read the doubles sigma stored under label
   sigma2_label in PSIF_EOM_SIjAb, restrict it to Q, divide by
   (omega_fixed - D_bare), and leave the result in PSIF_EOM_TMP under
   out_label. Valid ONLY if the C2 that produced that sigma was zero on Q. */
void relin_fold(int sigma2_file, const char *sigma2_label, const char *out_label, int C_irr, int index);

/* B-norm of the eliminated amplitudes belonging to trial vector `index`. Summed
   with |alpha| weights these bound the eigenvector's eliminated B-norm. */
double relin_vector_q_norm(int index);

/* Carry the per-vector eliminated norms through a subspace collapse, where new
   vector i = sum_j alpha[j][i] * old vector j and no sigma is recomputed. */
void relin_collapse_q_norms(double **alpha, int L, int num);

/* ---- packed storage for the explicit doubles (see relin.cc) ---- */

/* Is the packed layout in effect? True only when RELIN_PACKED is set AND a
   packed instance exists for this irrep (it does not when the active space is
   empty). */
bool relin_packed_on();

/* Labels of the packed Davidson doubles vectors. Distinct from "CMnEf i" /
   "SIjAb i" because a packed and a full-size buffer cannot share a psio key, and
   downstream code (write_Rs, R0, printing) expects "CMnEf i" to be full size. */
const char *relin_packed_C2_label(int index);
const char *relin_packed_S2_label(int index);

/* Full-size working buffers through which packed vectors are read and written
   by code that needs whole (ij,ab) blocks, notably the sigma. */
const char *relin_work_C2_label();
const char *relin_work_S2_label();
/* Scratch for accumulating into one vector while reading others through the
   working buffers. */
const char *relin_acc_label();
int relin_work_file();

/* Accessors for the Davidson's stored doubles vectors: return the (file, label)
   to open, unpacking into a full-size working buffer first when the store is
   packed. Otherwise they return the conventional "CMnEf i" / "SIjAb i", so
   converted call sites behave as before for every method. */
void relin_open_C2(int index, int C_irr, int *file, const char **label);
void relin_open_S2(int index, int C_irr, int *file, const char **label);

/* Write a doubles vector back into the store, packing it when packed. */
void relin_put_C2(int index, int C_irr, dpdbuf4 *src);
void relin_put_S2(int index, int C_irr, dpdbuf4 *src);

/* Pin the working buffers: while held, the accessors return them without
   reloading and the save calls do nothing. Used for reconstructed eigenvectors,
   whose amplitudes outside the active corner the packed store would discard. */
void relin_work_hold(bool on);

/* packed vector <-> full-size working buffer */
void relin_load_C2(int index, int C_irr);
void relin_load_S2(int index, int C_irr);
void relin_save_C2(int index, int C_irr);
void relin_save_S2(int index, int C_irr);

/* Make the packed DPD instance the default for a scope, restoring instance 0 on
   every exit path. A buf4_init under the wrong instance silently gets wrong
   offsets, so never switch with bare dpd_set_default calls. */
struct RelinPackedScope {
    RelinPackedScope();
    ~RelinPackedScope();
    RelinPackedScope(const RelinPackedScope &) = delete;
    RelinPackedScope &operator=(const RelinPackedScope &) = delete;
};

/* Build / tear down the reduced DPD instance describing the active space. The
   instance is per irrep, because the active window is. */
void relin_packed_setup(int C_irr);
void relin_packed_teardown();

/* Create (and zero) a packed doubles buffer. */
void relin_packed_init(int packed_file, const char *packed_label, int C_irr);

/* Move amplitudes between a full-size buffer and a packed one. relin_pack
   gathers the explicit corner; relin_unpack scatters it back and zeroes the
   eliminated part. */
void relin_pack(int full_file, const char *full_label, int packed_file, const char *packed_label, int C_irr);
void relin_unpack(int packed_file, const char *packed_label, int full_file, const char *full_label, int C_irr);

/* Self-test: a P-restricted buffer must survive pack/unpack bit for bit. */
void relin_packed_check(int C_irr);

/* Post-processing (RELIN_FINALIZE), once the Davidson has converged; each must
   precede write_Rs. */

/* RELIN_FINALIZE: rebuild each converged root's eliminated amplitudes at its own
   eigenvalue, renormalise, and store the completed vector; then replace each
   eigenvalue with the completed vector's Rayleigh quotient with the unfolded
   operator and report its residual norm. One sigma per root, which determines
   both. */
void relin_reconstruct(int C_irr, double *lambda, const std::vector<bool> &converged);

/* Used instead of relin_reconstruct when RELIN_FINALIZE is off: writes the
   explicit amplitudes to the full-size "CMnEf i" and reports the bound on the
   eliminated share. */
void relin_export_pspace(int C_irr, const double *lambda_hint, const double *share_bound,
                         const std::vector<bool> &converged);

}  // namespace cceom
}  // namespace psi

#endif  // _psi_src_bin_cceom_relin_h
