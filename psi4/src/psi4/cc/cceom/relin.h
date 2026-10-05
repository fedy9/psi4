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
void relin_fold(const char *sigma2_label, const char *out_label, int C_irr);

/* omega used in the fold denominator (mean of the targeted guess energies). */
double relin_omega_fixed();

/* ---- packed storage for the explicit doubles (see relin.cc) ---- */

/* Is the packed layout in effect? True only when the user asked for it AND a
   packed instance exists for this irrep (it does not when the active space is
   empty, since then there are no explicit doubles to store). Every site that
   opens a stored doubles vector must agree with this. */
bool relin_packed_on();

/* Labels under which the Davidson's doubles vectors live when packed. They are
   deliberately distinct from "CMnEf i" / "SIjAb i": a packed and a full-size
   buffer cannot share a psio key in the same file, and keeping the conventional
   labels free means everything downstream (write_Rs, R0, amplitude printing)
   still finds a full-size vector where it expects one. */
const char *relin_packed_C2_label(int index);
const char *relin_packed_S2_label(int index);

/* Full-size working buffers through which packed vectors are read and written
   by code that operates on whole (ij,ab) blocks -- notably the sigma. Only one
   vector is ever materialised at a time, so the footprint stays at a couple of
   full buffers regardless of how many subspace vectors there are. */
const char *relin_work_C2_label();
const char *relin_work_S2_label();
int relin_work_file();

/* Uniform accessors for the Davidson's stored doubles vectors. These are the
   only thing the solver needs to know about packing: each returns the (file,
   label) to open, materialising the vector into a full-size working buffer
   first when the store is packed. With RELIN_PACKED off -- and for EOM-CCSD and
   EOM-CC3, which never set it -- they hand back the conventional
   "CMnEf i" / "SIjAb i" and cost nothing, so a converted call site behaves
   exactly as before. Keeping the policy in one place is what stops the read and
   write sides drifting apart, the same reason the P/Q masking has one
   predicate. */
void relin_open_C2(int index, int C_irr, int *file, const char **label);
void relin_open_S2(int index, int C_irr, int *file, const char **label);

/* Write a doubles vector back into the store, packing it when packed. */
void relin_put_C2(int index, int C_irr, dpdbuf4 *src);
void relin_put_S2(int index, int C_irr, dpdbuf4 *src);

/* packed vector <-> full-size working buffer */
void relin_load_C2(int index, int C_irr);
void relin_load_S2(int index, int C_irr);
void relin_save_C2(int index, int C_irr);
void relin_save_S2(int index, int C_irr);

/* Run a block of doubles-only DPD work in the packed instance. Restores the
   default instance on every exit path, including exceptions. A buf4_init under
   the wrong instance silently gets wrong offsets, so this is never to be
   replaced by bare dpd_set_default calls. */
struct RelinPackedScope {
    RelinPackedScope();
    ~RelinPackedScope();
    RelinPackedScope(const RelinPackedScope &) = delete;
    RelinPackedScope &operator=(const RelinPackedScope &) = delete;
};

/* Is a packed doubles instance available for this irrep? False when the active
   space is empty, in which case there are no explicit doubles to store. */
bool relin_packed_ready();

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

/* Post-processing, once the Davidson has converged. For each converged root,
   rebuild the eliminated amplitudes at that root's OWN converged eigenvalue
   (not omega_fixed), store the completed vector, and measure it against the
   real, unfolded EOM-CC2 operator. Two things come out of one extra sigma: a
   refined Rayleigh-quotient energy, and a residual norm that exposes a vector
   which is not actually an eigenvector. Must be called before write_Rs, so
   that the amplitudes handed to CC_RAMPS are the complete ones. */
void relin_finalize(int C_irr, double *lambda, const std::vector<bool> &converged);

}  // namespace cceom
}  // namespace psi

#endif  // _psi_src_bin_cceom_relin_h
