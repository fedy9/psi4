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
