/*
    This file is part of ParTI!.

    ParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    ParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with ParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef HIPARTI_H
#define HIPARTI_H

#include <cstddef>
#include <cstdio>
#include <cinttypes>
#include <cmath>
#include <cstdbool>
#include <cstdint>
#ifdef HIPARTI_USE_OPENMP
    #include <omp.h>
#endif
#ifdef HIPARTI_USE_MPI
    #include <mpi.h>
#endif

#ifdef __cplusplus
//extern "C" {
#endif



/*************************************************
 * TYPES
 *************************************************/
#include "includes/types.h"


/*************************************************
 * MACROS
 *************************************************/
#include "includes/macros.h"


/*************************************************
 * STRUCTS
 *************************************************/
#include "includes/structs.h"


/*************************************************
 * HELPER FUNCTIONS
 *************************************************/
#include "includes/helper_funcs.h"

/*************************************************
 * MMIO
 *************************************************/
#include "includes/mmio.h"

/*************************************************
 * ERRORS
 *************************************************/
#include "includes/error.h"

/*************************************************
 * FUNCTIONS
 *************************************************/
/* Vector functions */
#include "includes/vectors.h"
/* Dense matrix functions */
#include "includes/matrices.h"
/* Sparse matrix functions */
#include "includes/spmatrix.h"
/* Sparse tensor functions */
#include "includes/sptensors.h"
/* Semi-sparse tensor functions */
#include "includes/ssptensors.h"
/* Kruskal tensor functions */
#include "includes/ktensors.h"
/* CPD functions */
#include "includes/cpds.h"



#ifdef __cplusplus
//}
#endif

#endif
