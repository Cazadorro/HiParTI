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

#include <HiParTI.h>
#include <stdlib.h>
#include <string.h>

int ptiNewKruskalTensor(ptiKruskalTensor *ktsr, ptiIndex nmodes, const ptiIndex ndims[], ptiIndex rank)
{
    ktsr->nmodes = nmodes;
    ktsr->rank = rank;
    ktsr->ndims = (ptiIndex*)malloc(nmodes*sizeof(ptiIndex));
    for(ptiIndex i=0; i<nmodes; ++i)
        ktsr->ndims[i] = ndims[i];
    ktsr->lambda = (ptiValue*)malloc(rank*sizeof(ptiValue));
    ktsr->fit = 0.0;
    
	return 0;
}


/**
 * Shuffle factor matrices row indices.
 *
 * @param[in] ktsr Kruskal tensor to be shuffled
 * @param[out] map_inds is the renumbering mapping 
 *
 */
void ptiKruskalTensorInverseShuffleIndices(ptiKruskalTensor * ktsr, ptiIndex ** map_inds) {
    /* Renumber factor matrices rows */
    ptiIndex new_i;
    for(ptiIndex m=0; m < ktsr->nmodes; ++m) {
        ptiMatrix * mtx = ktsr->factors[m];
        ptiIndex * mode_map_inds = map_inds[m];
        ptiValue * tmp_values = reinterpret_cast<ptiValue *>(malloc(mtx->cap * mtx->stride * sizeof (ptiValue)));

        for(ptiIndex i=0; i<mtx->nrows; ++i) {
            new_i = mode_map_inds[i];
            for(ptiIndex j=0; j<mtx->ncols; ++j) {
                tmp_values[i * mtx->stride + j] = mtx->values[new_i * mtx->stride + j];
            }
        }
        free(mtx->values);
        mtx->values = tmp_values;
    }    
}


void ptiFreeKruskalTensor(ptiKruskalTensor *ktsr)
{
	ktsr->rank = 0;
	ktsr->fit = 0.0;
	free(ktsr->ndims);
	free(ktsr->lambda);
	for(ptiIndex i=0; i<ktsr->nmodes; ++i)
		ptiFreeMatrix(ktsr->factors[i]);
    free(ktsr->factors);
	ktsr->nmodes = 0;
}


double KruskalTensorFit(
  ptiSparseTensor const * const ptien,
  ptiValue const * const __restrict lambda,
  ptiMatrix ** mats,
  ptiMatrix ** ata)
{
  ptiIndex const nmodes = ptien->nmodes;

  double ptien_normsq = SparseTensorFrobeniusNormSquared(ptien);
  // printf("ptien_normsq: %lf\n", ptien_normsq);
  double const norm_mats = KruskalTensorFrobeniusNormSquared(nmodes, lambda, ata);
  // printf("norm_mats: %lf\n", norm_mats);
  double const inner = SparseKruskalTensorInnerProduct(nmodes, lambda, mats);
  // printf("inner: %lf\n", inner);
  double residual = ptien_normsq + norm_mats - 2 * inner;
  // printf("residual: %lf\n", residual);
  if (residual > 0.0) {
    residual = sqrt(residual);
  }
  double fit = 1 - (residual / sqrt(ptien_normsq));

  return fit;
}



// Column-major. 
/* Compute a Kruskal tensor's norm is compute on "ata"s. Check Tammy's sparse  */
double KruskalTensorFrobeniusNormSquared(
  ptiIndex const nmodes,
  ptiValue const * const __restrict lambda,
  ptiMatrix ** ata) // ata: column-major
{
  ptiIndex const rank = ata[0]->ncols;
  ptiIndex const stride = ata[0]->stride;
  ptiValue * const __restrict tmp_atavals = ata[nmodes]->values;    // Column-major
  double norm_mats = 0;

#ifdef HIPARTI_USE_OPENMP
  #pragma omp parallel for
#endif
  for(ptiIndex x=0; x < rank*stride; ++x) {
    tmp_atavals[x] = 1.;
  }

  /* Compute Hadamard product for all "ata"s */
  for(ptiIndex m=0; m < nmodes; ++m) {
    ptiValue const * const __restrict atavals = ata[m]->values;
#ifdef HIPARTI_USE_OPENMP
  #pragma omp parallel for
#endif
    for(ptiIndex i=0; i < rank; ++i) {
        for(ptiIndex j=i; j < rank; ++j) {
            tmp_atavals[j * stride + i] *= atavals[j * stride + i];
        }
    }
  }

  /* compute lambda^T * aTa[MAX_NMODES] * lambda, only compute a half of them because of its symmetric */
#ifdef HIPARTI_USE_OPENMP
  #pragma omp parallel for reduction(+:norm_mats)
#endif
  for(ptiIndex i=0; i < rank; ++i) {
    norm_mats += tmp_atavals[i+(i*stride)] * lambda[i] * lambda[i];
    for(ptiIndex j=i+1; j < rank; ++j) {
      norm_mats += tmp_atavals[i+(j*stride)] * lambda[i] * lambda[j] * 2;
    }
  }

  return fabs(norm_mats);
}



// Row-major, compute via MTTKRP result (mats[nmodes]) and mats[nmodes-1].
double SparseKruskalTensorInnerProduct(
  ptiIndex const nmodes,
  ptiValue const * const __restrict lambda,
  ptiMatrix ** mats)
{
  ptiIndex const rank = mats[0]->ncols;
  ptiIndex const stride = mats[0]->stride;
  ptiIndex const last_mode = nmodes - 1;
  ptiIndex const I = mats[last_mode]->nrows;

  // printf("mats[nmodes-1]:\n");
  // ptiDumpMatrix(mats[nmodes-1], stdout);
  // printf("mats[nmodes]:\n");
  // ptiDumpMatrix(mats[nmodes], stdout);
  
  ptiValue const * const last_vals = mats[last_mode]->values;
  ptiValue const * const tmp_vals = mats[nmodes]->values;
  ptiValue * buffer_accum;

  double inner = 0;

  double * const __restrict accum = (double *) malloc(rank*sizeof(*accum));

#ifdef HIPARTI_USE_OPENMP
  #pragma omp parallel for
#endif
  for(ptiIndex r=0; r < rank; ++r) {
    accum[r] = 0.0; 
  }

#ifdef HIPARTI_USE_OPENMP
  #pragma omp parallel
  {
    int const nthreads = omp_get_num_threads();
    #pragma omp master
    {
      buffer_accum = (ptiValue *)malloc(nthreads * rank * sizeof(ptiValue));
      for(ptiIndex j=0; j < nthreads * rank; ++j)
          buffer_accum[j] = 0.0;
    }
  }
#endif

#ifdef HIPARTI_USE_OPENMP
  #pragma omp parallel
  {
    int const tid = omp_get_thread_num();
    int const nthreads = omp_get_num_threads();
    ptiValue * loc_accum = buffer_accum + tid * rank;

    #pragma omp for
    for(ptiIndex i=0; i < I; ++i) {
      for(ptiIndex r=0; r < rank; ++r) {
        loc_accum[r] += last_vals[r+(i*stride)] * tmp_vals[r+(i*stride)];
      }
    }

    #pragma omp for
    for(ptiIndex j=0; j < rank; ++j) {
      for(int i=0; i < nthreads; ++i) {
        accum[j] += buffer_accum[i*rank + j];
      }
    }

  }

#else

  for(ptiIndex i=0; i < I; ++i) {
    for(ptiIndex r=0; r < rank; ++r) {
      accum[r] += last_vals[r+(i*stride)] * tmp_vals[r+(i*stride)];
    }
  }

#endif

#ifdef HIPARTI_USE_OPENMP
  #pragma omp parallel for reduction(+:inner)
#endif
  for(ptiIndex r=0; r < rank; ++r) {
    inner += accum[r] * lambda[r];
  }

#ifdef HIPARTI_USE_OPENMP
  free(buffer_accum);
#endif

  return inner;
}
