// -*- mode: c++; -*-

////////////////////////////////////////////////////////////////////////////////

// Copyright 2012 Nick Polson, James Scott, and Jesse Windle.

// This file is part of BayesLogit.

// BayesLogit is free software: you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.

// BayesLogit is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

// You should have received a copy of the GNU General Public License along with
// BayesLogit.  If not, see <http://www.gnu.org/licenses/>.

////////////////////////////////////////////////////////////////////////////////

// See <http://arxiv.org/abs/1205.0310> for implementation details.

#ifndef __PolyaGammaD__
#define __PolyaGammaD__
#include "PolyaGamma.h"
#include <cmath>
#include <vector>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif
#include <time.h>

#define rng_max_val      (4294967296.0)                 // 2^32
#define MAX_SOURCE_SIZE (0x100000)
  
using std::vector;

// The numerical accuracy of __PI will affect your distribution.

class PolyaGammaD
{

  // For sum of Gammas.
  int T;
  vector<double> bvec;

 public:
  int test;
  int initialized;
  // Constructors.
  PolyaGammaD(int trunc = 200);

  unsigned int rng_rand32();
  unsigned int step_unif();
  void rng_init(int static_seed, int seed);
  double rng_unif(); 
  double rng_normal();
  double rng_exp();
  void insertEvent(cl_event* event,CLptrs *clptr);
  void copyCL(int N, int J, CLptrs *clptr);
  void step1CL(int N, int J, int j, CLptrs *clptr);
  void step2CL(int j, int N, int P, CLptrs *clptr);
  void betaCL(int P, int N, int j,int J, int m, CLptrs *clptr);
  void initData(int nr,int nc,int P, int *nn, double *tX, double *z,double *P0, double *b0, double *XB,CLptrs *clptr);
  void freeData(CLptrs *clptr);
  double calculateTime(cl_event* event1,cl_event* event2);
  void cholesky(double* b, double* R, double*d, int P);
  
  //OpenCL
  void initCL(int n, int b, int static_seed, int seed, CLptrs *clptr);
  void freeCL();
  double draw(int n, double z);
    
  void drawCL(double* x, int num, double* z, int local,CLptrs *clptr);
  void drawCL( int local ,CLptrs *clptr);
  
  void prepare(int num);
  void setseed(int i);
  
  
  clock_t start, end;
  clock_t start1, end1;
  clock_t start2, end2;
  clock_t start3, end3;
  
  cl_int status;
  cl_mem seed_table_buffer;
  cl_mem result;
  cl_mem zbuf;
  cl_mem xb_no_j_buffer;
  cl_mem xb_buffer;
  cl_mem eta_j_buffer;
  cl_mem c_j_buffer;
  cl_mem A_buffer;
  cl_mem n_buffer,n1_buffer;
  cl_mem tX_buffer;
  cl_mem tXOmega_buffer;
  cl_mem w_buffer;
  cl_mem tXOmX_buffer;
  cl_mem tXOmC_buffer;
  cl_mem Z_buffer;
  cl_mem P0_buffer;
  cl_mem b0_buffer;
  cl_mem L_buffer;
  cl_mem V_buffer;
  cl_mem mean_buffer;
  cl_mem beta_buffer;
  cl_mem tXOmX1_buffer;
  cl_mem minmax_buffer;
  cl_mem lower_buffer;
  cl_mem V1_buffer;
  
  int batch;
  int nums;
  int N_gpu,J_gpu,P_gpu,M_gpu, m_gpu, j_gpu,n_gpu;
  
  cl_uint4* seed_table;
  
  char *source_str;
  size_t source_size;
  
  double* mean;
  double* lower;
  
  typedef struct {
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int t;
  } rng_st;
  
  rng_st internal_state;
  rng_st rng_state;

};

#endif
