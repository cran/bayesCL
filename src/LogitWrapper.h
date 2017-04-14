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

#ifndef __LOGITWRAPPER__
#define __LOGITWRAPPER__

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif


#include "PolyaGamma.h"


extern "C" {
  SEXP prepareCL(SEXP pl, SEXP dev, SEXP Rpath_src, SEXP Rpath_bin, SEXP Rparameters);
  SEXP find_platforms();
  SEXP find_devices(SEXP platform_id);
  SEXP rpg_devroye(SEXP openclptr,SEXP Rbatch, SEXP Rn, SEXP Rz, SEXP Rnum, SEXP Rlocal, SEXP Rstatic_seed,SEXP Rseed );
  SEXP rpg_devroye_double(SEXP openclptr,SEXP Rbatch, SEXP Rn, SEXP Rz, SEXP Rnum, SEXP Rlocal, SEXP Rstatic_seed,SEXP Rseed );
  SEXP mult_gibbs( SEXP Rbetap,SEXP Rtyp, SEXP RtXp, SEXP Rnp,SEXP Rm0p, SEXP RP0p,SEXP RN, SEXP RP, SEXP RJ,SEXP Rsamp, SEXP Rburn, SEXP Rtiming, SEXP pointer);
  SEXP mult_gibbs_double(SEXP Rbetap, SEXP Rtyp, SEXP RtXp, SEXP Rnp, SEXP Rm0p, SEXP RP0p, SEXP RN, SEXP RP, SEXP RJ, SEXP Rsamp, SEXP Rburn, SEXP Rtiming, SEXP pointer);
  int read_file(char **output, size_t *size, const char *name);
  int write_file(const char *name, const unsigned char *content, size_t size);
  cl_int get_platform_list(cl_platform_id **platforms_out, cl_uint *num_platforms_out);
  void free_platform_list(cl_platform_id *platforms, cl_uint num_platforms);
  char* get_platform_info(cl_platform_id platform, cl_platform_info param);
  cl_int get_device_list(cl_device_id **devices_out, cl_uint *num_devices_out, cl_platform_id platform);
  void free_device_list(cl_device_id *devices, cl_uint num_devices);
  cl_int write_binaries(int dev, cl_program program, unsigned num_devices, cl_uint platform_idx,const char** path);
  cl_int compile_program(int dev, cl_uint *num_devices_out, const char *src,size_t src_size, cl_platform_id platform,cl_uint platform_idx, const char** path,char* params);
  void compile_all(int pl, int dev, const char *src, size_t src_size, const char** path, char* params);  
  
}
#endif
