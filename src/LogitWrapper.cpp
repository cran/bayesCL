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
#define R_NO_REMAP 

#ifdef USE_R
#include "R.h"
#include "Rmath.h"
#include <Rinternals.h>
#include <Rdefines.h>
#endif

#include "LogitWrapper.h"
#include "MultLogit.h"
#include "MultLogitD.h"
#include "PolyaGamma.h"
#include "PolyaGammaD.h"
#include <exception>
#include <stdio.h>
#include <time.h>

static void
  _finalizer(SEXP ext)
  {
    if (NULL == R_ExternalPtrAddr(ext))
      return;
    CLptrs *ptr = (CLptrs *) R_ExternalPtrAddr(ext);
    
    free(ptr->devices);
    free(ptr->platforms);
    clReleaseKernel( ptr->kernel);
    clReleaseKernel( ptr->kernelSt);
    clReleaseKernel( ptr->kernelMinMax);
    clReleaseKernel( ptr->kernelRowSum);
    clReleaseKernel( ptr->kernelCEta);
    clReleaseKernel( ptr->kernels1);
    clReleaseKernel( ptr->kernels2);
    clReleaseKernel( ptr->kernels3);
    clReleaseKernel( ptr->kernels3a);
    clReleaseKernel( ptr->kernels4);
    clReleaseKernel( ptr->kernels4a);
    clReleaseKernel( ptr->kernelVMmul1);
    clReleaseKernel( ptr->kernelCopy);
    clReleaseKernel( ptr->dummy);
    clReleaseKernel( ptr->kernelChol);
    clReleaseKernel( ptr->kernelLeft);
    clReleaseKernel( ptr->kernelUpdate);
    clReleaseKernel( ptr->kernelInvProd);
    clReleaseKernel( ptr->kernelInvProd2);
    clReleaseKernel( ptr->kernelInv1);
    clReleaseKernel( ptr->kernelInv2);
    clReleaseKernel( ptr->kernelInv3);
    clReleaseKernel( ptr->kernelZero);
    clReleaseKernel( ptr->kernelInitChol);
    clReleaseKernel( ptr->kernelAddS2);
    
    clReleaseProgram( ptr->program);
    
    clReleaseCommandQueue(ptr->cmdQueue);
    clReleaseContext(ptr->context);
    
    Free(ptr);
    R_ClearExternalPtr(ext);
  } 

SEXP prepareCL(SEXP pl, SEXP dev, SEXP Rpath_src, SEXP Rpath_bin, SEXP Rparameters){
  FILE *fp;
  fp=NULL;
  char *source_str;
  size_t source_size;
  cl_int status;
  //prepare the OpenCL context, platform, etc objects
  CLptrs *CLpointers = Calloc(sizeof(CLptrs), CLptrs);
  int* parameters;
  parameters=INTEGER(Rparameters);
  SEXP ext;
  int error;
  
  CLpointers->PGbatch=parameters[0];
  CLpointers->PGlocal=parameters[1];
  CLpointers->MMblock=parameters[2];
  CLpointers->MMT=parameters[3];
  CLpointers->MVT=parameters[4];
  CLpointers->FWDblocks=parameters[5];
  CLpointers->Cholblock=parameters[6];
  CLpointers->MMThreads=parameters[7];
  CLpointers->WPT=parameters[8];
  
  clGetPlatformIDs(0, NULL, &(CLpointers->numPlatforms));
  
  CLpointers->platforms = (cl_platform_id*)malloc(CLpointers->numPlatforms*sizeof(cl_platform_id));
  
  clGetPlatformIDs(CLpointers->numPlatforms, CLpointers->platforms, NULL);
  
  CLpointers->numDevices = 0;
  CLpointers->devices = NULL;
  
  clGetDeviceIDs(CLpointers->platforms[INTEGER(pl)[0]], CL_DEVICE_TYPE_GPU, 0, NULL, &(CLpointers->numDevices));
  
  CLpointers->devices = (cl_device_id*)malloc(CLpointers->numDevices*sizeof(cl_device_id));
  
  clGetDeviceIDs(CLpointers->platforms[INTEGER(pl)[0]], CL_DEVICE_TYPE_GPU, CLpointers->numDevices, CLpointers->devices, NULL);
 
  CLpointers->context = NULL;
  
  CLpointers->context = clCreateContext(NULL, CLpointers->numDevices, CLpointers->devices, NULL, NULL, &status);
 
  CLpointers->cmdQueue = clCreateCommandQueue(CLpointers->context, CLpointers->devices[INTEGER(dev)[0]], CL_QUEUE_PROFILING_ENABLE, &status);
  const char *path_src,*path_bin;
  path_src=CHAR(STRING_ELT(Rpath_src,0));
  path_bin=CHAR(STRING_ELT(Rpath_bin,0));
  char params[200];
  int rts=CLpointers->MMThreads/CLpointers->WPT;
  sprintf (params, "-D PART=%d -D PARTs4=%d -D TS1=%d -D WPT1=%d -D RTS1=%d", CLpointers->MMT,CLpointers->MVT,CLpointers->MMThreads ,CLpointers->WPT, rts  );
  char *src = NULL;
  size_t src_size = 0;
  if (read_file(&src, &src_size, path_src) != 0) {
    Rprintf("Could not read OpenCL source file. Exiting...");
    error=1;
  }
  if(!error){
    ext=R_NilValue;
    UNPROTECT(1);
    return ext;
  }
  compile_all(INTEGER(pl)[0],INTEGER(dev)[0],src, src_size,&path_bin,params);
  free(src);
  fp = fopen(path_bin, "rb");
  
  source_str = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);

  
  status=0;
  cl_int clStat;
  CLpointers->program = clCreateProgramWithBinary(CLpointers->context, 1, &(CLpointers->devices[INTEGER(dev)[0]]),(const size_t *)&source_size, (const unsigned char **)&source_str, &clStat, &status);
  free(source_str);
  if(status!=0){
    Rprintf("Compile error. Exiting...");
    ext=R_NilValue;
    UNPROTECT(1);
    return ext;
  }
  
  
  status=0;
  
  status |= clBuildProgram( CLpointers->program, 1, CLpointers->devices, NULL, NULL, NULL);

  if(status!=0){
    Rprintf("Build error. Exiting...");
    ext=R_NilValue;
    UNPROTECT(1);
    return ext;
  }
  

  // create the kernel
  CLpointers->kernel = clCreateKernel(CLpointers->program, "polyagammaf", &status);
  CLpointers->kernelSt = clCreateKernel(CLpointers->program, "polyagamma_standalone", &status);
  CLpointers->kernelMinMax = clCreateKernel(CLpointers->program, "gibbs_minmax", &status);
  CLpointers->kernelRowSum = clCreateKernel(CLpointers->program, "gibbs_rowSum", &status);
  CLpointers->kernelCEta = clCreateKernel(CLpointers->program, "gibbs_c_eta", &status);
  CLpointers->kernels1 = clCreateKernel(CLpointers->program, "gibbs_step1", &status);
  CLpointers->kernels2 = clCreateKernel(CLpointers->program, "gibbs_step2", &status);
  CLpointers->kernels3 = clCreateKernel(CLpointers->program, "gibbs_step3", &status);
  CLpointers->kernels3a = clCreateKernel(CLpointers->program, "gibbs_step3a", &status);
  CLpointers->kernels4 = clCreateKernel(CLpointers->program, "gibbs_step4", &status);
  CLpointers->kernels4a = clCreateKernel(CLpointers->program, "gibbs_step4a", &status);
  CLpointers->kernelVMmul1 = clCreateKernel(CLpointers->program, "gibbs_vmmul1", &status);
  CLpointers->kernelCopy = clCreateKernel(CLpointers->program, "gibbs_copy", &status);
  CLpointers->kernelChol = clCreateKernel(CLpointers->program, "cholesky", &status);
  CLpointers->kernelLeft = clCreateKernel(CLpointers->program, "leftUpdate", &status);
  CLpointers->kernelUpdate = clCreateKernel(CLpointers->program, "midUpdate", &status);
  CLpointers->kernelInvProd = clCreateKernel(CLpointers->program, "invprod", &status);
  CLpointers->kernelInvProd2 = clCreateKernel(CLpointers->program, "invprod2", &status);
  CLpointers->kernelInv1 = clCreateKernel(CLpointers->program, "inverse1", &status);
  CLpointers->kernelInv2 = clCreateKernel(CLpointers->program, "inverse2", &status);
  CLpointers->kernelInv3 = clCreateKernel(CLpointers->program, "inverse3", &status);
  CLpointers->kernelZero = clCreateKernel(CLpointers->program, "zeroTop", &status);
  CLpointers->kernelInitChol = clCreateKernel(CLpointers->program, "initChol", &status);
  CLpointers->dummy = clCreateKernel(CLpointers->program, "dummy", &status);
  CLpointers->kernelAddS2 = clCreateKernel(CLpointers->program, "addS2", &status);
  if(status!=0){
    Rprintf("Creating kernel error. Exiting...");
    ext=R_NilValue;
    UNPROTECT(1);
    return ext;
  }

  ext=PROTECT(R_MakeExternalPtr(CLpointers, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(ext, _finalizer, TRUE);

  UNPROTECT(1);
  return ext;
}

SEXP rpg_devroye_double(SEXP openclptr,SEXP Rbatch, SEXP Rn, SEXP Rz, SEXP Rnum, SEXP Rlocal, SEXP Rstatic_seed,SEXP Rseed )
{  
  CLptrs *clptr = (CLptrs *) R_ExternalPtrAddr(openclptr);
  
  PolyaGammaD pg(1);
#ifdef USE_R
  GetRNGstate();
#endif
  
  int batch,num,local, static_seed,seed;
  int *n;
  double *z;
  n=INTEGER(Rn);
  batch=INTEGER(Rbatch)[0];
  num=INTEGER(Rnum)[0];
  local=INTEGER(Rlocal)[0];
  static_seed=INTEGER(Rstatic_seed)[0];
  seed=INTEGER(Rseed)[0];
  pg.rng_init(0,0);
  SEXP x = PROTECT(Rf_allocVector(REALSXP, num));
  z=(double*)malloc(num*sizeof(double));
  for(int jj=0;jj<num;jj++) z[jj]=REAL(Rz)[jj];
  
  //join all n-s in one table
  int nn=0;
  double *zz,*xx;
  
  for(int i=0;i<num;i++){
    nn+=n[i];
  }
  zz=(double*)malloc(nn*sizeof(double));
  xx=(double*)malloc(nn*sizeof(double));
  
  int ii=0;
  //malloc the z table
  for(int i=0;i<num;i++){
    for(int j=0;j<n[i];j++){
      zz[ii]=z[i];
      ii++;
    }
  }
  
  pg.rng_init(0,0);
  pg.initCL(nn,batch, static_seed, seed,clptr);
  pg.drawCL(xx, nn,zz,local,clptr);
  ii=0;
  for(int i=0;i<num;i++){
    REAL(x)[i]=0.0;
    for(int j=0;j<n[i];j++){
      REAL(x)[i]+=xx[ii];
      ii++;
    }
  }
  
  pg.freeCL();
  free(xx);
  free(zz);
  free(z);
#ifdef USE_R
  PutRNGstate();
#endif
  UNPROTECT(1);
  return x;
}

SEXP rpg_devroye(SEXP openclptr,SEXP Rbatch, SEXP Rn, SEXP Rz, SEXP Rnum, SEXP Rlocal, SEXP Rstatic_seed,SEXP Rseed )
{  
  CLptrs *clptr = (CLptrs *) R_ExternalPtrAddr(openclptr);

  PolyaGamma pg(1);
#ifdef USE_R
  GetRNGstate();
#endif
  
  int batch,num,local, static_seed,seed;
  int *n;
  float *z;
  n=INTEGER(Rn);
  batch=INTEGER(Rbatch)[0];
  num=INTEGER(Rnum)[0];
  local=INTEGER(Rlocal)[0];
  static_seed=INTEGER(Rstatic_seed)[0];
  seed=INTEGER(Rseed)[0];
  pg.rng_init(0,0);
  SEXP x = PROTECT(Rf_allocVector(REALSXP, num));
  z=(float*)malloc(num*sizeof(float));
  for(int jj=0;jj<num;jj++) z[jj]=REAL(Rz)[jj];
  
  //join all n-s in one table
  int nn=0;
  float *zz,*xx;
  
  for(int i=0;i<num;i++){
    nn+=n[i];
  }
  zz=(float*)malloc(nn*sizeof(float));
  xx=(float*)malloc(nn*sizeof(float));
  
  int ii=0;
  //malloc the z table
  for(int i=0;i<num;i++){
    for(int j=0;j<n[i];j++){
      zz[ii]=z[i];
      ii++;
    }
  }
  
  pg.rng_init(0,0);
  pg.initCL(nn,batch, static_seed, seed,clptr);
  pg.drawCL(xx, nn,zz,local,clptr);
  ii=0;
  for(int i=0;i<num;i++){
    REAL(x)[i]=0.0;
    for(int j=0;j<n[i];j++){
      REAL(x)[i]+=xx[ii];
      ii++;
    }
  }
  
  pg.freeCL();
  free(xx);
  free(zz);
  free(z);
#ifdef USE_R
  PutRNGstate();
#endif
  UNPROTECT(1);
  return x;
} 

SEXP find_platforms(){
  //prepare the OpenCL context, platform, etc objects
  CLptrs *CLpointers = Calloc(sizeof(CLptrs), CLptrs);
  
  clGetPlatformIDs(0, NULL, &(CLpointers->numPlatforms));
  
  // Allocate space for platforms
  CLpointers->platforms = (cl_platform_id*)malloc(CLpointers->numPlatforms*sizeof(cl_platform_id));
  
  // get platforms data
  clGetPlatformIDs(CLpointers->numPlatforms, CLpointers->platforms, NULL);
  
  SEXP result = PROTECT(Rf_allocVector(STRSXP, CLpointers->numPlatforms));
  for(unsigned int i=0;i<CLpointers->numPlatforms;i++){
    SET_STRING_ELT(result, i, Rf_mkChar(get_platform_info(CLpointers->platforms[i], CL_PLATFORM_NAME)));
  }
  UNPROTECT(1);
  return(result);
  
  
}

SEXP find_devices(SEXP platform_id){
  
  int a=INTEGER(platform_id)[0];
  cl_device_id* devices = NULL;
  cl_uint num_devices = 0;
  size_t valueSize;
  char **value;
  //prepare the OpenCL context, platform, etc objects
  CLptrs *CLpointers = Calloc(sizeof(CLptrs), CLptrs);
  
  clGetPlatformIDs(0, NULL, &(CLpointers->numPlatforms));
  
  // Allocate space for platforms
  CLpointers->platforms = (cl_platform_id*)malloc(CLpointers->numPlatforms*sizeof(cl_platform_id));
  
  // get platforms data
  clGetPlatformIDs(CLpointers->numPlatforms, CLpointers->platforms, NULL);
  
  get_device_list(&devices, &num_devices, CLpointers->platforms[a]);

  SEXP result = PROTECT(Rf_allocVector(STRSXP,num_devices));
  value = (char**) malloc(sizeof(char*)*num_devices);	
   
  for(unsigned int i=0;i<num_devices;i++){
    clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, NULL, &valueSize);
    value[i] = (char*) malloc(valueSize);
    clGetDeviceInfo(devices[i], CL_DEVICE_NAME, valueSize, value[i], NULL);
    SET_STRING_ELT(result, i, Rf_mkChar(value[i]));
  }
  UNPROTECT(1);
  return(result);
  
  
}

int read_file(char **output, size_t *size, const char *name) {
  FILE *fp = fopen(name, "rb");
  if (!fp) {
    return -1;
  }
  
  fseek(fp, 0, SEEK_END);
  *size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  
  *output = (char *)malloc(*size);
  if (!*output) {
    fclose(fp);
    return -1;
  }
  
  fread(*output, *size, 1, fp);
  fclose(fp);
  return 0;
}

int write_file(const char *name, const unsigned char *content, size_t size) {
  FILE *fp = fopen(name, "w+");
  if (!fp) {
    return -1;
  }
  fwrite(content, size, 1, fp);
  fclose(fp);
  return 0;
}

cl_int get_platform_list(cl_platform_id **platforms_out, cl_uint *num_platforms_out) {
  cl_int err;
  
  // Read the number of platforms
  cl_uint num_platforms;
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  if (err != CL_SUCCESS) {
    return err;
  }
  if (num_platforms == 0) {
    return CL_INVALID_VALUE;
  }
  
  // Allocate the array of cl_platform_id
  cl_platform_id *platforms =
    (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
  if (!platforms) {
    return CL_OUT_OF_HOST_MEMORY;
  }
  
  // Get the result
  err = clGetPlatformIDs(num_platforms, platforms, NULL);
  if (err != CL_SUCCESS) {
    free(platforms);
    return err;
  }
  
  *platforms_out = platforms;
  *num_platforms_out = num_platforms;
  return CL_SUCCESS;
}

void free_platform_list(cl_platform_id *platforms, cl_uint num_platforms) {
  free(platforms);
}

char* get_platform_info(cl_platform_id platform, cl_platform_info param) {
  cl_int err;
  
  // Read the size of the buffer for platform name
  size_t buf_size;
  err = clGetPlatformInfo(platform, param, 0, NULL, &buf_size);
  if (err != CL_SUCCESS) {
    return NULL;
  }
  if (buf_size == 0) {
    return NULL;
  }
  
  // Allocate the buffer for platform name
  char *buf = (char *)malloc(buf_size);
  if (!buf) {
    return NULL;
  }
  
  // Read the platform name
  err = clGetPlatformInfo(platform, param, buf_size, buf, NULL);
  if (err != CL_SUCCESS) {
    free(buf);
    return NULL;
  }
  
  return buf;
}

cl_int get_device_list(cl_device_id **devices_out, cl_uint *num_devices_out, cl_platform_id platform) {
  cl_int err;
  
  // Read the number of devices of the given platform
  cl_uint num_devices;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL,
                       &num_devices);
  if (err != CL_SUCCESS) {
    return err;
  }
  
  // Allocate the array of cl_device_id
  cl_device_id *devices =
    (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
  if (!devices) {
    return CL_OUT_OF_HOST_MEMORY;
  }
  
  // Read the result
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices,
                       devices, NULL);
  if (err != CL_SUCCESS) {
    free(devices);
    return err;
  }
  
  *devices_out = devices;
  *num_devices_out = num_devices;
  return CL_SUCCESS;
}

void free_device_list(cl_device_id *devices, cl_uint num_devices) {
  cl_uint i;
  for (i = 0; i < num_devices; ++i) {
    //clReleaseDevice(devices[i]);
  }
  free(devices);
}

cl_int write_binaries(int dev, cl_program program, unsigned num_devices, cl_uint platform_idx,const char** path) {
  int i;
  cl_int err = CL_SUCCESS;
  size_t *binaries_size = NULL;
  unsigned char **binaries_ptr = NULL;
  
  // Read the binaries size
  size_t binaries_size_alloc_size = sizeof(size_t) * num_devices;
  binaries_size = (size_t *)malloc(binaries_size_alloc_size);
  size_t binaries_ptr_alloc_size = sizeof(unsigned char *) * num_devices;
  if (!binaries_size) {
    err = CL_OUT_OF_HOST_MEMORY;
    goto cleanup;
  }
  
  err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                         binaries_size_alloc_size, binaries_size, NULL);
  if (err != CL_SUCCESS) {
    goto cleanup;
  }
  
  // Read the binaries
  binaries_ptr = (unsigned char **)malloc(binaries_ptr_alloc_size);
  if (!binaries_ptr) {
    err = CL_OUT_OF_HOST_MEMORY;
    goto cleanup;
  }
  memset(binaries_ptr, 0, binaries_ptr_alloc_size);
  for (i = 0; i < (int)num_devices; ++i) {
    binaries_ptr[i] = (unsigned char *)malloc(binaries_size[i]);
    if (!binaries_ptr[i]) {
      err = CL_OUT_OF_HOST_MEMORY;
      goto cleanup;
    }
  }
  
  err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, binaries_ptr_alloc_size, binaries_ptr, NULL);
  if (err != CL_SUCCESS) {
    goto cleanup;
  }
  
  // Write the binaries to file
  for (i = dev; i < dev+1; ++i) {
    // Create output file name
    char filename[128];
    
    snprintf(filename, sizeof(filename), *path, (unsigned)platform_idx, (unsigned)i);
    
    // Write the binary to the output file
    write_file(filename, binaries_ptr[i], binaries_size[i]);
  }
  
  cleanup:
    // Free the return value buffer
    if (binaries_ptr) {
      for (i = 0; i < (int)num_devices; ++i) {
        free(binaries_ptr[i]);
      }
      free(binaries_ptr);
    }
    free(binaries_size);
    
    return err;
}

cl_int compile_program(int dev, cl_uint *num_devices_out, const char *src,size_t src_size, cl_platform_id platform,cl_uint platform_idx, const char** path, char* params) {
  cl_int err = CL_SUCCESS;
  
  // Get the device list
  cl_device_id* devices = NULL;
  cl_uint num_devices = 0;
  get_device_list(&devices, &num_devices, platform);
  *num_devices_out = num_devices;
  cl_program program;
  // Create context
  cl_context_properties ctx_properties[] = {
    CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0
  };
  cl_context ctx = clCreateContext(ctx_properties, num_devices, devices, NULL,
                                   NULL, &err);
  if (err != CL_SUCCESS) {
    //goto cleanup;
  }
  
  // Create program
  program = clCreateProgramWithSource(ctx, 1, &src, &src_size, &err);
  if (err != CL_SUCCESS) {
    Rprintf("Could not compile OpenCL program. Exiting....\n");
  }
  
  // Compile program
  
  
  
  err = clBuildProgram(program, 1, devices, params, NULL, NULL);
  
  if (err != CL_SUCCESS) {
    char *programLog;
    size_t logSize;
    // check build error and build status first
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_STATUS, 
                          sizeof(cl_build_status), &err, NULL);
    
    // check build log
    clGetProgramBuildInfo(program, devices[0], 
                          CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    programLog = (char*) calloc (logSize+1, sizeof(char));
    clGetProgramBuildInfo(program, devices[0], 
                          CL_PROGRAM_BUILD_LOG, logSize+1, programLog, NULL);
    Rprintf("Build failed; error=%d, status=%d, programLog:nn%s", 
           err, err, programLog);
    free(programLog);
    goto cleanup_program;
  }
  
  // Write the binaries
  write_binaries(dev, program, num_devices, platform_idx, path);
  
  cleanup_program:
    // Free the built program
    clReleaseProgram(program);
    clReleaseContext(ctx);

  return err;
}

void compile_all(int pl, int dev, const char *src, size_t src_size, const char** path, char* params) {
  int i;
  
  // Get the platform list
  cl_platform_id *platforms = NULL;
  cl_uint num_platforms = 0;
  if (get_platform_list(&platforms, &num_platforms) != CL_SUCCESS) {
    return;
  }
  
  // For each platform compile binaries for each devices
  for (i = pl; i < (int)(pl+1); ++i) {
    // Compile for each devices
    cl_uint num_devices = 0;
    compile_program(dev, &num_devices, src, src_size, platforms[i], i, path, params);
    
    // Print the result
    char *platform_name = get_platform_info(platforms[i], CL_PLATFORM_NAME);
    free(platform_name);
  }
  
  // Free the platform list
  free_platform_list(platforms, num_platforms);
}

////////////////////////////////////////////////////////////////////////////////
// Multinomial Gibbs //
////////////////////////////////////////////////////////////////////////////////
SEXP mult_gibbs(SEXP Rbetap, SEXP Rtyp, SEXP RtXp, SEXP Rnp, SEXP Rm0p, SEXP RP0p,
                SEXP RN, SEXP RP, SEXP RJ, SEXP Rsamp, SEXP Rburn, SEXP Rtiming, SEXP pointer)
{
  double *betap, *typ, *tXp, *np, *m0p, *P0p, *timingp;
  int *N, *P, *J, *samp, *burn;
  
  betap=REAL(Rbetap); typ=REAL(Rtyp); tXp=REAL(RtXp); np=REAL(Rnp); m0p=REAL(Rm0p); P0p=REAL(RP0p);
  N=INTEGER(RN); P=INTEGER(RP); J=INTEGER(RJ); samp=INTEGER(Rsamp); burn=INTEGER(Rburn); timingp=REAL(Rtiming);
  
  
  char names[3][50] = {"N", "beta", "timing"};
  SEXP list_names, list;
  PROTECT(list_names = Rf_allocVector(STRSXP,3));    
  
  for(int i = 0; i < 3; i++)   
    SET_STRING_ELT(list_names,i,Rf_mkChar(names[i])); 
  
  int unk = *J-1; // number unknown

  //need to do because R does not support float
  float* P0=(float*)malloc((*J-1)*(*P)*(*P)*sizeof(float));
  float* m0=(float*)malloc((*J-1)*(*P)*sizeof(float));
  float* beta=(float*)malloc((*P)*unk*(*samp)*sizeof(float));
  float* ty=(float*)malloc(unk*(*N)*sizeof(float));
  float* tX=(float*)malloc((*N)*(*P)*sizeof(float));
  int* n=(int*)malloc((*N)*sizeof(int));
  float timing[6];
  
  for(int i=0;i<(*J-1)*(*P)*(*P);i++)  P0[i]=P0p[i];
  for(int i=0;i<(*J-1)*(*P);i++)  m0[i]=m0p[i];
  for(int i=0;i<(*N);i++) n[i]=np[i];
  for(int i=0;i<unk*(*N);i++) ty[i]=typ[i];
  for(int i=0;i<(*N);i++) for(int j=0;j<(*P);j++) tX[j*(*N)+i]=tXp[i*(*P)+j];
  // Multinomial Logistic Regression via Gibbs
  try{
    MultLogit logit;
    logit.CLpointer= (CLptrs *) R_ExternalPtrAddr(pointer);
    logit.gibbs(tX,ty,n,beta, m0, P0, *samp, *burn, *P,*N, *J, timing);
    
    for(int i=0;i<(*samp)*(*P)*(*J-1);i++) betap[i]=beta[i];
    for(int i=0;i<6;i++) timingp[i]=timing[i];
    
  }
  catch (std::exception& e) {
    Rprintf("Error: %s\n", e.what());
    Rprintf("Aborting Gibbs sampler.\n");
  }
  
  free(P0);free(m0);free(beta);free(tX);free(ty);free(n);

  Rprintf("Gibbs sampling finished...\n");
  PROTECT(list = Rf_allocVector(VECSXP, 3)); 
  // attaching myint vector to list:
  SET_VECTOR_ELT(list, 0, RN); 
  // attaching myfloat vector to list:
  SET_VECTOR_ELT(list, 1, Rbetap); 
  
  SET_VECTOR_ELT(list, 2, Rtiming); 
  // and attaching the vector names:
  Rf_setAttrib(list, R_NamesSymbol, list_names); 
  UNPROTECT(2);
  return(list);
  
}

SEXP mult_gibbs_double(SEXP Rbetap, SEXP Rtyp, SEXP RtXp, SEXP Rnp, SEXP Rm0p, SEXP RP0p,
                SEXP RN, SEXP RP, SEXP RJ, SEXP Rsamp, SEXP Rburn, SEXP Rtiming, SEXP pointer)
{
  double *betap, *typ, *tXp, *np, *m0p, *P0p, *timingp;
  int *N, *P, *J, *samp, *burn;
  
  betap=REAL(Rbetap); typ=REAL(Rtyp); tXp=REAL(RtXp); np=REAL(Rnp); m0p=REAL(Rm0p); P0p=REAL(RP0p);
  N=INTEGER(RN); P=INTEGER(RP); J=INTEGER(RJ); samp=INTEGER(Rsamp); burn=INTEGER(Rburn); timingp=REAL(Rtiming);
  
  
  char names[3][50] = {"N", "beta", "timing"};
  SEXP list_names, list;
  PROTECT(list_names = Rf_allocVector(STRSXP,3));    
  
  for(int i = 0; i < 3; i++)   
    SET_STRING_ELT(list_names,i,Rf_mkChar(names[i])); 
  
  int unk = *J-1; // number unknown
  
  //need to do because R does not support float
  double* P0=(double*)malloc((*J-1)*(*P)*(*P)*sizeof(double));
  double* m0=(double*)malloc((*J-1)*(*P)*sizeof(double));
  double* beta=(double*)malloc((*P)*unk*(*samp)*sizeof(double));
  double* ty=(double*)malloc(unk*(*N)*sizeof(double));
  double* tX=(double*)malloc((*N)*(*P)*sizeof(double));
  int* n=(int*)malloc((*N)*sizeof(int));
  double timing[6];
  
  for(int i=0;i<(*J-1)*(*P)*(*P);i++)  P0[i]=P0p[i];
  for(int i=0;i<(*J-1)*(*P);i++)  m0[i]=m0p[i];
  for(int i=0;i<(*N);i++) n[i]=np[i];
  for(int i=0;i<unk*(*N);i++) ty[i]=typ[i];
  for(int i=0;i<(*N);i++) for(int j=0;j<(*P);j++) tX[j*(*N)+i]=tXp[i*(*P)+j];
  
  // Multinomial Logistic Regression via Gibbs
  try{
    MultLogitD logit;
    
    logit.CLpointer= (CLptrs *) R_ExternalPtrAddr(pointer);
    //set the optimization parameters
    
    logit.gibbs(tX,ty,n,beta, m0, P0, *samp, *burn, *P,*N, *J, timing);
    
    for(int i=0;i<(*samp)*(*P)*(*J-1);i++) betap[i]=beta[i];
    for(int i=0;i<6;i++) timingp[i]=timing[i];
    
  }
  catch (std::exception& e) {
    Rprintf("Error: %s\n", e.what());
    Rprintf("Aborting Gibbs sampler.\n");
  }
  
  free(P0);free(m0);free(beta);free(tX);free(ty);free(n);
  
  Rprintf("Gibbs sampling finished...\n");
  PROTECT(list = Rf_allocVector(VECSXP, 3)); 
  // attaching myint vector to list:
  SET_VECTOR_ELT(list, 0, RN); 
  // attaching mydouble vector to list:
  SET_VECTOR_ELT(list, 1, Rbetap); 
  
  SET_VECTOR_ELT(list, 2, Rtiming); 
  // and attaching the vector names:
  Rf_setAttrib(list, R_NamesSymbol, list_names); 
  UNPROTECT(2);
  return(list);
  
}


