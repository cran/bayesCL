#include "PolyaGamma.h"
#include <stdexcept>
#include <time.h>

#ifdef USE_R
#include "R.h"
#include "Rmath.h"
#include <Rinternals.h>
#include <Rdefines.h>
#endif

using std::pow;


void PolyaGamma::rng_init(int static_seed, int seed){

  
  if(static_seed==0){
    internal_state.x = time(NULL);
  }
  else{
    internal_state.x = seed;
  }
  
  internal_state.y = 12345;
  internal_state.z = 31415926;
  internal_state.t =3 ;//3
  //srand(internal_state.x);

  rng_state.x=(unsigned int)(unif_rand()*4294967295 );
  rng_state.y=(unsigned int)(unif_rand()*4294967295 );
  rng_state.z=(unsigned int)(unif_rand()*4294967295 );
  rng_state.t=(unsigned int)(unif_rand()*4294967295 );

}

PolyaGamma::PolyaGamma(int trunc) : T(trunc), bvec(T)
{
} // PolyaGamma

void PolyaGamma::insertEvent(cl_event* event,CLptrs *clptr){
  size_t globalWorkSize[1];
  globalWorkSize[0]=1;
  status=clEnqueueNDRangeKernel( clptr->cmdQueue, clptr->dummy, 1, NULL, globalWorkSize, NULL, 0, NULL, event);
  if(status!=0)
    Rprintf("event: %d\n",status);
}
float PolyaGamma::calculateTime(cl_event* event1,cl_event* event2){
  cl_ulong time_start, time_end;
  float total_time;
  clWaitForEvents(1 , event1);
  clWaitForEvents(1 , event2);
  
  clGetEventProfilingInfo(*event1, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
  clGetEventProfilingInfo(*event2, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
  total_time = time_end - time_start;
  return total_time / 1000000000.0;
}

void PolyaGamma::initCL(int n, int b, int static_seed, int seed,CLptrs *clptr){

  nums=n;
  batch=b;
  n=(int)(ceil(n/((float)b)));
  
  status=0;
  end=0;
  end1=0;
  end2=0;
  end3=0;
  
  
  seed_table= (cl_uint4*) calloc(n,sizeof(cl_uint4));
  
  rng_init(static_seed,seed);
  
  for(int i=0;i<n;i++){
    
    if(static_seed==1){
      //TODO
    }else{
      seed_table[i].s[0]=(unsigned int)(unif_rand()*4294967295 );
      seed_table[i].s[1]=(unsigned int)(unif_rand()*4294967295 );
      seed_table[i].s[2]=(unsigned int)(unif_rand()*4294967295 );
      seed_table[i].s[3]=(unsigned int)(unif_rand()*4294967295 ); 
    }
    
  }
  
  seed_table_buffer = clCreateBuffer( clptr->context, CL_MEM_READ_ONLY, sizeof(cl_uint4)*n, NULL, &status);
  
  zbuf = clCreateBuffer( clptr->context, CL_MEM_READ_ONLY, nums*sizeof(float), NULL, &status);
  result = clCreateBuffer( clptr->context, CL_MEM_WRITE_ONLY, nums*sizeof(float), NULL, &status);
  
  status = clEnqueueWriteBuffer( clptr->cmdQueue, seed_table_buffer, CL_TRUE, 0, n*sizeof(cl_uint4), seed_table, 0, NULL, NULL);
  if(status!=0)
    Rprintf("seed: %d\n",status);
  status |= clSetKernelArg( clptr->kernelSt, 0, sizeof(cl_mem), &seed_table_buffer);
  status |= clSetKernelArg( clptr->kernelSt, 1, sizeof(cl_mem), &result);
  status |= clSetKernelArg( clptr->kernelSt, 2, sizeof(cl_mem), &zbuf);
  status |= clSetKernelArg( clptr->kernelSt, 3, sizeof(int), &batch);
  status |= clSetKernelArg( clptr->kernelSt, 4, sizeof(int), &nums);
  
  if(status!=0)
    Rprintf("init: %d\n",status);
}


//TODO: make separate function if PG draw or mlogit
void PolyaGamma::freeCL(){
  
  clReleaseMemObject(seed_table_buffer);
  clReleaseMemObject(result);
  clReleaseMemObject(zbuf);
  //clReleaseCommandQueue(clptr->cmdQueue);
  //clReleaseContext(clptr->context);
  
}

void PolyaGamma::drawCL( float* x, int num, float* z, int local ,CLptrs *clptr){



  int n;
  n=(int)(ceil(nums/((float)batch)));
  for(int i=0;i<n;i++){
      seed_table[i].s[0]=(unsigned int)(unif_rand()*4294967295 );
      seed_table[i].s[1]=(unsigned int)(unif_rand()*4294967295 );
      seed_table[i].s[2]=(unsigned int)(unif_rand()*4294967295 );
      seed_table[i].s[3]=(unsigned int)(unif_rand()*4294967295 );
      
  }
  status |= clEnqueueWriteBuffer( clptr->cmdQueue, seed_table_buffer, CL_TRUE, 0, n*sizeof(cl_uint4), seed_table, 0, NULL, NULL);
  if(status!=0)
    Rprintf("seed: %d\n",status);
  size_t globalWorkSize[1];
  size_t localWorkSize[1];    
  localWorkSize[0] = local;
  globalWorkSize[0] = (int)ceil(num/((float)batch));
  globalWorkSize[0] += (local-globalWorkSize[0]%local);
  status |= clEnqueueWriteBuffer( clptr->cmdQueue, zbuf, CL_TRUE, 0, num*sizeof(float), z, 0, NULL, NULL);
  //execute the kernel
  status=0;
  
  status |= clEnqueueNDRangeKernel( clptr->cmdQueue, clptr->kernelSt, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
  if(status!=0)
    Rprintf("draw: %d\n",status);
  status |= clEnqueueReadBuffer( clptr->cmdQueue, result, CL_TRUE, 0, nums*sizeof(float), x, 0, NULL, NULL);

}

void PolyaGamma::drawCL( int local ,CLptrs *clptr){

  int n;
  if(test==0){
    n=(int)(ceil(nums/((float)batch)));
    for(int i=0;i<n;i++){
      seed_table[i].s[0]=(unsigned int)(unif_rand()*4294967295 );
      seed_table[i].s[1]=(unsigned int)(unif_rand()*4294967295 );
      seed_table[i].s[2]=(unsigned int)(unif_rand()*4294967295 );
      seed_table[i].s[3]=(unsigned int)(unif_rand()*4294967295 ); 
    }
    
    status |= clEnqueueWriteBuffer( clptr->cmdQueue, seed_table_buffer, CL_TRUE, 0, n*sizeof(cl_uint4), seed_table, 0, NULL, NULL);
    if(status!=0)
      Rprintf("seedw: %d\n",status);
//    test++;
  }
  size_t globalWorkSize[1];
  size_t localWorkSize[1];    
  status=0;

  localWorkSize[0] = local;
  globalWorkSize[0] = (int)ceil(nums/((float)batch));
  globalWorkSize[0] += (local-globalWorkSize[0]%local);
  status = clEnqueueNDRangeKernel( clptr->cmdQueue, clptr->kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &(clptr->ev4));
  if(status!=0)
    Rprintf("draw: %d\n",status);
}


void PolyaGamma::step2CL(int j, int N, int P, CLptrs *clptr){

  j_gpu=j;
  status=0;
  status|=clSetKernelArg( clptr->kernels1, 0, sizeof(cl_mem), &result);
  status|=clSetKernelArg( clptr->kernels3, 6, sizeof(int), &j_gpu);
  status|=clSetKernelArg( clptr->kernels4, 5, sizeof(int), &j_gpu);

  size_t globalWorkSize[1];
  globalWorkSize[0] = N;
  status = clEnqueueNDRangeKernel( clptr->cmdQueue, clptr->kernels1, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
  if(status!=0)
    Rprintf("s1 %d\n",status);
  globalWorkSize[0] = N*P;
  status|=clSetKernelArg( clptr->kernels2, 0, sizeof(cl_mem), &tX_buffer);
  status|=clSetKernelArg( clptr->kernels2, 1, sizeof(cl_mem), &tXOmega_buffer);
  status|=clSetKernelArg( clptr->kernels2, 2, sizeof(cl_mem), &w_buffer);
  status|=clSetKernelArg( clptr->kernels2, 3, sizeof(int), &N_gpu);
  status = clEnqueueNDRangeKernel( clptr->cmdQueue, clptr->kernels2, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
  if(status!=0)
    Rprintf("s2 %d\n",status);
  globalWorkSize[0] = P*P;
  unsigned int block_size=16;
  unsigned int Ppadded=((P+block_size-1)/block_size)*block_size;
  size_t global[2]={Ppadded,Ppadded*clptr->MMT};
  size_t local[2]={block_size,block_size};

  status = clEnqueueNDRangeKernel( clptr->cmdQueue, clptr->kernels3, 2, NULL, global,local, 0, NULL, NULL);
  if(status!=0)
      Rprintf("s3 %d\n",status);
  clSetKernelArg( clptr->kernels3a, 0, sizeof(cl_mem), &tXOmX_buffer);
  clSetKernelArg( clptr->kernels3a, 1, sizeof(int), &P_gpu);
  size_t global1[2]={(unsigned)P,(unsigned)P};
  status = clEnqueueNDRangeKernel( clptr->cmdQueue, clptr->kernels3a, 2, NULL, global1,NULL, 0, NULL,NULL);  
  if(status!=0)
    Rprintf("s3a %d\n",status);
  size_t global2[2]={(unsigned)P,(unsigned)clptr->MVT};
  status = clEnqueueNDRangeKernel( clptr->cmdQueue, clptr->kernels4, 2, NULL, global2, NULL, 0, NULL, NULL);
  if(status!=0)
    Rprintf("s4 %d\n",status);
  globalWorkSize[0] = (unsigned)P;
  clSetKernelArg( clptr->kernels4a, 0, sizeof(cl_mem), &tXOmC_buffer);
  clSetKernelArg( clptr->kernels4a, 1, sizeof(int), &P_gpu);
  status = clEnqueueNDRangeKernel( clptr->cmdQueue, clptr->kernels4a, 1, NULL, globalWorkSize, NULL, 0, NULL, &(clptr->ev6));
  if(status!=0)
    Rprintf("s4a %d\n",status);
}


void PolyaGamma::betaCL(int P, int N, int j, int J, int m, CLptrs *clptr){
  size_t globalWorkSize[1];
  
  /*
  globalWorkSize[0] = 1;
  clSetKernelArg( clptr->kernelChol1, 0, sizeof(cl_mem), &L_buffer);
  clSetKernelArg( clptr->kernelChol1, 1, sizeof(cl_mem), &V_buffer);
  clSetKernelArg( clptr->kernelChol1, 2, sizeof(cl_mem), &tXOmX_buffer);
  clSetKernelArg( clptr->kernelChol1, 3, sizeof(int), &P_gpu);
  status=clEnqueueNDRangeKernel( clptr->cmdQueue, clptr->kernelChol1, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
  */
 
  j_gpu=j;
  N_gpu=N;
  J_gpu=J;
  globalWorkSize[0] = N;
  clSetKernelArg( clptr->kernelVMmul1, 0, sizeof(cl_mem), &tX_buffer);
  clSetKernelArg( clptr->kernelVMmul1, 1, sizeof(cl_mem), &beta_buffer);
  clSetKernelArg( clptr->kernelVMmul1, 2, sizeof(cl_mem), &xb_buffer);
  clSetKernelArg( clptr->kernelVMmul1, 3, sizeof(cl_mem), &xb_no_j_buffer);
  clSetKernelArg( clptr->kernelVMmul1, 4, sizeof(int), &j_gpu);
  clSetKernelArg( clptr->kernelVMmul1, 5, sizeof(int), &P_gpu);
  clSetKernelArg( clptr->kernelVMmul1, 6, sizeof(int), &N_gpu);
  clSetKernelArg( clptr->kernelVMmul1, 7, sizeof(int), &J_gpu);
  status=clEnqueueNDRangeKernel( clptr->cmdQueue, clptr->kernelVMmul1, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
  if(status!=0)
    Rprintf("vmul: %d\n",status);
    
  
}
void PolyaGamma::cholesky(float* b, float* R, float*d, int P){
  float *L;
  float *V;
  float *V1;
  L=(float*)malloc(P*P*sizeof(float));
  V=(float*)malloc(P*P*sizeof(float));
  V1=(float*)malloc(P*P*sizeof(float));
  mean=(float*)malloc(P*sizeof(float));
  lower=(float*)malloc(P*P*sizeof(float));
  
  for(int i=0;i<P;i++){
    for(int j=0;j<P;j++){
      L[i*P+j]=0.0;
      lower[i*P+j]=0.0;
      if(i==j){
        V[i*P+j]=1.0;
        V1[i*P+j]=1.0;
      }else{
        V[i*P+j]=0.0;
        V1[i*P+j]=0.0;
      }
      
    }
  }
  
  int n=P;
  
  float faktor;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < (i+1); j++) {
      float s = 0;
      for (int k = 0; k < j; k++)
        s += L[i*P+k] * L[j*P+k];
      
      L[i*P+j] = (i == j) ?  sqrt(R[i*P+i] - s) :  (1.0 / L[j*P+j] * (R[i*P+j] - s));
    }
    
  for (int i = 0; i < n; i++){
    if(i>0){
      for (int j = i; j < n; j++) {
        faktor=L[j*P+i-1];
        for (int k = 0; k < n; k++) {
          L[j*P+k]=L[j*P+k]-faktor*L[(i-1)*P+k];
          V1[j*P+k]=V1[j*P+k]-faktor*V1[(i-1)*P+k];
        }
      }
    }
    faktor=L[i*P+i];
    for (int k = 0; k < n; k++) {
      L[i*P+k]=L[i*P+k]/faktor;
      V1[(i*P+k)]=V1[(i*P+k)]/faktor;  
    }
  }
  for (int i = 0; i < n; i++){
    for (int j = 0; j < n; j++) {
      V[(i*P+j)]=0;
      for (int k = 0; k < n; k++) {
        V[(i*P+j)]+=V1[(k*P+i)]*V1[(k*P+j)];
      }
    }
  }
    
    
  for (int j = 0; j < n; j++) {
    mean[j]=0;
    for (int k = 0; k < n; k++) {
      mean[j]+=V[k*P+j]*b[k];
    }
  }
    
  // Cholesky
  for (int i = 0; i < n; i++)
    for (int j = 0; j < (i+1); j++) {
      float s = 0;
      for (int k = 0; k < j; k++)
        s += lower[i*P+k] * lower[j*P+k];
      
      lower[i*P+j] = (i == j) ?  sqrt(V[i*P+i] - s) :  (1.0 / lower[j*P+j] * (V[i*P+j] - s));
    }
  
  float acc=0;
  for (int i=0; i < P; i++){
    acc=0.0;
    for (int k=0; k < i; k++){
      acc+=lower[i*P+k]*d[i];
    }
    d[i]=acc+mean[i];
  }
  
  free(mean);free(lower);free(L);free(V);free(V1);
}
void PolyaGamma::step1CL(int N, int J, int j,CLptrs *clptr){
  
  size_t globalWorkSize[1];
  globalWorkSize[0] = N;
  clSetKernelArg( clptr->kernelCEta, 7, sizeof(int), &j);
  status=clEnqueueNDRangeKernel( clptr->cmdQueue, clptr->kernelMinMax, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
  if(status!=0)
    Rprintf("minMax: %d\n",status);
  status=clEnqueueNDRangeKernel( clptr->cmdQueue, clptr->kernelRowSum, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
  if(status!=0)
    Rprintf("rowSum: %d\n",status);
  status=clEnqueueNDRangeKernel( clptr->cmdQueue, clptr->kernelCEta, 1, NULL, globalWorkSize, NULL, 0, NULL, &(clptr->ev2));
  if(status!=0)
    Rprintf("cEta: %d\n",status);

}

void PolyaGamma::initData(int nr,int nc,int P, int *nn, float *tX, float *z,float *P0, float *b0, float *XB,CLptrs *clptr){
  
  xb_no_j_buffer = clCreateBuffer( clptr->context, CL_MEM_READ_WRITE, sizeof(float)*(nc-1)*nr, NULL, &status);
  if(status!=0)
    Rprintf("create1: %d\n",status);
  minmax_buffer = clCreateBuffer( clptr->context, CL_MEM_READ_WRITE, sizeof(float)*nr, NULL, &status);
  if(status!=0)
    Rprintf("create2: %d\n",status);
  xb_buffer = clCreateBuffer( clptr->context, CL_MEM_READ_WRITE, sizeof(float)*nc*nr, NULL, &status);
  if(status!=0)
    Rprintf("create3: %d\n",status);
  c_j_buffer = clCreateBuffer( clptr->context, CL_MEM_READ_WRITE, sizeof(float)*nr, NULL, &status);
  if(status!=0)
    Rprintf("create4: %d\n",status);
  A_buffer = clCreateBuffer( clptr->context, CL_MEM_READ_WRITE, sizeof(float)*nr, NULL, &status);
  if(status!=0)
    Rprintf("create5: %d\n",status);
  eta_j_buffer = clCreateBuffer( clptr->context, CL_MEM_READ_WRITE, sizeof(float)*nr, NULL, &status);
  if(status!=0)
    Rprintf("create6: %d\n",status);
  tX_buffer = clCreateBuffer( clptr->context, CL_MEM_READ_WRITE, sizeof(float)*P*nr, NULL, &status);
  if(status!=0)
    Rprintf("create7: %d\n",status);
  tXOmega_buffer = clCreateBuffer( clptr->context, CL_MEM_READ_WRITE, sizeof(float)*P*nr, NULL, &status);
  if(status!=0)
    Rprintf("create8: %d\n",status);
  w_buffer=clCreateBuffer( clptr->context, CL_MEM_READ_WRITE, sizeof(float)*nr*(nc-1), NULL, &status);
  if(status!=0)
    Rprintf("create9: %d\n",status);
  tXOmX_buffer=clCreateBuffer( clptr->context, CL_MEM_READ_WRITE, sizeof(float)*P*P*clptr->MMT, NULL, &status);
  if(status!=0)
    Rprintf("create10: %d\n",status);
  tXOmC_buffer=clCreateBuffer( clptr->context, CL_MEM_READ_WRITE, sizeof(float)*P*clptr->MVT, NULL, &status);
  if(status!=0)
    Rprintf("create11: %d\n",status);
  mean_buffer=clCreateBuffer( clptr->context, CL_MEM_READ_WRITE, sizeof(float)*P, NULL, &status);
  if(status!=0)
    Rprintf("create12: %d\n",status);
  beta_buffer=clCreateBuffer( clptr->context, CL_MEM_READ_WRITE, sizeof(float)*P, NULL, &status);
  if(status!=0)
    Rprintf("create13: %d\n",status);
  Z_buffer=clCreateBuffer( clptr->context, CL_MEM_READ_WRITE, sizeof(float)*P*(nc-1), NULL, &status);
  if(status!=0)
    Rprintf("create14: %d\n",status);
  P0_buffer=clCreateBuffer( clptr->context, CL_MEM_READ_WRITE, sizeof(float)*P*P*(nc-1), NULL, &status);
  if(status!=0)
    Rprintf("create15: %d\n",status);
  b0_buffer=clCreateBuffer( clptr->context, CL_MEM_READ_WRITE, sizeof(float)*P*(nc-1), NULL, &status);
  if(status!=0)
    Rprintf("create16: %d\n",status);
  L_buffer=clCreateBuffer( clptr->context, CL_MEM_READ_WRITE, sizeof(float)*P*P, NULL, &status);
  if(status!=0)
    Rprintf("create17: %d\n",status);
  V_buffer=clCreateBuffer( clptr->context, CL_MEM_READ_WRITE, sizeof(float)*P*P, NULL, &status);
  if(status!=0)
    Rprintf("create18: %d\n",status);
  V1_buffer=clCreateBuffer( clptr->context, CL_MEM_READ_WRITE, sizeof(float)*P*P, NULL, &status);
  if(status!=0)
    Rprintf("create19: %d\n",status);
  lower_buffer=clCreateBuffer( clptr->context, CL_MEM_READ_WRITE, sizeof(float)*P*P, NULL, &status);
  if(status!=0)
    Rprintf("create20: %d\n",status);
  int *n_buf,*n1_buf;
  
  int sum_n=0;
  for(int jj=0;jj<nr;jj++){
    sum_n+=nn[jj];
  }
  n_buf=(int*)malloc(sizeof(int)*sum_n);
  n1_buf=(int*)malloc(sizeof(int)*nr);
  int k=0;
  for(int jj=0;jj<nr;jj++){
    n1_buf[jj]=k;
    for(int ii=0;ii<nn[jj];ii++){
      n_buf[k]=jj;
      k++;
    }
  }
  n_buffer = clCreateBuffer( clptr->context, CL_MEM_READ_ONLY, sizeof(int)*sum_n, NULL, &status);
  if(status!=0)
    Rprintf("create20: %d\n",status);
  n1_buffer = clCreateBuffer( clptr->context, CL_MEM_READ_ONLY, sizeof(int)*nr, NULL, &status);
  if(status!=0)
    Rprintf("create21: %d\n",status);
  status=clEnqueueWriteBuffer( clptr->cmdQueue, n_buffer, CL_TRUE, 0, sum_n*sizeof(int), n_buf, 0, NULL, NULL);
  if(status!=0)
    Rprintf("write n: %d\n",status);
  status=clEnqueueWriteBuffer( clptr->cmdQueue, n1_buffer, CL_TRUE, 0, sum_n*sizeof(int), n1_buf, 0, NULL, NULL);
  if(status!=0)
    Rprintf("write n1: %d\n",status);
  status=clEnqueueWriteBuffer( clptr->cmdQueue, tX_buffer, CL_TRUE, 0, sizeof(float)*P*nr, tX, 0, NULL, NULL);
  if(status!=0)
    Rprintf("write tX: %d\n",status);
  status=clEnqueueWriteBuffer( clptr->cmdQueue, xb_buffer, CL_TRUE, 0, nc*nr*sizeof(float), XB, 0, NULL, NULL);
  if(status!=0)
    Rprintf("write xb: %d\n",status);
  status=clEnqueueWriteBuffer( clptr->cmdQueue, P0_buffer, CL_TRUE, 0, sizeof(float)*P*P*(nc-1), P0, 0, NULL, NULL);
  if(status!=0)
    Rprintf("write p0: %d\n",status);
  
  status=clEnqueueWriteBuffer( clptr->cmdQueue, b0_buffer, CL_TRUE, 0, sizeof(float)*P*(nc-1), b0, 0, NULL, NULL);
  if(status!=0)
    Rprintf("write b0: %d\n",status);
  status=clEnqueueWriteBuffer( clptr->cmdQueue, Z_buffer, CL_TRUE, 0, sizeof(float)*P*(nc-1), z, 0, NULL, NULL);
  if(status!=0)
    Rprintf("write Z: %d\n",status);
  
  N_gpu=nr;
  J_gpu=nc;
  P_gpu=P;
  n_gpu=sum_n;
  status=0;
  
  
  status|=clSetKernelArg( clptr->kernelMinMax, 0, sizeof(cl_mem), &xb_no_j_buffer);
  status|=clSetKernelArg( clptr->kernelMinMax, 1, sizeof(cl_mem), &minmax_buffer);
  status|=clSetKernelArg( clptr->kernelMinMax, 2, sizeof(int), &N_gpu);
  status|=clSetKernelArg( clptr->kernelMinMax, 3, sizeof(int), &J_gpu);
  
  status|=clSetKernelArg( clptr->kernelRowSum, 0, sizeof(cl_mem), &xb_no_j_buffer);
  status|=clSetKernelArg( clptr->kernelRowSum, 1, sizeof(cl_mem), &minmax_buffer);
  status|=clSetKernelArg( clptr->kernelRowSum, 2, sizeof(cl_mem), &A_buffer);
  status|=clSetKernelArg( clptr->kernelRowSum, 3, sizeof(int), &N_gpu);
  status|=clSetKernelArg( clptr->kernelRowSum, 4, sizeof(int), &J_gpu);
  
  status|=clSetKernelArg( clptr->kernelCEta, 0, sizeof(cl_mem), &xb_buffer);
  status|=clSetKernelArg( clptr->kernelCEta, 1, sizeof(cl_mem), &A_buffer);
  status|=clSetKernelArg( clptr->kernelCEta, 2, sizeof(cl_mem), &eta_j_buffer);
  status|=clSetKernelArg( clptr->kernelCEta, 3, sizeof(cl_mem), &c_j_buffer);
  status|=clSetKernelArg( clptr->kernelCEta, 4, sizeof(cl_mem), &minmax_buffer);
  status|=clSetKernelArg( clptr->kernelCEta, 5, sizeof(int), &N_gpu);
  status|=clSetKernelArg( clptr->kernelCEta, 6, sizeof(int), &J_gpu);
  
  status|=clSetKernelArg( clptr->kernels1, 1, sizeof(cl_mem), &w_buffer);
  status|=clSetKernelArg( clptr->kernels1, 2, sizeof(cl_mem), &n1_buffer);
  status|=clSetKernelArg( clptr->kernels1, 3, sizeof(int), &N_gpu);
  status|=clSetKernelArg( clptr->kernels1, 4, sizeof(int), &n_gpu);
  

  
  status|=clSetKernelArg( clptr->kernels3, 0, sizeof(cl_mem), &tX_buffer);
  status|=clSetKernelArg( clptr->kernels3, 1, sizeof(cl_mem), &tXOmega_buffer);
  status|=clSetKernelArg( clptr->kernels3, 2, sizeof(cl_mem), &tXOmX_buffer);
  status|=clSetKernelArg( clptr->kernels3, 3, sizeof(cl_mem), &P0_buffer);
  status|=clSetKernelArg( clptr->kernels3, 4, sizeof(cl_mem), &L_buffer);
  status|=clSetKernelArg( clptr->kernels3, 5, sizeof(cl_mem), &V_buffer);
  status|=clSetKernelArg( clptr->kernels3, 7, sizeof(int), &N_gpu);
  status|=clSetKernelArg( clptr->kernels3, 8, sizeof(int), &P_gpu);
  
  status|=clSetKernelArg( clptr->kernels4, 0, sizeof(cl_mem), &tXOmC_buffer);
  status|=clSetKernelArg( clptr->kernels4, 1, sizeof(cl_mem), &tXOmega_buffer);
  status|=clSetKernelArg( clptr->kernels4, 2, sizeof(cl_mem), &c_j_buffer);
  status|=clSetKernelArg( clptr->kernels4, 3, sizeof(cl_mem), &Z_buffer);
  status|=clSetKernelArg( clptr->kernels4, 4, sizeof(cl_mem), &b0_buffer);
  
  status|=clSetKernelArg( clptr->kernels4, 6, sizeof(int), &J_gpu);
  status|=clSetKernelArg( clptr->kernels4, 7, sizeof(int), &N_gpu);
  status|=clSetKernelArg( clptr->kernels4, 8, sizeof(int), &P_gpu);
  if(status!=0)
    Rprintf("arg: %d\n",status);
  free(n_buf);
  free(n1_buf);
  
  nums=sum_n;
  batch=clptr->PGbatch;
  
  int nnn=(int)(ceil(nums/((float)batch)));
  
  seed_table= (cl_uint4*) calloc(nnn,sizeof(cl_uint4));
  
  seed_table_buffer = clCreateBuffer( clptr->context, CL_MEM_READ_ONLY, sizeof(cl_uint4)*nnn, NULL, &status);
  if(status!=0)
    Rprintf("create22: %d\n",status);
  result = clCreateBuffer( clptr->context, CL_MEM_READ_WRITE, nums*sizeof(float), NULL, &status);
  if(status!=0)
    Rprintf("create23: %d\n",status);
  status |= clSetKernelArg( clptr->kernel, 0, sizeof(cl_mem), &seed_table_buffer);
  status |= clSetKernelArg( clptr->kernel, 1, sizeof(cl_mem), &result);
  status |= clSetKernelArg( clptr->kernel, 2, sizeof(cl_mem), &eta_j_buffer);
  status |= clSetKernelArg( clptr->kernel, 3, sizeof(cl_mem), &n_buffer);
  status |= clSetKernelArg( clptr->kernel, 4, sizeof(int), &batch);
  status |= clSetKernelArg( clptr->kernel, 5, sizeof(int), &nums);
  if(status!=0)
    Rprintf("arg1: %d\n",status);
}

void PolyaGamma::copyCL(int N, int J, CLptrs *clptr){
  
  size_t globalWorkSize[1];
  globalWorkSize[0] = N*(J-1);
  clSetKernelArg( clptr->kernelCopy, 0, sizeof(cl_mem), &xb_buffer);
  clSetKernelArg( clptr->kernelCopy, 1, sizeof(cl_mem), &xb_no_j_buffer);
  N_gpu=N;
  J_gpu=J;
  
  clSetKernelArg( clptr->kernelCopy, 2, sizeof(int), &N_gpu);
  clSetKernelArg( clptr->kernelCopy, 3, sizeof(int), &J_gpu);
  status=clEnqueueNDRangeKernel( clptr->cmdQueue, clptr->kernelCopy, 1, NULL, globalWorkSize, NULL, 0, NULL,  &(clptr->ev9));
  if(status!=0)
    Rprintf("copy: %d\n",status);
}

void PolyaGamma::freeData(CLptrs *clptr){
  
  status=clReleaseMemObject(xb_no_j_buffer);
  if(status!=0)
    Rprintf("free 1: %d\n",status);
  clReleaseMemObject(minmax_buffer);
  if(status!=0)
    Rprintf("free 2: %d\n",status);
  clReleaseMemObject(xb_buffer);
  if(status!=0)
    Rprintf("free 3: %d\n",status);
  clReleaseMemObject(eta_j_buffer);
  if(status!=0)
    Rprintf("free 4: %d\n",status);
  clReleaseMemObject(c_j_buffer);
  if(status!=0)
    Rprintf("free 5: %d\n",status);
  clReleaseMemObject(A_buffer);
  if(status!=0)
    Rprintf("free 6: %d\n",status);
  clReleaseMemObject(tX_buffer);
  if(status!=0)
    Rprintf("free 7: %d\n",status);
  clReleaseMemObject(tXOmega_buffer);
  if(status!=0)
    Rprintf("free 8: %d\n",status);
  clReleaseMemObject(w_buffer);
  if(status!=0)
    Rprintf("free 9: %d\n",status);
  clReleaseMemObject(tXOmX_buffer);
  if(status!=0)
    Rprintf("free 10: %d\n",status);
  clReleaseMemObject(tXOmC_buffer);
  if(status!=0)
    Rprintf("free 11: %d\n",status);
  clReleaseMemObject(mean_buffer);
  if(status!=0)
    Rprintf("free 12: %d\n",status);
  clReleaseMemObject(beta_buffer);
  if(status!=0)
    Rprintf("free 13: %d\n",status);
  clReleaseMemObject(Z_buffer);
  if(status!=0)
    Rprintf("free 14: %d\n",status);
  clReleaseMemObject(P0_buffer);
  if(status!=0)
    Rprintf("free 15: %d\n",status);
  clReleaseMemObject(b0_buffer);
  if(status!=0)
    Rprintf("free 16: %d\n",status);
  clReleaseMemObject(L_buffer);
  if(status!=0)
    Rprintf("free 17: %d\n",status);
  clReleaseMemObject(V_buffer);
  if(status!=0)
    Rprintf("free 18: %d\n",status);
  clReleaseMemObject(V1_buffer);
  if(status!=0)
    Rprintf("free 19: %d\n",status);
  clReleaseMemObject(n_buffer);
  if(status!=0)
    Rprintf("free 20: %d\n",status);
  clReleaseMemObject(n1_buffer);
  if(status!=0)
    Rprintf("free 21: %d\n",status);
  clReleaseMemObject(lower_buffer);  
  if(status!=0)
    Rprintf("free 22: %d\n",status);
  clReleaseMemObject(seed_table_buffer);
  if(status!=0)
    Rprintf("free 23: %d\n",status);
  clReleaseMemObject(result);
  if(status!=0)
    Rprintf("free 24: %d\n",status);

  //clReleaseCommandQueue(clptr->cmdQueue);
  //clReleaseContext(clptr->context);
  
}
