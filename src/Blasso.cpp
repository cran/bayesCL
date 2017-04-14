#include "R.h"
extern "C"
{
#include "Rmath.h"
  
  
#include <time.h>
  
}
#ifdef USE_R
#include "R.h"
#include "Rmath.h"
#include <Rinternals.h>
#include <Rdefines.h>
#endif

#include "PolyaGamma.h"

#include "Blasso.h"
void Blasso::InitX(const unsigned int N, double *Xorigt,
                   const bool normalize)
{
  
  this->n = N;
  this->N = N;
  
  /* copy the input matrix */
  this->Xorig = (double*)malloc(sizeof(double)*n*M);
  for(unsigned int i=0;i<n*M;i++){
    this->Xorig[i]=Xorigt[i];
  }
  
  // /* calculate the mean of each column of X*/
  Xmean = (double*)malloc(sizeof(double)*M);
  for(unsigned int i=0;i<M;i++){
    Xmean[i]=0;
    for(unsigned int j=0;j<n;j++){
      Xmean[i]+=this->Xorig[j*M+i];
    }
    Xmean[i]=Xmean[i]/n;
  }
  
  /* center X */
  double *X = (double*)malloc(sizeof(double)*n*M);
  for(unsigned int i=0;i<M;i++){
    
    for(unsigned int j=0;j<n;j++){
      X[j*M+i]=Xorigt[j*M+i]-Xmean[i];
    }
  }
  
  /* normalize X, like Efron & Hastie */
  /* (presumably so that [good] lambda doesn't increase with n ??) */
  this->Xnorm_scale = 1.0;
  this->normalize = normalize;
  if (this->normalize)
  {
    Xnorm = (double*)malloc(sizeof(double)*M);
    for(unsigned int i=0;i<M;i++) Xnorm[i]=0.0;
    for(unsigned int i=0;i<M;i++){
      Xnorm[i]=0.0;
      for(unsigned int j=0;j<n;j++){
        Xnorm[i]+=X[j*M+i]*X[j*M+i];
      }
      Xnorm[i]=sqrt(Xnorm[i]);
    }
    for(unsigned int i=0;i<M;i++){
      
      for(unsigned int j=0;j<n;j++){
        X[j*M+i]=X[j*M+i]/Xnorm[i];
      }
    }
  }
  this->Xp=(double*)malloc(sizeof(double)*n*M);
  
  for (unsigned int i = 0; i < M; i++) for (unsigned int j = 0; j < n; j++)
    this->Xp[j*M+i] = X[j*M+pin[i]];
  free(X);
  
}

void Blasso::InitParams(double *beta, const double lambda2,
                        const double s2, double *tau2i)
{
  
  this->lambda2 = lambda2;
  this->s2 = s2;
  
  if (tau2i != NULL) {
    this->tau2i = (double*)malloc(sizeof(double)*M);
    for(unsigned int i=0;i<M;i++){
      this->tau2i[i]=tau2i[i];
    }
  }
  
  /* allocate beta */
  this->beta = (double*)malloc(sizeof(double)*M);
  
  /* norm the beta samples, like Efron and Hastie */
  if (normalize && this->M > 0) {
    for(unsigned int i=0;i<M;i++){
      beta[i]=beta[i]*Xnorm[i]*Xnorm_scale;
    }
  }
  
  for(unsigned int i=0;i<M;i++){
    this->beta[i]=beta[pin[i]];
  }
  
}

void Blasso::InitY(const unsigned int N, double *Y)
{
  /* center Y, and account for Rmiss if necessary */
  this->Y = (double*)malloc(sizeof(double)*n);
  unsigned int k, ell;
  k = ell = 0;
  
  /* for each entry of Y */
  Ymean = 0.0;
  for (unsigned int i = 0; i < N; i++)
  {
    /* copy Y to this->Y */
    this->Y[k] = Y[i];
    Ymean += this->Y[k];
    k++;
  }
  
  Ymean /= n;
  
  /* undo Ymean calculation if no intercept in the model */
  if (!icept) Ymean = 0.0;
  
  /* do not center Y if there is an explicit intercept in the model */
  if (icept){
    for (unsigned int i = 0; i < N; i++)
    {
      this->Y[i]-=Ymean;
    }
  }
  
  
  /* initialize the residual vector */
  resid = (double*)malloc(sizeof(double)*n);
  for (unsigned int i = 0; i < N; i++)
  {
    resid[i]=this->Y[i];
  }
  if (M > 0) {
    for(unsigned int i=0;i<N;i++){
      for(unsigned int j=0;j<M;j++){
        resid[i]+=-1.0*Xp[i*M+j]*beta[j];
      }
    }
  }
  
  /* for calculating t(X) %*% Y -- filled in UpdateXY */
  XtY = (double*)malloc(sizeof(double)*M);
  
  UpdateXY();
  
}

void Blasso::UpdateXY(void)
{
  /* calculate t(X) %*% Y */
  double *DiXp;
  DiXp = Xp;
  for(unsigned int j=0;j<M;j++){
    XtY[j]=0.0;
    for(unsigned int i=0;i<N;i++){
      XtY[j]+=DiXp[i*M+j]*this->Y[i];
    }
  }
  
  /* calculate YtY possibly using imega */
  YtY=0.0;
  for(unsigned int i=0;i<N;i++){
    YtY+=this->Y[i]*this->Y[i];
  }
}

BayesReg* new_BayesReg(const unsigned int M, const unsigned int n,
                       double *Xp, double *DiXp,CLptrs *clptr)
{
  /* allocate the structure */
  BayesReg *breg = (BayesReg*)malloc(sizeof(struct bayesreg));
  breg->M = M;
  
  /* allocate A and XtX-diag */
  breg->A = (float*)malloc(sizeof(float)*M*M);
  
  /* allocate the rest and return */
  /* fill A with t(X) %*% Di %*% X -- i.e. XtDxsX */
  
  
  alloc_rest_BayesReg(breg);
  return(breg);
}

void alloc_rest_BayesReg(BayesReg* breg)
{
  /* utility for Gibbs sampling parameters for beta */
  breg->Ai = (float*)malloc(sizeof(float)*breg->M*breg->M);
  breg->ABmu = (float*)malloc(sizeof(float)*breg->M);
  /* allocate the Gibbs sampling parameters for beta */
  breg->bmu = (float*)malloc(sizeof(float)*breg->M);
  breg->Vb = (float*)malloc(sizeof(float)*breg->M*breg->M);
}


/*
* refresh_Vb:
*
* copy breg->Ai*s2 into the breg->Vb matrix and
* subsequently reset the Vb_state indicator
*/

void refresh_Vb(BayesReg *breg, const double s2)
{
  /* compute: Vb = s2*Ai */
  for(unsigned int i=0;i<breg->M;i++){
    for(unsigned int j=0;j<breg->M;j++){
      breg->Vb[i*breg->M+j]=breg->Ai[i*breg->M+j];
    }
  }
  for(unsigned int i=0;i<breg->M*breg->M;i++)
    breg->Vb[i]*=s2;
  
}

double p1=0.0,p2=0.0,p3=0.0,p4=0.0,p5=0.0;

int ppp=0;
bool compute_BayesReg(const unsigned int M, double *XtY, double *tau2i,
                      const double lambda2, const double s2, BayesReg *breg, CLptrs *clptr, Blasso *bl)
{
  cl_ulong time_start, time_end;
  cl_int st;
  float* V1=(float*)malloc(M*M*sizeof(float));
  for(unsigned int i=0;i<M;i++)
    V1[i]=tau2i[i];
  cl_int status;
  size_t single[1]={1};
  clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->dummy, 1, NULL, single, NULL, 0, NULL,&(clptr->ev6));
  clWaitForEvents(1, &(clptr->ev6));
  status=clEnqueueWriteBuffer(clptr->cmdQueue, bl->tau_mem, CL_TRUE, 0,sizeof(float)*M,V1, 0, NULL, NULL);
  if(status!=0)
    Rprintf("error 20\n");
  
  clSetKernelArg(clptr->kernelInitChol, 0, sizeof(cl_mem), (void *)&(bl->a_mem));
  clSetKernelArg(clptr->kernelInitChol, 1, sizeof(cl_mem), (void *)&(bl->tau_mem));
  clSetKernelArg(clptr->kernelInitChol, 2, sizeof(cl_mem), (void *)&(bl->a2_mem));
  clSetKernelArg(clptr->kernelInitChol, 3, sizeof(int), (void *)&(M));
  size_t globalInit[2]={M,M};
  st=clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->kernelInitChol, 2, NULL, globalInit, NULL, 0, NULL,NULL);
  if(st!=0)
    Rprintf("error 21\n");
  if(st!=0)
    Rprintf("st111a: %d\n",st);
  cl_int status1,status2;
  bl->offset=0;
  size_t  globalUpdate[2];
  size_t single_thread[1];
  size_t local_single[1];
  size_t globalLeftUpdate[2];
  
  size_t localUpdate[2]={16,16};
  clSetKernelArg(clptr->kernelChol, 4, sizeof(int), (void *)&(bl->block));
  
  int threadsGlobal, threadsLeft;
  while((unsigned)(bl->offset+bl->block)<(M)){
    threadsLeft=M-bl->offset-bl->block;
    globalLeftUpdate[0]=((threadsLeft+15)/16)*16;
    globalLeftUpdate[1]=((bl->block+15)/16)*16;
    
    threadsGlobal=M-bl->offset-bl->block;
    globalUpdate[0]=((threadsGlobal+15)/16)*16;
    globalUpdate[1]=((threadsGlobal+15)/16)*16;
    
    
    
    clSetKernelArg(clptr->kernelLeft, 3, sizeof(int), (void *)&(bl->offset));
    clSetKernelArg(clptr->kernelLeft, 6, sizeof(int), (void *)&(threadsLeft));
    
    clSetKernelArg(clptr->kernelUpdate, 2, sizeof(int), (void *)&(bl->offset));
    clSetKernelArg(clptr->kernelUpdate, 5, sizeof(int), (void *)&threadsGlobal);
    clSetKernelArg(clptr->kernelChol, 2, sizeof(int), (void *)&(bl->offset));
    single_thread[0]=bl->block;
    local_single[0]=bl->block;
    status1=clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->kernelChol, 1, NULL, single_thread, local_single, 0, NULL,NULL);
    if(status1!=0)
      Rprintf("st1: %d\n",status1);
    st=clFinish(clptr->cmdQueue);
    if(st!=0)
      Rprintf("st333: %d\n",st);
    status2=clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->kernelLeft, 2, NULL, globalLeftUpdate, localUpdate, 0, NULL,NULL);
    if(status2!=0)
      Rprintf("st1: %d\n",status2);
    st=clFinish(clptr->cmdQueue);
    if(st!=0)
      Rprintf("st333: %d\n",st);
    status2=clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->kernelUpdate, 2, NULL, globalUpdate, localUpdate, 0, NULL,NULL);
    if(status2!=0)
      Rprintf("st3: %d\n",status2);
    st=clFinish(clptr->cmdQueue);
    if(st!=0)
      Rprintf("st333: %d\n",st);
    bl->offset+=bl->block;
  }
  
  int left=M-bl->offset;
  if(left>0){
    single_thread[0]=left;
    local_single[0]=left;
    clSetKernelArg(clptr->kernelChol, 4, sizeof(int), (void *)&left);
    clSetKernelArg(clptr->kernelChol, 2, sizeof(int), (void *)&(bl->offset));
    status2=clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->kernelChol, 1, NULL, single_thread, local_single, 0, NULL,NULL);
    if(status2!=0)
      Rprintf("error 22a\n");
  }
  size_t global_Zero[2]={M,M};
  clSetKernelArg(clptr->kernelZero, 0, sizeof(cl_mem), (void *)&(bl->a_mem));
  clSetKernelArg(clptr->kernelZero, 1, sizeof(int), (void *)&(M));
  status2=clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->kernelZero, 2, NULL, global_Zero, NULL, 0, NULL,NULL);
  if(status2!=0)
    Rprintf("error 22b\n");
  
  clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->dummy, 1, NULL, single, NULL, 0, NULL,&(clptr->ev7));
  clWaitForEvents(1, &(clptr->ev7));
  clGetEventProfilingInfo(clptr->ev6, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL); 
  clGetEventProfilingInfo(clptr->ev7, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL); 
  bl->time1 += (time_end - time_start)/ 1000000000.0;
  clReleaseEvent(clptr->ev6);
  clReleaseEvent(clptr->ev7);
  
  clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->dummy, 1, NULL, single, NULL, 0, NULL,&(clptr->ev6));
  clWaitForEvents(1, &(clptr->ev6));
  st |= clSetKernelArg(clptr->kernelInv1, 0, sizeof(cl_mem), (void *)&(bl->a_mem));
  st |= clSetKernelArg(clptr->kernelInv1, 1, sizeof(int), (void *)&(bl->remainder));
  st |= clSetKernelArg(clptr->kernelInv1, 2, sizeof(int), (void *)&(bl->part_size_fixed));
  st |= clSetKernelArg(clptr->kernelInv1, 3, sizeof(int), (void *)&M);
  
  size_t globalInv1[1]={(unsigned )bl->parts};
  size_t localInv1[1]={1};
  st = clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->kernelInv1, 1, NULL, globalInv1, localInv1, 0, NULL,NULL);
  
  if(st!=0)
    Rprintf("a55: %d\n",st);
  int repeat=1;
  
  size_t global2[3]={(unsigned )bl->parts,1,1};
  size_t local2[3]={32,32,1};
  
  for(int pp=bl->parts;pp>1;pp/=2){
    
    st = clSetKernelArg(clptr->kernelInv2, 0, sizeof(cl_mem), (void *)&(bl->a_mem));
    st = clSetKernelArg(clptr->kernelInv2, 1, sizeof(cl_mem), (void *)&(bl->sizes_mem));
    st = clSetKernelArg(clptr->kernelInv2, 2, sizeof(cl_mem), (void *)&(bl->MM_mem));
    st = clSetKernelArg(clptr->kernelInv2, 3, sizeof(int), (void *)&(repeat));
    st = clSetKernelArg(clptr->kernelInv2, 4, sizeof(int), (void *)&(bl->remainder));
    st = clSetKernelArg(clptr->kernelInv2, 5, sizeof(int), (void *)&(bl->part_size_fixed));
    st = clSetKernelArg(clptr->kernelInv2, 6, sizeof(int), (void *)&M);
    st = clSetKernelArg(clptr->kernelInv3, 0, sizeof(cl_mem), (void *)&(bl->a_mem));
    st = clSetKernelArg(clptr->kernelInv3, 1, sizeof(cl_mem), (void *)&(bl->sizes_mem));
    st = clSetKernelArg(clptr->kernelInv3, 2, sizeof(cl_mem), (void *)&(bl->MM_mem));
    st = clSetKernelArg(clptr->kernelInv3, 3, sizeof(int), (void *)&(repeat));
    st = clSetKernelArg(clptr->kernelInv3, 4, sizeof(int), (void *)&(bl->remainder));
    st = clSetKernelArg(clptr->kernelInv3, 5, sizeof(int), (void *)&(bl->part_size_fixed));
    st = clSetKernelArg(clptr->kernelInv3, 6, sizeof(int), (void *)&M);
    global2[2]=pp/2;
    global2[0]=(((bl->part_size_fixed+1)*repeat+31)/32)*32;
    global2[1]=(((bl->part_size_fixed+1)*repeat+31)/32)*32/4;
    local2[0]=32;
    local2[1]=32/4;
    
    st = clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->kernelInv2, 3, NULL, global2, local2, 0, NULL,NULL );
    if(st!=0)
      Rprintf("st1: %d\n",st);
    
    
    global2[0]=(((bl->part_size_fixed+1)*repeat+31)/32)*32;
    global2[1]=(((bl->part_size_fixed+1)*repeat+31)/32)*32/4;
    local2[0]=32;
    local2[1]=32/4;
    st = clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->kernelInv3, 3, NULL, global2, local2, 0, NULL,NULL );
    
    repeat*=2;
  }
  clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->dummy, 1, NULL, single, NULL, 0, NULL,&(clptr->ev7));
  clWaitForEvents(1, &(clptr->ev7));
  clGetEventProfilingInfo(clptr->ev6, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL); 
  clGetEventProfilingInfo(clptr->ev7, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL); 
  bl->time2 += (time_end - time_start)/ 1000000000.0;
  clReleaseEvent(clptr->ev6);
  clReleaseEvent(clptr->ev7);
  
  clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->dummy, 1, NULL, single, NULL, 0, NULL,&(clptr->ev6));
  clWaitForEvents(1, &(clptr->ev6));
  
  int prodBlock=clptr->MMThreads;
  int Mpad=((M+prodBlock-1)/prodBlock)*prodBlock;
  size_t global[2]={(unsigned)Mpad,(unsigned)Mpad/clptr->WPT};
  size_t local[2]={(unsigned)prodBlock,(unsigned)prodBlock/clptr->WPT};
  
  st=clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->kernelInvProd, 2, NULL, global, local, 0, NULL,NULL);
  if(st!=0)
    Rprintf("error 1111a: %d\n", st);
  
  clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->dummy, 1, NULL, single, NULL, 0, NULL,&(clptr->ev7));
  clWaitForEvents(1, &(clptr->ev7));
  clGetEventProfilingInfo(clptr->ev6, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL); 
  clGetEventProfilingInfo(clptr->ev7, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL); 
  bl->time3 += (time_end - time_start)/ 1000000000.0;
  clReleaseEvent(clptr->ev6);
  clReleaseEvent(clptr->ev7);
  
  
  st=clEnqueueReadBuffer(clptr->cmdQueue, bl->a1_mem, CL_TRUE, 0,M*M*sizeof(float),breg->Ai, 0, NULL, NULL);
  
  for (unsigned int i = 0; i < M; i++){
    breg->bmu[i]=0.0;
    for (unsigned int j = 0; j < M; j++) {
      breg->bmu[i]+=breg->Ai[i*M+j]*XtY[j];
    }
  }
  float s2_float;
  s2_float=s2;
  st |= clSetKernelArg(clptr->kernelAddS2, 0, sizeof(cl_mem), (void *)&(bl->a1_mem));
  st |= clSetKernelArg(clptr->kernelAddS2, 1, sizeof(cl_mem), (void *)&(bl->a_mem));
  st |= clSetKernelArg(clptr->kernelAddS2, 2, sizeof(float), (void *)&(s2_float));
  st |= clSetKernelArg(clptr->kernelAddS2, 3, sizeof(int), (void *)&M);
  
  size_t globalAddS2[2]={(unsigned )M,(unsigned )M};
  
  st = clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->kernelAddS2, 2, NULL, globalAddS2, NULL, 0, NULL,NULL);
  
  free(V1);
  return true;
}
Blasso::Blasso(const unsigned int M, const unsigned int n, double *X,
               double *Y, double *beta, const double lambda2, const double s2,
               double *tau2i, const double r,
               const double delta, const double a, const double b,
               const bool normalize, CLptrs *clptr)
{
  
  
  this->beta = this->rn = this->BtDi = NULL;
  this->tau2i = NULL;
  this->r = r;  this->delta = delta;  this->icept = icept;
  pin = NULL;
  this->M = M;
  
  pin = (int*) malloc(sizeof(int)*M);
  unsigned int j = 0;
  for (unsigned int i = 0; i < M; i++)
  {
    pin[j++] = i;
  }
  
  /* initialize the input data */
  InitX(n, X, normalize);
  /* this function will be modified to depend on whether OLS must become lasso */
  InitParams(beta, lambda2, s2, tau2i);
  /* Y initization must come after InitParams */
  InitY(N, Y);
  /* set the s2 Inv-Gamma prior */
  this->a = a;
  this->b = b;
  this->inverseTime=0.0;
  this->drawBetaTime=0.0;
  this->part1Time=0.0;
  this->part2Time=0.0;
  this->time1=0.0;
  this->time2=0.0;
  this->time3=0.0;
  this->time4=0.0;
  this->time5=0.0;
  this->time6=0.0;
  this->time7=0.0;
  this->time8=0.0;
  /* must call ::Init function first thing */
  breg = NULL;
  
  breg = new_BayesReg(M, n, Xp, Xp, clptr);
  
  BtDi = (double*)malloc(sizeof(double)*M);
  rn = (double*)malloc(sizeof(double)*M);
  
  int abc=0;
  abc=log(M*1.0)/log(2.0);
  abc=abc+(clptr->FWDblocks);
  this->block=clptr->Cholblock;
  this->offset=0;
  this->parts=pow(2,abc);
  this->datasize = sizeof(float)*M*M;
  this->datasize_temp=sizeof(float)*this->block*this->block;
  this->datasize_leftUpdate=sizeof(float)*M*this->block;
  this->datasize_Update=sizeof(float)*M*M;
  
  float* VV;
  Rprintf("M: %d, N: %d\n",M,n);
  VV=(float*)malloc(sizeof(float)*M*n);
  
  for(unsigned int i=0;i<M*n;i++)
    VV[i]=Xp[i];
  
  int *sizes;
  sizes=(int*)malloc(sizeof(int)*parts);
  
  this->part_size_fixed=M/parts;
  this->part_size=part_size_fixed+1;
  this->remainder=M%parts;
  
  for(int i=0;i<parts;i++){
    if(i<remainder)
      sizes[i]=part_size_fixed+1;
    else
      sizes[i]=part_size_fixed;
  }
  
  this->a_mem = clCreateBuffer(clptr->context, CL_MEM_READ_WRITE,this->datasize, NULL, &(this->status));
  if(this->status!=0)
    Rprintf("error 1\n");
  this->a2_mem = clCreateBuffer(clptr->context, CL_MEM_READ_WRITE,sizeof(float)*M*M, NULL, &(this->status));
  if(this->status!=0)
    Rprintf("error 2\n");
  this->Xp_mem = clCreateBuffer(clptr->context, CL_MEM_READ_ONLY,sizeof(float)*M*n, NULL, &(this->status));
  if(this->status!=0)
    Rprintf("error 3\n");
  this->tau_mem = clCreateBuffer(clptr->context, CL_MEM_READ_WRITE,sizeof(float)*M, NULL, &(this->status));
  if(this->status!=0)
    Rprintf("error 4\n");
  this->temp_mem = clCreateBuffer(clptr->context, CL_MEM_READ_WRITE,this->datasize_temp, NULL, &(this->status));
  if(this->status!=0)
    Rprintf("error 5\n");
  this->V_mem = clCreateBuffer(clptr->context, CL_MEM_READ_WRITE,this->datasize_temp, NULL, &(this->status));
  if(this->status!=0)
    Rprintf("error 6\n");
  this->l_mem = clCreateBuffer(clptr->context, CL_MEM_READ_WRITE,this->datasize_leftUpdate, NULL, &(this->status));
  if(this->status!=0)
    Rprintf("error 7\n");
  this->v1_mem = clCreateBuffer(clptr->context, CL_MEM_READ_WRITE,sizeof(float)*M*M, NULL, &(this->status));
  if(this->status!=0)
    Rprintf("error 8\n");
  this->a1_mem = clCreateBuffer(clptr->context, CL_MEM_READ_WRITE,sizeof(float)*M*M, NULL, &(this->status));
  if(this->status!=0)
    Rprintf("error 9\n");
  this->sizes_mem = clCreateBuffer(clptr->context, CL_MEM_READ_ONLY,sizeof(int)*this->parts, NULL, &(this->status));
  if(this->status!=0)
    Rprintf("error 10\n");
  this->MM_mem = clCreateBuffer(clptr->context, CL_MEM_READ_WRITE,sizeof(float)*M*M*2, NULL, &(this->status));
  if(this->status!=0)
    Rprintf("error 11\n");
  
  status=clEnqueueWriteBuffer(clptr->cmdQueue, this->sizes_mem, CL_TRUE, 0,sizeof(int)*this->parts,sizes, 0, NULL, NULL);
  if(this->status!=0)
    Rprintf("error 12\n");
  status=clEnqueueWriteBuffer(clptr->cmdQueue, this->Xp_mem, CL_TRUE, 0,sizeof(float)*M*n,VV, 0, NULL, NULL);
  if(this->status!=0)
    Rprintf("error 13\n");
  status=clSetKernelArg(clptr->kernelInvProd2, 0, sizeof(cl_mem), (void *)&(this->Xp_mem));
  if(this->status!=0)
    Rprintf("error 14\n");
  status=clSetKernelArg(clptr->kernelInvProd2, 1, sizeof(cl_mem), (void *)&(this->a2_mem));
  if(this->status!=0)
    Rprintf("error 15\n");
  status=clSetKernelArg(clptr->kernelInvProd2, 2, sizeof(int), (void *)&M);
  if(this->status!=0)
    Rprintf("error 16\n");
  status=clSetKernelArg(clptr->kernelInvProd2, 3, sizeof(int), (void *)&n);
  if(this->status!=0)
    Rprintf("error 17\n");
  int prodBlock=16;
  int Mpad=((M+prodBlock-1)/prodBlock)*prodBlock;
  size_t global[2]={(unsigned)Mpad,(unsigned)Mpad};
  size_t local[2]={(unsigned)prodBlock,(unsigned)prodBlock};
  
  this->status=clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->kernelInvProd2, 2, NULL, global, local, 0, NULL,NULL);
  if(this->status!=0)
    Rprintf("error 18\n");
  size_t single[1]={1};
  cl_ulong time_start, time_end;
  
  free(VV);
  clSetKernelArg(clptr->kernelChol, 0, sizeof(cl_mem), (void *)&(this->temp_mem));
  clSetKernelArg(clptr->kernelChol, 1, sizeof(cl_mem), (void *)&(this->a_mem));
  clSetKernelArg(clptr->kernelChol, 2, sizeof(int), (void *)&(this->offset));
  clSetKernelArg(clptr->kernelChol, 3, sizeof(int), (void *)&M);
  clSetKernelArg(clptr->kernelChol, 4, sizeof(int), (void *)&(this->block));
  clSetKernelArg(clptr->kernelChol, 5, sizeof(cl_mem), (void *)&(this->V_mem));
  
  clSetKernelArg(clptr->kernelLeft, 0, sizeof(cl_mem), (void *)&(this->l_mem));
  clSetKernelArg(clptr->kernelLeft, 1, sizeof(cl_mem), (void *)&(this->a_mem));
  clSetKernelArg(clptr->kernelLeft, 2, sizeof(cl_mem), (void *)&(this->V_mem));
  clSetKernelArg(clptr->kernelLeft, 3, sizeof(int), (void *)&(this->offset));
  clSetKernelArg(clptr->kernelLeft, 4, sizeof(int), (void *)&(this->block));
  clSetKernelArg(clptr->kernelLeft, 5, sizeof(int), (void *)&M);
  
  clSetKernelArg(clptr->kernelUpdate, 0, sizeof(cl_mem), (void *)&(this->l_mem));
  clSetKernelArg(clptr->kernelUpdate, 1, sizeof(cl_mem), (void *)&(this->a_mem));
  clSetKernelArg(clptr->kernelUpdate, 2, sizeof(int), (void *)&(this->offset));
  clSetKernelArg(clptr->kernelUpdate, 3, sizeof(int), (void *)&(this->block));
  clSetKernelArg(clptr->kernelUpdate, 4, sizeof(int), (void *)&M);
  
  clSetKernelArg(clptr->kernelInvProd, 0, sizeof(cl_mem), (void *)&(this->a_mem));
  clSetKernelArg(clptr->kernelInvProd, 1, sizeof(cl_mem), (void *)&(this->a1_mem));
  status=clSetKernelArg(clptr->kernelInvProd, 2, sizeof(int), (void *)&M);
  if(this->status!=0)
    Rprintf("error 19\n");
  free(sizes);
  clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->dummy, 1, NULL, single, NULL, 0, NULL,&(clptr->ev2));
  clWaitForEvents(1, &(clptr->ev2));
  compute_BayesReg(M, XtY, tau2i, lambda2, s2, breg, clptr, this);
  clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->dummy, 1, NULL, single, NULL, 0, NULL,&(clptr->ev3));
  clWaitForEvents(1, &(clptr->ev3));
  clGetEventProfilingInfo(clptr->ev2, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL); 
  clGetEventProfilingInfo(clptr->ev3, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL); 
  this->inverseTime += (time_end - time_start)/ 1000000000.0;
  clReleaseEvent(clptr->ev2);
  clReleaseEvent(clptr->ev3);  
}

/*
* draw_tau2i_lasso:
*
* draw the latent inverse tau2 vector posterior conditional(s)
* under the lasso (double exponential) prior
*/
double rinvgauss(const double mu, const double lambda)
{
  double u, y, x1, mu2, l2;
  double tmp=norm_rand();
  y = tmp*tmp;
  mu2 = mu*mu;
  l2 = 2 * lambda;
  x1 = mu + mu2*y / l2 - (mu / l2)* sqrt(4 * mu*lambda*y + mu2*y*y);
  
  u = unif_rand();
  if (u <= mu / (mu + x1)) return x1;
  else return mu2 / x1;
}

void draw_tau2i_lasso(const unsigned int M, double *tau2i, double *beta,
                      double lambda2, double s2)
{
  double l_numer;
  
  /* part of the mu parameter to the inv-gauss distribution */
  l_numer = log(lambda2) + log(s2);
  
  for (unsigned int j = 0; j < M; j++) {
    
    /* the rest of the mu parameter */
    double l_mup = 0.5*l_numer - log(fabs(beta[j]));
    
    /* sample from the inv-gauss distn */
    double tau2i_temp = rinvgauss(exp(l_mup), lambda2);
    
    tau2i[j] = tau2i_temp;
  }
}

float a1=0.0,a2=0.0,a3=0.0,a4=0.0,a5=0.0;

void Blasso::Draw(const unsigned int thin, const bool fixnu,CLptrs *clptr)
{
  
  
  size_t single[1]={1};
  cl_ulong time_start, time_end;
  for (unsigned int t = 0; t < thin; t++)
  {
    
    
    draw_tau2i_lasso(M, tau2i, beta, lambda2, s2);
    
    clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->dummy, 1, NULL, single, NULL, 0, NULL,&(clptr->ev2));
    clWaitForEvents(1, &(clptr->ev2));
    compute_BayesReg(M, XtY, tau2i, lambda2, s2, breg, clptr, this);
    clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->dummy, 1, NULL, single, NULL, 0, NULL,&(clptr->ev3));
    clWaitForEvents(1, &(clptr->ev3));
    clGetEventProfilingInfo(clptr->ev2, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL); 
    clGetEventProfilingInfo(clptr->ev3, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL); 
    this->inverseTime += (time_end - time_start)/ 1000000000.0;
    clReleaseEvent(clptr->ev2);
    clReleaseEvent(clptr->ev3);
    
    double shape = ((double)M) + r;
    double rate = 0.0;
    for (unsigned int j = 0; j < M; j++) {
      if (tau2i[j] == 0) { 
        shape -= 1;
      }else{
        rate += 1.0 / tau2i[j];
      }
    }
    rate = rate / 2.0 + delta;
    
    lambda2 = rgamma(shape, 1.0 / rate);
    
    clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->dummy, 1, NULL, single, NULL, 0, NULL,&(clptr->ev2));
    clWaitForEvents(1, &(clptr->ev2));
    
    //clEnqueueWriteBuffer(clptr->cmdQueue, this->a_mem, CL_TRUE, 0,sizeof(int)*M*M,breg->Vb, 0, NULL, NULL);
    cl_int status1,status2;
    this->offset=0;
    size_t  globalUpdate[2];
    size_t single_thread[1];
    size_t local_single[1];
    size_t globalLeftUpdate[2];
    
    size_t localUpdate[2]={16,16};
    clSetKernelArg(clptr->kernelChol, 4, sizeof(int), (void *)&(this->block));
    
    int threadsGlobal, threadsLeft;
    while((unsigned)(this->offset+this->block)<(M)){
      threadsLeft=M-this->offset-this->block;
      globalLeftUpdate[0]=((threadsLeft+15)/16)*16;
      globalLeftUpdate[1]=((this->block+15)/16)*16;
      
      threadsGlobal=M-this->offset-this->block;
      globalUpdate[0]=((threadsGlobal+15)/16)*16;
      globalUpdate[1]=((threadsGlobal+15)/16)*16;
      
      clSetKernelArg(clptr->kernelLeft, 3, sizeof(int), (void *)&(this->offset));
      clSetKernelArg(clptr->kernelLeft, 6, sizeof(int), (void *)&(threadsLeft));
      
      clSetKernelArg(clptr->kernelUpdate, 2, sizeof(int), (void *)&(this->offset));
      clSetKernelArg(clptr->kernelUpdate, 5, sizeof(int), (void *)&threadsGlobal);
      clSetKernelArg(clptr->kernelChol, 2, sizeof(int), (void *)&(this->offset));
      single_thread[0]=this->block;
      local_single[0]=this->block;
      status1=clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->kernelChol, 1, NULL, single_thread, local_single, 0, NULL,NULL);
      if(status1!=0)
        Rprintf("st1: %d\n",status1);
      status2=clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->kernelLeft, 2, NULL, globalLeftUpdate, localUpdate, 0, NULL,NULL);
      if(status2!=0)
        Rprintf("st1: %d\n",status2);
      status2=clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->kernelUpdate, 2, NULL, globalUpdate, localUpdate, 0, NULL,NULL);
      if(status2!=0)
        Rprintf("st3: %d\n",status2);
      this->offset+=this->block;
      
    }
    
    int left=M-this->offset;
    if(left>0){
      single_thread[0]=left;
      local_single[0]=left;
      clSetKernelArg(clptr->kernelChol, 4, sizeof(int), (void *)&left);
      clSetKernelArg(clptr->kernelChol, 2, sizeof(int), (void *)&(this->offset));
      status2=clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->kernelChol, 1, NULL, single_thread, local_single, 0, NULL,NULL);
    }
    size_t global_Zero[2]={M,M};
    clSetKernelArg(clptr->kernelZero, 0, sizeof(cl_mem), (void *)&(this->a_mem));
    clSetKernelArg(clptr->kernelZero, 1, sizeof(int), (void *)&(M));
    status2=clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->kernelZero, 2, NULL, global_Zero, NULL, 0, NULL,NULL);
    clEnqueueReadBuffer(clptr->cmdQueue, this->a_mem, CL_TRUE, 0,M*M*sizeof(float),breg->Vb, 0, NULL, NULL);
    
    
    for (unsigned int i = 0; i < M; i++) rn[i] = norm_rand();
    
    unsigned int i, j;
    for (j = 0; j < M; j++) {
      beta[j] = breg->bmu[j];
      for (i = 0; i < j + 1; i++) beta[j] += breg->Vb[j*M+i] * rn[i];
    }
    
    for (unsigned int i = 0; i < n; i++)  resid[i]=Y[i];
    
    for(unsigned int i=0;i<N;i++){
      for(unsigned int j=0;j<M;j++){
        resid[i]+=-1.0*Xp[i*M+j]*beta[j];
      }
    }
    double sums2=0.0;
    for(unsigned int i=0;i<n;i++) sums2+=resid[i]*resid[i];
    
    double BtDiB;
    if (M > 0)
    {
      for(unsigned int j=0;j<M;j++) BtDi[j]=beta[j];
      if (tau2i){
        for(unsigned int j=0;j<M;j++)  BtDi[j]*=tau2i[j];
      } else {
        for(unsigned int j=0;j<M;j++)  BtDi[j]*=1.0 / lambda2;
      }
      BtDiB = 0.0;
      for(unsigned int j=0;j<M;j++) BtDiB+=BtDi[j]*beta[j];
    }
    else 
      BtDiB = 0.0;
    
    
    
    shape = a;
    shape += (n - 1) / 2.0 + (M) / 2.0;
    
    double scale = b + sums2 / 2.0 + BtDiB / 2.0;
    
    s2 = 1.0 / rgamma(shape, 1.0 / scale);
    
    clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->dummy, 1, NULL, single, NULL, 0, NULL,&(clptr->ev3));
    clWaitForEvents(1, &(clptr->ev3));
    clGetEventProfilingInfo(clptr->ev2, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL); 
    clGetEventProfilingInfo(clptr->ev3, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL); 
    this->drawBetaTime += (time_end - time_start)/ 1000000000.0;
    clReleaseEvent(clptr->ev2);
    clReleaseEvent(clptr->ev3);
  }
}
void Blasso::GetParams(double *mu, double *beta, unsigned int *M, double *s2,
                       double *tau2i, double *lambda2) const
{
  if (icept) *mu = rnorm(Ymean, sqrt(this->s2 / n));
  else *mu = 0;
  
  for(unsigned int i=0;i<*M;i++) beta[i]=0.0;
  
  for(unsigned int i=0;i<*M;i++) beta[pin[i]]=this->beta[i];
  
  *s2 = this->s2;
  if (tau2i)
  {
    unsigned int MIndex = (*M);
    for (unsigned int i = 0; i < MIndex; i++) tau2i[i] = -1.0;
    if (this->M > 0 && this->tau2i){
      for(unsigned int i=0;i<*M;i++) tau2i[pin[i]]=this->tau2i[i];
    }
  }
  if (lambda2) *lambda2 = this->lambda2;
}

void computeMuResidualsByParts(unsigned int n, unsigned int M, unsigned int T, double* X, double* beta, double* outMuResiduals)
{
  // Xbeta = Txn
  // Xorig: nxM
  // beta mat: TxM
  // compute beta matrix subset size: Tpartial x M and how many parts there are
  unsigned int maxNumMatrixElementsPerPart = 1000 * 1000 * 10;
  unsigned int requiredNumMatrixElements = T*n;
  unsigned int numRequiredParts = requiredNumMatrixElements / maxNumMatrixElementsPerPart;
  unsigned int numResidualElements = requiredNumMatrixElements - maxNumMatrixElementsPerPart * numRequiredParts;
  if (numResidualElements > 0)
  {
    numRequiredParts += 1;
  }
  unsigned int Tpartial = (T / numRequiredParts);
  if ((T % numRequiredParts) > 0 && numRequiredParts > 1)
  {
    Tpartial = T / (numRequiredParts - 1);
  }
  
  
  // compute adjustment by parts
  double* betaPartial =  (double*)malloc(sizeof(double)*M*Tpartial);
  double* Xbeta = (double*)malloc(sizeof(double)*n*Tpartial);
  double* muResiduals = (double*)malloc(sizeof(double)*Tpartial*numRequiredParts);
  
  for (unsigned int partIndex = 0; partIndex < numRequiredParts; ++partIndex)
  {
    for (unsigned int rowIndex = 0; rowIndex < Tpartial; ++rowIndex)
    {
      unsigned int row = rowIndex + partIndex*Tpartial;
      row = row >= T ? (T - 1) : row;
      
      for (unsigned int colIndex = 0; colIndex < M; ++colIndex)
      {
        betaPartial[rowIndex*M+colIndex] = beta[row*M+colIndex];
      }
    }
    for(unsigned int i=0;i<Tpartial;i++){
      for(unsigned int j=0;j<n;j++){
        Xbeta[i*n+j]=0.0;
        for(unsigned int k=0;k<M;k++){
          Xbeta[i*n+j]+=X[j*M+k]*betaPartial[i*M+k];
        }
      }
    }
    //linalg_dgemm(CblasTrans, CblasNoTrans, n, Tpartial, M, 1.0, X, M, betaPartial, M, 0.0, Xbeta, n);
    double* offsetedMuResiduals = muResiduals + partIndex*Tpartial; // pointer arithemtic so wmean_of_rows writes to correct offseted location
    //wmean_of_rows(offsetedMuResiduals, Xbeta, Tpartial, n, NULL);
    for (unsigned int i = 0; i < Tpartial; i++) {
      offsetedMuResiduals[i] = 0;
      for (unsigned int j = 0; j < n; j++) offsetedMuResiduals[i] += Xbeta[i*n+j];
      offsetedMuResiduals[i] = offsetedMuResiduals[i] / n;
    }
  }
  // copy only the number of required mu residuals to output result
  for (unsigned int i = 0; i < T; ++i)
  {
    outMuResiduals[i] = muResiduals[i];
  }
  free(betaPartial);
  free(Xbeta);
  free(muResiduals);
}
/*
* Rounds:
*
* perform T rounds of mcmc/Gibbs sampling the lasso parameters
* beta, s2, tau2i and lambda -- taking thin number of them before
* returning one set.
*/

void Blasso::Rounds(const unsigned int T, const unsigned int thin,
                    double *mu, double *beta, unsigned int *M, double *s2,
                    double *tau2i, double *lambda2,CLptrs *clptr)
{
  /* for helping with periodic interrupts */
  /* assume that the initial values reside in position 0 */
  /* do T-1 MCMC rounds */
  for (unsigned int t = 0; t < T; t++) {
    /* do thin number of MCMC draws */
    Draw(thin, false,clptr);
    
    double *tau2i_samp = NULL;
    if (tau2i) tau2i_samp = &(tau2i[t*(*M)]);
    // 
    // 
    /* if LASSO/HORSESHE/NG/RIDGE */
    double *lambda2_samp = NULL;
    if (lambda2)
    {
      lambda2_samp = &(lambda2[t]);
    }
    
    /* copy the sampled parameters */
    GetParams(&(mu[t]), &(beta[t*(*M)]), M, &(s2[t]), tau2i_samp, lambda2_samp);
    
    /* print progress meter */
    if (t > 0 && ((t + 1) % 100 == 0))
      Rprintf("t=%d, m=%d\n", t + 1, this->M);
    
  }
  
  /* (un)-norm the beta samples, like Efron and Hastie */
  if (normalize) {
    for(unsigned int i=0;i<*M;i++){
      for(unsigned int j=0;j<T;j++){
        beta[j*(*M)+i]/=Xnorm[i];
      }  
    }
    for(unsigned int i=0;i<(*M)*T;i++) 
      beta[i]*=(1.0 / Xnorm_scale);
  }
  
  // Compute residual vector by parts, otherwise one very large matrix can get allocated and mess up RAM
  double* mu_resid = (double*)malloc(sizeof(double)*T);
  computeMuResidualsByParts(n, *M, T, Xorig, beta, mu_resid);
  
  for (unsigned t = 0; t < T; t++) {
    if (icept) mu[t] = mu[t] - mu_resid[t];
  }
  // clean up
  free(mu_resid);
}

void Blasso::Cleanup(){
  
  free(this->pin);
  free(this->BtDi);
  free(this->rn);
  free(this->Xorig);
  free(this->Xmean);
  free(this->Xnorm);
  free(this->Xp);
  free(this->Y);
  free(this->resid);
  free(this->tau2i);
  free(this->beta);
  free(this->breg->A);
  free(this->breg->ABmu);
  free(this->breg->bmu);
  free(this->breg->Vb);
  free(this->breg->Ai);
  free(this->breg);
  free(this->XtY);
}
extern "C"
{
  double* X;
  double* beta_mat;
  double* tau2i_mat;
  Blasso *blasso = NULL;
  //add pointer as in gibbs
  SEXP blassoGPU(SEXP RT, SEXP Rthin, SEXP RM, SEXP Rn, SEXP RX_in, SEXP RY, SEXP Rlambda2, SEXP Rmu, SEXP Rbeta, SEXP Rs2, SEXP Rtau2i, SEXP Rr, SEXP Rdelta, SEXP Ra, SEXP Rb, SEXP Rnormalize, SEXP Rtiming, SEXP ptr)
    
  {
    size_t single[1]={1};
    double seconds=0;
    cl_ulong time_start, time_end;
    CLptrs *clptr = (CLptrs *) R_ExternalPtrAddr(ptr);
    cl_int status;
    status=clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->dummy, 1, NULL, single, NULL, 0, NULL,&(clptr->ev1));
    if(status!=0)
      Rprintf("timing start: %d\n",status);
    clWaitForEvents(1, &(clptr->ev1)); 
    
    
    char names[6][50] = {"mu","beta","s2","lambda2","tau2i","timing"};
    
    SEXP list_names, list;
    PROTECT(list_names = Rf_allocVector(STRSXP,6));    
    
    for(unsigned int i = 0; i < 2; i++)   
      SET_STRING_ELT(list_names,i,Rf_mkChar(names[i])); 
    
    int *T,*thin,*n,*normalize;
    unsigned int *M;
    double *X_in,*Y, *lambda2, *mu,*beta,*s2,*tau2i,*r,*delta,*a,*b,*timing;
    
    T=INTEGER(RT);thin=INTEGER(Rthin);n=INTEGER(Rn);normalize=INTEGER(Rnormalize);
    M=(unsigned int*)INTEGER(RM);
    X_in=REAL(RX_in);Y=REAL(RY); lambda2=REAL(Rlambda2); mu=REAL(Rmu); beta=REAL(Rbeta); s2=REAL(Rs2); tau2i=REAL(Rtau2i); r=REAL(Rr); delta=REAL(Rdelta); a=REAL(Ra);b=REAL(Rb);timing=REAL(Rtiming);
    
    //xin
    X=(double*)malloc((*n)*(*M)*sizeof(double));
    for(int i=0;i<(*n);i++){
      for(unsigned int j=0;j<(*M);j++){
        X[i*(*M)+j]=X_in[i*(*M)+j];
      }
    }
    GetRNGstate(); 
    
    tau2i_mat=(double*)malloc(((*T) - 1)*(*M)*sizeof(double));
    for(unsigned int i=0;i<((*T) - 1)*(*M);i++){
      tau2i_mat[i]=tau2i[*M+i];
    }
    double lambda2_start = 0.0;
    double *lambda2_samps = NULL;
    lambda2_start = lambda2[0];
    lambda2_samps = &(lambda2[1]);
    // /* create a new Bayesian lasso regression */
    blasso = new Blasso(*M, *n, X, Y, beta,	lambda2_start, s2[0], tau2i, *r, *delta, *a, *b,(bool)*normalize, clptr);
    
    // /* Gibbs draws for the parameters */
    blasso->Rounds((*T) - 1, *thin, &(mu[1]), &(beta[*M]), M,	&(s2[1]), tau2i_mat, lambda2_samps, clptr);
    

    PutRNGstate(); 
    
    blasso->Cleanup();
    // /* clean up */
    clReleaseMemObject(blasso->a_mem);
    clReleaseMemObject(blasso->temp_mem);
    clReleaseMemObject(blasso->V_mem);
    clReleaseMemObject(blasso->l_mem);
    clReleaseMemObject(blasso->v1_mem);
    clReleaseMemObject(blasso->a1_mem);
    clReleaseMemObject(blasso->sizes_mem);
    clReleaseMemObject(blasso->MM_mem);
    clReleaseMemObject(blasso->a2_mem);
    clReleaseMemObject(blasso->Xp_mem);
    clReleaseMemObject(blasso->tau_mem);
    
    free(X);free(tau2i_mat);
    clEnqueueNDRangeKernel(clptr->cmdQueue, clptr->dummy, 1, NULL, single, NULL, 0, NULL,&(clptr->ev5));
    clWaitForEvents(1, &(clptr->ev5));
    
    clGetEventProfilingInfo(clptr->ev1, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL); 
    clGetEventProfilingInfo(clptr->ev5, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL); 
    seconds = (time_end - time_start)/ 1000000000.0;
    clReleaseEvent(clptr->ev1);clReleaseEvent(clptr->ev5);
    Rprintf("Time: %f, %f, %f\n",blasso->time1,blasso->time2,blasso->time3);
    timing[2]=blasso->inverseTime;
    timing[3]=blasso->drawBetaTime;
    timing[1]=seconds-timing[2]-timing[3];
    timing[0]=seconds;
    PROTECT(list = Rf_allocVector(VECSXP, 6)); 
    SET_VECTOR_ELT(list, 0, Rmu); 
    SET_VECTOR_ELT(list, 1, Rbeta);
    SET_VECTOR_ELT(list, 2, Rs2); 
    SET_VECTOR_ELT(list, 3, Rlambda2); 
    SET_VECTOR_ELT(list, 4, Rtau2i); 
    SET_VECTOR_ELT(list, 5, Rtiming); 
    Rf_setAttrib(list, R_NamesSymbol, list_names); 
    UNPROTECT(2);
    
    return(list);
    
  }
}
