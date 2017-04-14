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

#ifndef __MULTLOGIT__
#define __MULTLOGIT__

#include <list>
#include "PolyaGamma.h"
#include <stdexcept>
#include <time.h>
#include <algorithm>
#include <stdio.h>
#include <math.h>


// I need this to interrupt the Gibbs sampler from R.
#ifdef USE_R
#include <R_ext/Utils.h>
#endif

using std::list;

////////////////////////////////////////////////////////////////////////////////
// LOGIT //
////////////////////////////////////////////////////////////////////////////////

class MultLogit{
  
  // Variables.
  int P; // Dimension of beta_j.
  int N; // Number of observations.
  int J; // Total number of categories
  
  // Random variates.
  PolyaGamma pg;
  
public:
  CLptrs *CLpointer;
  
  MultLogit();

  void gibbs(float* tX,float* ty, int* n, float* betas, float* m0, float *P0, int samp, int burn, int pp, int nn, int jj, float* timing);
  
}; 


MultLogit::MultLogit()
{
 
} 

// Gibbs sampling -- Default MultLogit.
void MultLogit::gibbs(float* tX,float* ty,int* n, float* betas, float* m0, float* P0, 
                      int samp, int burn, int PP, int NN, int JJ, float* timing)
{
  float ttime;
  int goAway=0;
  float tstep1=0,tPG=0,tstep2=0,tbeta=0,tcopy=0;

  P=PP;N=NN;J=JJ;
  int M = (int)samp;
  
  float* tkappa;
  tkappa=(float*)malloc((J-1)*N*sizeof(float));
  for(int i = 0; i < N; ++i)
    for(int j = 0; j < J-1; ++j)
      tkappa[j*N+i] = n[i] * (ty[i*(J-1)+j] - 0.5);
  
  float *XBB,*ZZ,*b00;
  b00=(float*)malloc(sizeof(float)*P*(J-1));
  XBB=(float*)malloc(N*J*sizeof(float));
  ZZ=(float*)malloc(sizeof(float)*P*(J-1));
  
  float acc=0;
  int ff=0;
  for (int i=0; i < P; i++){
    for (int j=0; j < J-1; j++){
      acc=0.0;
      for (int k=0; k < N; k++){
        acc+=tX[i*N+k]*tkappa[j*N+k];
      }
      ZZ[i*(J-1)+j]=acc;
    }
  }
  
  free(tkappa);
  
  ff=0;
  for(int i=0;i<(int)M;i++){
    for(int j=0;j<(int)P;j++){
      for(int k=0;k<(int)(J-1);k++){
        betas[i*P*(J-1)+j*(J-1)+k]=0.0;
        ff++;
      } 
    } 
  }

  acc=0;
  for (int j=0; j < J-1; j++){
    for (int i=0; i < P; i++){
      acc=0.0;
      for (int k=0; k < P; k++){
        acc+=P0[j*P*P+i*P+k]*m0[j*P+k];
      }
      b00[i*(J-1)+j]=acc;
    }
  }
  
  for (int i=0; i < N; i++){
    for (int j=0; j < J-1; j++){
      XBB[i*J+j]=0.0;
      for(int k=0;k<P;k++){
        XBB[i*J+j]+=tX[k*N+i]*betas[k*(J-1)+j];
      }
    }
  }
  for (int i=0; i < N; i++){
    XBB[i*J+J-1]=0.0;
  }

  float *P11;
  P11=(float*)malloc(sizeof(float)*P*P);
  float *b11;
  b11=(float*)malloc(sizeof(float)*P);

  pg.rng_init(0,0);
  
  char abc[100];
  pg.test=0;
  pg.initData(N,J,P,n,tX,ZZ,P0,b00,XBB,CLpointer);
  free(ZZ);free(b00);free(XBB);
  
  Rprintf("Start burn..\n");
  pg.initialized=0;
  int kk=0;
  int nan_error=0;
  pg.insertEvent(&(CLpointer->startEvent),CLpointer);
  
  for(int m = 0; (m < burn+1)&&!goAway; ++m){
    pg.copyCL(N,J,CLpointer);
    clReleaseEvent(CLpointer->ev9);
    for (int j = 0; (j < J-1)&&!goAway; j++) {
      

      pg.step1CL(N,J,j,CLpointer);
      
      pg.drawCL(CLpointer->PGlocal,CLpointer);
  
      pg.step2CL(j,N,P,CLpointer);
   
      clEnqueueReadBuffer( CLpointer->cmdQueue, pg.tXOmX_buffer, CL_TRUE, 0, P*P*sizeof(float), P11, 0, NULL, NULL);

      clEnqueueReadBuffer( CLpointer->cmdQueue, pg.tXOmC_buffer, CL_TRUE, 0, P*sizeof(float), b11, 0, NULL, NULL);
      
      for(int jj=0;jj<P;jj++)
      {
        betas[j*P+jj]=rnorm(0,1.0);
      }
      pg.cholesky(b11,P11,&betas[j*P],P);

      
      //if(j==0)printf("beta: %d - %f\n",m, beta[0].col(j)(0));
      
      kk=0;
      
      sprintf(abc,"%f",betas[0]);
      if(strcmp(abc,"nan")==0 || strcmp(abc,"-nan")==0){
        kk++;
      }
      
      if(kk==0){
        clEnqueueWriteBuffer( CLpointer->cmdQueue, pg.beta_buffer, CL_TRUE, 0, sizeof(float)*P, &betas[j*P], 0, NULL, NULL);
        pg.betaCL(P,N,j,J,0,CLpointer);  
        nan_error=0;
      }else{
        Rprintf("Repeat due to error\n");
        j--;
        nan_error++;
        if(nan_error>10){
          //exit if multiple consecutive errors
          goAway++;
        }
      }
      clReleaseEvent(CLpointer->ev2);clReleaseEvent(CLpointer->ev4);
      clReleaseEvent(CLpointer->ev6); 
   }
#ifdef USE_R
    if (m%1==0) R_CheckUserInterrupt();
#endif
  }
  
  pg.insertEvent(&(CLpointer->stopEvent),CLpointer);
  ttime=pg.calculateTime(&(CLpointer->startEvent),&(CLpointer->stopEvent));
  Rprintf("Burn-in execution time %g seconds\n",ttime);
  Rprintf("Expect approx. %g sec. for %i samples.\n", ttime * samp / burn, samp);
  clReleaseEvent(CLpointer->stopEvent);
  clReleaseEvent(CLpointer->startEvent);

  float last_time=0.0;
  float curr_time=0.0;
  pg.insertEvent(&(CLpointer->startEvent),CLpointer);
  
  for(int m = 1; (m < samp)&&!goAway; ++m){
    
    pg.copyCL(N,J,CLpointer);
    curr_time=pg.calculateTime(&(CLpointer->startEvent),&(CLpointer->ev9));
    tcopy+=curr_time-last_time;
    last_time=curr_time;
    
    clReleaseEvent(CLpointer->ev9);
    
    for (int j = 0; (j < J-1)&&!goAway; j++) {
      
      
      pg.step1CL(N,J,j,CLpointer);
      curr_time=pg.calculateTime(&(CLpointer->startEvent),&(CLpointer->ev2));
      tstep1+=curr_time-last_time;
      last_time=curr_time;
      
      
      pg.drawCL(CLpointer->PGlocal,CLpointer);
      curr_time=pg.calculateTime(&(CLpointer->startEvent),&(CLpointer->ev4));
      tPG+=curr_time-last_time;
      last_time=curr_time;
      
      pg.step2CL(j,N,P,CLpointer);
      curr_time=pg.calculateTime(&(CLpointer->startEvent),&(CLpointer->ev6));
      tstep2+=curr_time-last_time;
      last_time=curr_time;
      
      
      clEnqueueReadBuffer( CLpointer->cmdQueue, pg.tXOmX_buffer, CL_TRUE, 0, P*P*sizeof(float), P11, 0, NULL, NULL);
      clEnqueueReadBuffer( CLpointer->cmdQueue, pg.tXOmC_buffer, CL_TRUE, 0, P*sizeof(float), b11, 0, NULL, NULL);

      
      for(int jj=0;jj<P;jj++)
      {
        betas[m*P*(J-1)+j*P+jj]=rnorm(0,1.0);
      }
      pg.cholesky(b11,P11,&betas[m*P*(J-1)+j*P],P);

      //if(j==0)printf("beta: %d - %f\n",m, beta[m].col(j)(0));
      kk=0;
      
      sprintf(abc,"%f",betas[m*P*(J-1)]);
      if(strcmp(abc,"nan")==0 || strcmp(abc,"-nan")==0){
        kk++;
      }
      
      if(kk==0){
        clEnqueueWriteBuffer( CLpointer->cmdQueue, pg.beta_buffer, CL_TRUE, 0, sizeof(float)*P, &betas[m*P*(J-1)+j*P], 0, NULL, &(CLpointer->ev8));
        pg.betaCL(P,N,j,J,m,CLpointer);  
        nan_error=0;
      }else{
        Rprintf("Repeat due to error\n");
        j--;
        nan_error++;
        if(nan_error>10){
          //exit if multiple consecutive errors
          goAway++;
        }
      }
      
      
      curr_time=pg.calculateTime(&(CLpointer->startEvent),&(CLpointer->ev8));
      tbeta+=curr_time-last_time;
      last_time=curr_time;
      clReleaseEvent(CLpointer->ev2);clReleaseEvent(CLpointer->ev4);
      clReleaseEvent(CLpointer->ev6);clReleaseEvent(CLpointer->ev8); 
    }
  }
  pg.insertEvent(&(CLpointer->stopEvent),CLpointer);
  ttime=pg.calculateTime(&(CLpointer->startEvent),&(CLpointer->stopEvent));
  Rprintf("\nExecution time for %d samples = %0.5f s\n", samp,  ttime );
  free(P11);free(b11);
  clReleaseEvent(CLpointer->stopEvent);
  clReleaseEvent(CLpointer->startEvent); 
  clFlush(CLpointer->cmdQueue);
  clFinish(CLpointer->cmdQueue);
  pg.freeData(CLpointer);
  
  if(goAway>0)
    timing[0]=-5.0;
  else
    timing[0]=ttime;
  timing[1]=tcopy;
  timing[2]=tstep1;
  timing[3]=tPG;
  timing[4]=tstep2;
  timing[5]=tbeta;
  
}
#endif

////////////////////////////////////////////////////////////////////////////////
// APPENDIX //
////////////////////////////////////////////////////////////////////////////////


