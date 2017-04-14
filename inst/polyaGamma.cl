#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error "double precision doubleing point not supported by OpenCL implementation."
#endif
#define rng_max_val (4294967296.0f)
#define __PI  3.141592653589793238462643383279502884197f
#define HALFPISQ  (0.5f * __PI * __PI)
#define FOURPISQ  (4.0f * __PI * __PI)
#define __TRUNC  0.64f
#define __TRUNC_RECIP (1.0f / __TRUNC)

inline double unif_step(uint4 *seed){
    uint t;
    t = ((*seed).x^((*seed).x<<11));
    
    (*seed).x = (*seed).y;
    (*seed).y = (*seed).z;
    (*seed).z = (*seed).w;
    (*seed).w = ((*seed).w^((*seed).w>>19))^(t^(t>>8));

    return ((double)(*seed).w)/rng_max_val;
}

inline double phi(double x){
  // constants
  double a1 =  0.254829592f;
  double a2 = -0.284496736f;
  double a3 =  1.421413741f;
  double a4 = -1.453152027f;
  double a5 =  1.061405429f;
  double p  =  0.3275911f;
  
  // Save the sign of x
  int sign = 1;
  if (x < 0)
    sign = -1;
  x = fabs(x)/sqrt(2.0f);
  
  double t = 1.0f/(1.0f + p*x);
  double y = 1.0f - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);
  
  return 0.5f*(1.0f + sign*y);
}


inline double mass_texpon(double Z){
  double t = __TRUNC;

  double fz = 0.125f * __PI*__PI + 0.5f * Z*Z;
  double b = sqrt(1.0f / t) * (t * Z - 1);
  double a = sqrt(1.0f / t) * (t * Z + 1) * -1.0f;

  double x0 = log(fz) + fz * t;
  double xb = x0 - Z + log(phi(b));
  double xa = x0 + Z + log(phi(a));

  double qdivp = 4 / __PI * ( exp(xb) + exp(xa) );

  return 1.0f / (1.0f + qdivp);
}

inline double rtigauss(double Z, uint4 *seed){
   Z = fabs(Z);
  double t = __TRUNC;
  double X = t + 1.0f;
  double aa,bb;
  uint tt;

  if (__TRUNC_RECIP > Z) { // mu > t
    double alpha = 0.0f;
    tt = ((*seed).x^((*seed).x<<11));
    
    (*seed).x = (*seed).y;
    (*seed).y = (*seed).z;
    (*seed).z = (*seed).w;
    (*seed).w = ((*seed).w^((*seed).w>>19))^(tt^(tt>>8));

    aa=((double)(*seed).w)/rng_max_val;
    
    while (aa > alpha) {
      
      tt = ((*seed).x^((*seed).x<<11));
    
    (*seed).x = (*seed).y;
    (*seed).y = (*seed).z;
    (*seed).z = (*seed).w;
    (*seed).w = ((*seed).w^((*seed).w>>19))^(tt^(tt>>8));

    aa=((double)(*seed).w)/rng_max_val;

    tt = ((*seed).x^((*seed).x<<11));
    
    (*seed).x = (*seed).y;
    (*seed).y = (*seed).z;
    (*seed).z = (*seed).w;
    (*seed).w = ((*seed).w^((*seed).w>>19))^(tt^(tt>>8));

    bb=((double)(*seed).w)/rng_max_val;
    
      double E1 = -1.0f*log(aa);
      double E2 = -1.0f*log(bb);
      while ( E1*E1 > 2 * E2 / t) {
        tt = ((*seed).x^((*seed).x<<11));
    
    (*seed).x = (*seed).y;
    (*seed).y = (*seed).z;
    (*seed).z = (*seed).w;
    (*seed).w = ((*seed).w^((*seed).w>>19))^(tt^(tt>>8));

    aa=((double)(*seed).w)/rng_max_val;
    tt = ((*seed).x^((*seed).x<<11));
    
    (*seed).x = (*seed).y;
    (*seed).y = (*seed).z;
    (*seed).z = (*seed).w;
    (*seed).w = ((*seed).w^((*seed).w>>19))^(tt^(tt>>8));

    bb=((double)(*seed).w)/rng_max_val;
        E1 = -1.0f*log(aa); E2 = -1.0f*log(bb);
      }
      X = 1 + E1 * t;
      X = t / (X * X);
      alpha = exp(-0.5f * Z*Z * X);
      
      tt = ((*seed).x^((*seed).x<<11));
    
    (*seed).x = (*seed).y;
    (*seed).y = (*seed).z;
    (*seed).z = (*seed).w;
    (*seed).w = ((*seed).w^((*seed).w>>19))^(tt^(tt>>8));

    aa=((double)(*seed).w)/rng_max_val;

    }
  }
  else {
    
    double mu = 1.0f / Z;
    while (X > t) {
      uint t;
        double a,b;
        t = ((*seed).x^((*seed).x<<11));
          
        (*seed).x = (*seed).y;
        (*seed).y = (*seed).z;
        (*seed).z = (*seed).w;
        (*seed).w = ((*seed).w^((*seed).w>>19))^(t^(t>>8));

        a=((double)(*seed).w)/rng_max_val;

        t = ((*seed).x^((*seed).x<<11));
          
        (*seed).x = (*seed).y;
        (*seed).y = (*seed).z;
        (*seed).z = (*seed).w;
        (*seed).w = ((*seed).w^((*seed).w>>19))^(t^(t>>8));

        b=((double)(*seed).w)/rng_max_val;

         

      double Y = sqrt(-2*log(a)) * cos(2*__PI*b); Y *= Y;
      double half_mu = 0.5f * mu;
      double mu_Y    = mu  * Y;
      X = mu + half_mu * mu_Y - half_mu * sqrt(4 * mu_Y + mu_Y * mu_Y);
      tt = ((*seed).x^((*seed).x<<11));
    
    (*seed).x = (*seed).y;
    (*seed).y = (*seed).z;
    (*seed).z = (*seed).w;
    (*seed).w = ((*seed).w^((*seed).w>>19))^(tt^(tt>>8));

    aa=((double)(*seed).w)/rng_max_val;
      if (aa > mu / (mu + X))
      X = mu*mu / X;
    }
  }
  return X;
}

inline double a(int n, double x){
  double K = (n + 0.5f) * __PI;
  double y = 0;
  if (x > __TRUNC) {
    y = K * exp( -0.5f * K*K * x );
  }else if (x > 0) {
    double expnt = -1.5f * (log(0.5f * __PI)  + log(x)) + log(K) - 2.0f * (n+0.5f)*(n+0.5f) / x;
    y = exp(expnt);    
  }
  return y;
}

__kernel void polyagammaf(__global uint4 *seed_table, __global double *C,__global double *Z_all, __global int* nbuf,int batch, int n)                        
{         
  int idx = get_global_id(0);

  uint4 seed;
  if(idx<n)
  seed=seed_table[idx]; 
  double Z;                                      
  int ii;
  int index;
  for(ii=0;ii<batch;ii++){
    if((idx*batch+ii)<n){
    index=nbuf[idx*batch+ii];
      Z=Z_all[index];
      
      // Change the parameter.
      Z = fabs(Z) * 0.5f;
      
      double fz = 0.125f * __PI*__PI + 0.5f * Z*Z;

      double X = 0.0f;
      double S = 1.0f;
      double Y = 0.0f;
      bool go = true;
      int end=1;
      int n = 0;
      while (end) {
        
        
        if ( unif_step(&seed) < mass_texpon(Z) ){
          X = __TRUNC -log(unif_step(&seed)) / fz;
        }
        else{
          X = rtigauss(Z, &seed);          
        }
        
        
        S = a(0, X);       
        
        Y = unif_step(&seed) * S;
        n = 0;
        
        go = true;
        // Cap the number of iterations?
        while (go) {
          ++n;
          if (n%2==1) {
            S = S - a(n, X);
            if ( Y<=S ){
              C[idx*batch+ii]= 0.25f * X;
              //seed_table[idx]=seed; 
              end=0;
              go=false;
            } 
          } else {
            S = S + a(n, X);
            if ( Y>S ) go = false;
          }

        }
      }
    }
  }

}      

__kernel void polyagamma_standalone(__global uint4 *seed_table, __global double *C,__global double *Z_all, int batch, int n)                        
{
  
  int idx = get_global_id(0);

  uint4 seed;
  if(idx<n)
  seed=seed_table[idx]; 
  double Z;                                      
  int ii;
  int index;
  for(ii=0;ii<batch;ii++){
    if((idx*batch+ii)<n){    
      Z=Z_all[idx*batch+ii];
      
      // Change the parameter.
      Z = fabs(Z) * 0.5f;
      
      double fz = 0.125f * __PI*__PI + 0.5f * Z*Z;

      double X = 0.0f;
      double S = 1.0f;
      double Y = 0.0f;
      bool go = true;
      int end=1;
      int n = 0;
      while (end) {
        
        
        if ( unif_step(&seed) < mass_texpon(Z) ){
          X = __TRUNC -log(unif_step(&seed)) / fz;
        }
        else{
          X = rtigauss(Z, &seed);          
        }
        
        
        S = a(0, X);       
        
        Y = unif_step(&seed) * S;
        n = 0;
        
        go = true;
        // Cap the number of iterations?
        while (go) {
          ++n;
          if (n%2==1) {
            S = S - a(n, X);
            if ( Y<=S ){
              C[idx*batch+ii]= 0.25f * X;
              //seed_table[idx]=seed; 
              end=0;
              go=false;
            } 
          } else {
            S = S + a(n, X);
            if ( Y>S ) go = false;
          }

        }
      }
    }
  }

}   

__kernel void gibbs_step1(__global double *A,__global double *B, __global int *nbuf, int N, int n){
  int idx = get_global_id(0);
  int i, start, end;
  start=nbuf[idx];
  if(idx==(N-1)){
    end=n;
  }else{
    end=nbuf[idx+1];
  }
  double sum=0.0;
  for(i=start;i<end;i++){
    sum+=A[i];
  }
  B[idx]=sum;
  
  
}                                            
__kernel void gibbs_step2(__global double *tX,__global double *tXomega,__global double *w, int N){
  int idx = get_global_id(0);
  int idy;
  idy=idx/(N);
  idx=idx%(N);
  tXomega[idy*N+idx]=tX[idy*N+idx]*w[idx];    
}   
#define TS 16
#define PART 8
__kernel void gibbs_step3(__global double *tX,__global double *tXomega,__global double *tXOmX, __global double *P0, __global double *L, __global double *V, int j, int N, int P){

/// Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int part= get_group_id(1)%PART;
    const int groupy= get_group_id(1)/PART;
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = (TS*groupy + col); // Col ID of C (0..N)    
    // Local memory to fit a tile of TS*TS elements of A and B
    __local double Asub[TS][TS];
    __local double Bsub[TS][TS];
   
    // Initialise the accumulation register
    double acc = 0.0;

    // Loop over all tiles
    const int numTiles = (N+TS-1)/TS;
    const int partTiles=(numTiles+PART-1)/PART;
    //const int partTiles=numTiles;

    for (int t=part*partTiles; t<(part+1)*partTiles; t++) {
  //for (int t=0; t<numTiles; t++) {
      
      // Load one tile of A and B into local memory
      const int tiledRow = TS*t + row;
      const int tiledCol = TS*t + col;
    if(globalRow<P && tiledCol<N){
    Asub[col][row] = tXomega[globalRow*N+tiledCol ];
    }else{
    Asub[col][row] = 0.0;
    }
    
    if(globalCol<P && tiledRow<N){
    Bsub[row][col] = tX[globalCol*N + tiledRow];
    }else{
    
    Bsub[row][col] = 0.0;
    }
            // Synchronise to make sure the tile is loaded
      barrier(CLK_LOCAL_MEM_FENCE);
   
      // Perform the computation for a single tile
    if(globalCol<P && globalRow<P){
      for (int k=0; k<TS; k++) {
        acc+=Asub[k][row]*Bsub[k][col];

      }
    }
      // Synchronise before loading the next tile
      barrier(CLK_LOCAL_MEM_FENCE);
      
    }
  if(globalRow<P && globalCol<P){ 
    // Store the final result in C
    if(part==0)
    tXOmX[globalCol*P + globalRow] = acc+P0[j*P*P+globalCol*P+globalRow];
    else
      tXOmX[part*P*P+globalCol*P + globalRow] = acc;
  }
  
} 
__kernel void gibbs_step3a(__global double *tXOmX,int P){
  const int globalRow = get_global_id(0); // Row ID of C (0..M)
  const int globalCol = get_global_id(1); // Col /D of C (0..N)
  int i;
  double acc=0.0;
  for(i=0;i<PART;i++){
    acc+=tXOmX[i*P*P+globalCol*P + globalRow];    
  }
  
  tXOmX[globalCol*P + globalRow]=acc;
}


#define PARTs4 128
__kernel void gibbs_step4(__global double *tXOmC,__global double *tXomega,__global double *c_j,__global double *Z, __global double *b0,int j,int J,int N, int P){
  int idx = get_global_id(0);
  int part= get_global_id(1);

  int partN=(N+PARTs4-1)/PARTs4;
  int i=0;
  double sum=0.0;
  for(i=part*partN;i<(part+1)*partN&&i<N;i++){

    sum+=tXomega[idx*N+i]*c_j[i];       
  }
  if(part==0){
    if(idx<P)
      tXOmC[part*P+idx]=sum+Z[idx*(J-1)+j]+b0[idx*(J-1)+j];
  }
  else{
    if(idx<P)
      tXOmC[part*P+idx]=sum;
  }
}

__kernel void gibbs_step4a(__global double *tXOmC, int P){
  
  int idx = get_global_id(0);
  int i;
  double acc=0.0;
  for(i=0;i<PARTs4;i++){
    acc+=tXOmC[i*P+idx];
  }
  tXOmC[idx]=acc;  
} 

__kernel void gibbs_chol1(__global double *L,__global double *R, int P){
  double faktor;
  for (int i = 0; i < P; i++)
    for (int j = 0; j < (i+1); j++) {
      double s = 0;
      for (int k = 0; k < j; k++)
        s += L[i*P+k] * L[j*P+k];
      
    L[i*P+j] = (i == j) ?  sqrt(R[i*P+i] - s) :  (1.0 / L[j*P+j] * (R[i*P+j] - s));
  }                                        
}

__kernel void gibbs_forwardSubs(__global double *L,__global double *V1, int P){
  double faktor=0;
  int n=P;
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
} 

__kernel void gibbs_mmul(__global double *V,__global double *V1, int P){
  int k;
  int idx = get_global_id(0);
  int i,j;
  i=idx/P;
  j=idx%P;
  double sum=0.0;
  for (k = 0; k < P; k++) {
    sum=sum+V1[k*P+i]*V1[k*P+j];
  }  
  V[i*P+j]=sum;
}

__kernel void gibbs_chol_setup(__global double *L, __global double *V,__global double *V1,__global double *lower, int P){
  int k;
  int i = get_global_id(0);

  for (k = 0; k < P; k++) {
    L[i*P+k]=0.0;
    if(i==k)
      V[i*P+k]=1.0;
    else
      V[i*P+k]=0.0;
  }  
} 

__kernel void gibbs_vmmul(__global double *a,__global double *b,__global double *c, int P )                        
{         
  int idx = get_global_id(0);
  double s=0.0;
  int j;
  for(j=0;j<P;j++){
    s=s+a[idx*P+j]*b[j];
  }
  c[idx]=s;
}

__kernel void gibbs_copy(__global double *A,__global double *B,int N, int J )                        
{         
  int idx = get_global_id(0);
  int idy = idx/(J-1);
  idx=idx%(J-1);
  B[idy*(J-1)+idx]=A[idy*J+idx+1];
}

__kernel void gibbs_vmmul1(__global double *a,__global double *b,__global double *c,__global double *d,int j, int P, int N, int J)                        
{         
  int idx = get_global_id(0);
  double s;
  s=0.0;
  int i;
  
  for(i=0;i<P;i++){
    s=s+a[i*N+idx]*b[i];
  }
  c[idx*J+j]=s;  
  if(j<(J-2)){
  d[idx*(J-1)+j]=s;  
  }

}

__kernel void gibbs_addMean(__global double *lower,__global double *d,__global double *mean, int P )                        
{   
   double acc=0;
  for (uint i=0; i < P; i++){
    acc=0.0;
    for (uint k=0; k < i; k++){
      acc+=lower[i*P+k]*d[i];
    }
    d[i]=acc+mean[i];
  }
}

__kernel void gibbs_minmax(__global double *XB_no_j, __global double *minmax, int N, int J )                        
{
  int idx = get_global_id(0);
  int j;
  if(idx<N){
    minmax[idx]=0;
    double tempMin=XB_no_j[idx*(J-1)+0];
    double tempMax=tempMin;
    double temp;
    for(j=1;j<J-1;j++){
      temp=XB_no_j[idx*(J-1)+j];
      if(temp>tempMax){
        tempMax=temp;
      }
      /*if(temp<tempMin){
        tempMin=temp;
      }     */
    }
    /*if(tempMin<0)
      temp=-1.0*tempMin;
    else
      temp=tempMin;*/
    /*if(temp>tempMax){
      minmax[idx]=tempMin;
    }else{*/
      minmax[idx]=tempMax;
    //}   
  }
}

__kernel void gibbs_rowSum(__global double *XB_no_j,__global double *minmax, __global double *A, int N, int J )                        
{         
  int idx = get_global_id(0);
  int j;
  if(idx<N){
      double AA=0.0;
      double offset=minmax[idx];
      for(j=0;j<J-1;j++){
        AA=AA+exp(XB_no_j[idx*(J-1)+j]-offset);
      }
      A[idx]=AA;      
  }
}
__kernel void gibbs_c_eta(__global double *XB,__global double *A, __global double *eta_j, __global double *c_j,__global double *minmax, int N, int J, int j )                        
{         
  int idx = get_global_id(0);
  double temp;
  if(idx<N){
      temp=log(A[idx])+minmax[idx];
      c_j[idx]=temp;
      eta_j[idx]=XB[idx*J+j]-temp;
  }
}

__kernel void dummy_kernel( )                        
{         
  int idx = get_global_id(0);
}

__kernel void leftUpdate(__global float* l,__global float* ap, __global float* temp,int offset,int block,int M){
	int k;
	int i=get_global_id(0)+offset+block;
	int j=get_global_id(1);	
	float sum=0;
	for (k=0;k<block;k++){
		sum+=ap[i*M+offset+k]*temp[j*block+k];
	}	 
	l[(i-offset-block)*block+j]=sum;
	ap[i*M+offset+j]=sum;
}

__kernel void midUpdate(__global float* l,__global float* ap,int offset,int block,int M){
	int k;
	int i=get_global_id(0);
	int j=get_global_id(1);
	float sum;
	sum=0;
	for (k=0;k<block;k++){
		sum+=l[i*block+k]*l[j*block+k];
	}
	ap[(i+offset+block)*M+offset+block+j]-=sum;	
}

#define TS1 16
__kernel void invprod(__global double *tX,__global double *tXOmX, int P){

    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = TS1*get_group_id(0) + row;
    const int globalCol = TS1*get_group_id(1) + col;

    __local double Asub[TS1][TS1];
    __local double Bsub[TS1][TS1];
   
    double acc = 0.0;

    const int numTiles = (P+TS1-1)/TS1;
	
    for (int t=0; t<numTiles; t++) {
      
		const int tiledRow = TS1*t + row;
		const int tiledCol = TS1*t + col;
		
		if(globalRow<P && tiledCol<P){
			Asub[col][row] = tX[globalRow*P+tiledCol ];
		}else{
			Asub[col][row] = 0.0;
		}
		
		if(globalCol<P && tiledRow<P){
			Bsub[row][col] = tX[globalCol*P + tiledRow];
		}else{		
			Bsub[row][col] = 0.0;
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
	   
		if(globalCol<P && globalRow<P){
		  for (int k=0; k<TS1; k++) {
			acc+=Asub[k][row]*Bsub[k][col];
		  }
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
      
    }
	
	if(globalRow<P && globalCol<P){ 
		tXOmX[globalCol*P + globalRow] = acc;
	}
  
}