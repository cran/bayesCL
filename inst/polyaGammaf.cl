#define rng_max_val (4294967296.0f)
#define __PI  3.141592653589793238462643383279502884197f
#define HALFPISQ  (0.5f * __PI * __PI)
#define FOURPISQ  (4.0f * __PI * __PI)
#define __TRUNC  0.64f
#define __TRUNC_RECIP (1.0f / __TRUNC)

inline float unif_step(uint4 *seed){
    uint t;
    t = ((*seed).x^((*seed).x<<11));
    
    (*seed).x = (*seed).y;
    (*seed).y = (*seed).z;
    (*seed).z = (*seed).w;
    (*seed).w = ((*seed).w^((*seed).w>>19))^(t^(t>>8));

    return ((float)(*seed).w)/rng_max_val;
}

inline float phi(float x){
  // constants
  float a1 =  0.254829592f;
  float a2 = -0.284496736f;
  float a3 =  1.421413741f;
  float a4 = -1.453152027f;
  float a5 =  1.061405429f;
  float p  =  0.3275911f;
  
  // Save the sign of x
  int sign = 1;
  if (x < 0)
    sign = -1;
  x = fabs(x)/sqrt(2.0f);
  
  float t = 1.0f/(1.0f + p*x);
  float y = 1.0f - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);
  
  return 0.5f*(1.0f + sign*y);
}


inline float mass_texpon(float Z){
  float t = __TRUNC;

  float fz = 0.125f * __PI*__PI + 0.5f * Z*Z;
  float b = sqrt(1.0f / t) * (t * Z - 1);
  float a = sqrt(1.0f / t) * (t * Z + 1) * -1.0f;

  float x0 = log(fz) + fz * t;
  float xb = x0 - Z + log(phi(b));
  float xa = x0 + Z + log(phi(a));

  float qdivp = 4 / __PI * ( exp(xb) + exp(xa) );

  return 1.0f / (1.0f + qdivp);
}

inline float rtigauss(float Z, uint4 *seed){
   Z = fabs(Z);
  float t = __TRUNC;
  float X = t + 1.0f;
  float aa,bb;
  uint tt;

  if (__TRUNC_RECIP > Z) { // mu > t
    float alpha = 0.0f;
    tt = ((*seed).x^((*seed).x<<11));
    
    (*seed).x = (*seed).y;
    (*seed).y = (*seed).z;
    (*seed).z = (*seed).w;
    (*seed).w = ((*seed).w^((*seed).w>>19))^(tt^(tt>>8));

    aa=((float)(*seed).w)/rng_max_val;
    
    while (aa > alpha) {
      
      tt = ((*seed).x^((*seed).x<<11));
    
    (*seed).x = (*seed).y;
    (*seed).y = (*seed).z;
    (*seed).z = (*seed).w;
    (*seed).w = ((*seed).w^((*seed).w>>19))^(tt^(tt>>8));

    aa=((float)(*seed).w)/rng_max_val;

    tt = ((*seed).x^((*seed).x<<11));
    
    (*seed).x = (*seed).y;
    (*seed).y = (*seed).z;
    (*seed).z = (*seed).w;
    (*seed).w = ((*seed).w^((*seed).w>>19))^(tt^(tt>>8));

    bb=((float)(*seed).w)/rng_max_val;
    
      float E1 = -1.0f*log(aa);
      float E2 = -1.0f*log(bb);
      while ( E1*E1 > 2 * E2 / t) {
        tt = ((*seed).x^((*seed).x<<11));
    
    (*seed).x = (*seed).y;
    (*seed).y = (*seed).z;
    (*seed).z = (*seed).w;
    (*seed).w = ((*seed).w^((*seed).w>>19))^(tt^(tt>>8));

    aa=((float)(*seed).w)/rng_max_val;
    tt = ((*seed).x^((*seed).x<<11));
    
    (*seed).x = (*seed).y;
    (*seed).y = (*seed).z;
    (*seed).z = (*seed).w;
    (*seed).w = ((*seed).w^((*seed).w>>19))^(tt^(tt>>8));

    bb=((float)(*seed).w)/rng_max_val;
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

    aa=((float)(*seed).w)/rng_max_val;

    }
  }
  else {
    
    float mu = 1.0f / Z;
    while (X > t) {
      uint t;
        float a,b;
        t = ((*seed).x^((*seed).x<<11));
          
        (*seed).x = (*seed).y;
        (*seed).y = (*seed).z;
        (*seed).z = (*seed).w;
        (*seed).w = ((*seed).w^((*seed).w>>19))^(t^(t>>8));

        a=((float)(*seed).w)/rng_max_val;

        t = ((*seed).x^((*seed).x<<11));
          
        (*seed).x = (*seed).y;
        (*seed).y = (*seed).z;
        (*seed).z = (*seed).w;
        (*seed).w = ((*seed).w^((*seed).w>>19))^(t^(t>>8));

        b=((float)(*seed).w)/rng_max_val;

         

      float Y = sqrt(-2*log(a)) * cos(2*__PI*b); Y *= Y;
      float half_mu = 0.5f * mu;
      float mu_Y    = mu  * Y;
      X = mu + half_mu * mu_Y - half_mu * sqrt(4 * mu_Y + mu_Y * mu_Y);
      tt = ((*seed).x^((*seed).x<<11));
    
    (*seed).x = (*seed).y;
    (*seed).y = (*seed).z;
    (*seed).z = (*seed).w;
    (*seed).w = ((*seed).w^((*seed).w>>19))^(tt^(tt>>8));

    aa=((float)(*seed).w)/rng_max_val;
      if (aa > mu / (mu + X))
      X = mu*mu / X;
    }
  }
  return X;
}

inline float a(int n, float x){
  float K = (n + 0.5f) * __PI;
  float y = 0;
  if (x > __TRUNC) {
    y = K * exp( -0.5f * K*K * x );
  }else if (x > 0) {
    float expnt = -1.5f * (log(0.5f * __PI)  + log(x)) + log(K) - 2.0f * (n+0.5f)*(n+0.5f) / x;
    y = exp(expnt);    
  }
  return y;
}

__kernel void polyagammaf(__global uint4 *seed_table, __global float *C,__global float *Z_all, __global int* nbuf,int batch, int n)                        
{         
  int idx = get_global_id(0);

  uint4 seed;
  if(idx<n)
  seed=seed_table[idx]; 
  float Z;                                      
  int ii;
  int index;
  for(ii=0;ii<batch;ii++){
    if((idx*batch+ii)<n){
    index=nbuf[idx*batch+ii];
      Z=Z_all[index];
      
      // Change the parameter.
      Z = fabs(Z) * 0.5f;
      
      float fz = 0.125f * __PI*__PI + 0.5f * Z*Z;

      float X = 0.0f;
      float S = 1.0f;
      float Y = 0.0f;
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

__kernel void polyagamma_standalone(__global uint4 *seed_table, __global float *C,__global float *Z_all, int batch, int n)                        
{
  
  int idx = get_global_id(0);

  uint4 seed;
  if(idx<n)
  seed=seed_table[idx]; 
  float Z;                                      
  int ii;
  int index;
  for(ii=0;ii<batch;ii++){
    if((idx*batch+ii)<n){    
      Z=Z_all[idx*batch+ii];
      
      // Change the parameter.
      Z = fabs(Z) * 0.5f;
      
      float fz = 0.125f * __PI*__PI + 0.5f * Z*Z;

      float X = 0.0f;
      float S = 1.0f;
      float Y = 0.0f;
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

__kernel void gibbs_step1(__global float *A,__global float *B, __global int *nbuf, int N, int n){
  int idx = get_global_id(0);
  int i, start, end;
  start=nbuf[idx];
  if(idx==(N-1)){
    end=n;
  }else{
    end=nbuf[idx+1];
  }
  float sum=0.0;
  for(i=start;i<end;i++){
    sum+=A[i];
  }
  B[idx]=sum;
  
  
}                                            
__kernel void gibbs_step2(__global float *tX,__global float *tXomega,__global float *w, int N){
  int idx = get_global_id(0);
  int idy;
  idy=idx/(N);
  idx=idx%(N);
  tXomega[idy*N+idx]=tX[idy*N+idx]*w[idx];    
}   
#define TS 16
__kernel void gibbs_step3(__global float *tX,__global float *tXomega,__global float *tXOmX, __global float *P0, __global float *L, __global float *V, int j, int N, int P){

/// Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int part= get_group_id(1)%PART;
    const int groupy= get_group_id(1)/PART;
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = (TS*groupy + col); // Col ID of C (0..N)    
    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
   
    // Initialise the accumulation register
    float acc = 0.0;

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
__kernel void gibbs_step3a(__global float *tXOmX,int P){
  const int globalRow = get_global_id(0); // Row ID of C (0..M)
  const int globalCol = get_global_id(1); // Col /D of C (0..N)
  int i;
  float acc=0.0;
  for(i=0;i<PART;i++){
    acc+=tXOmX[i*P*P+globalCol*P + globalRow];    
  }
  
  tXOmX[globalCol*P + globalRow]=acc;
}


__kernel void gibbs_step4(__global float *tXOmC,__global float *tXomega,__global float *c_j,__global float *Z, __global float *b0,int j,int J,int N, int P){
  int idx = get_global_id(0);
  int part= get_global_id(1);

  int partN=(N+PARTs4-1)/PARTs4;
  int i=0;
  float sum=0.0;
  for(i=part*partN;i<(part+1)*partN&&i<N;i++){
    if(i<N&&idx<P)
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

__kernel void gibbs_step4a(__global float *tXOmC, int P){
  
  int idx = get_global_id(0);
  int i;
  float acc=0.0;
  for(i=0;i<PARTs4;i++){
    acc+=tXOmC[i*P+idx];
  }
  tXOmC[idx]=acc;  
} 

__kernel void gibbs_copy(__global float *A,__global float *B,int N, int J )                        
{         
  int idx = get_global_id(0);
  int idy = idx/(J-1);
  idx=idx%(J-1);
  B[idy*(J-1)+idx]=A[idy*J+idx+1];
}

__kernel void gibbs_vmmul1(__global float *a,__global float *b,__global float *c,__global float *d,int j, int P, int N, int J)                        
{         
  int idx = get_global_id(0);
  float s;
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

__kernel void gibbs_addMean(__global float *lower,__global float *d,__global float *mean, int P )                        
{   
   float acc=0;
  for (uint i=0; i < P; i++){
    acc=0.0;
    for (uint k=0; k < i; k++){
      acc+=lower[i*P+k]*d[i];
    }
    d[i]=acc+mean[i];
  }
}

__kernel void gibbs_minmax(__global float *XB_no_j, __global float *minmax, int N, int J )                        
{
  int idx = get_global_id(0);
  int j;
  if(idx<N){
    minmax[idx]=0;
    float tempMin=XB_no_j[idx*(J-1)+0];
    float tempMax=tempMin;
    float temp;
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

__kernel void gibbs_rowSum(__global float *XB_no_j,__global float *minmax, __global float *A, int N, int J )                        
{         
  int idx = get_global_id(0);
  int j;
  if(idx<N){
      float AA=0.0;
      float offset=minmax[idx];
      for(j=0;j<J-1;j++){
        AA=AA+exp(XB_no_j[idx*(J-1)+j]-offset);
      }
      A[idx]=AA;      
  }
}
__kernel void gibbs_c_eta(__global float *XB,__global float *A, __global float *eta_j, __global float *c_j,__global float *minmax, int N, int J, int j )                        
{         
  int idx = get_global_id(0);
  float temp;
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

__kernel void cholesky(__global float *A,__global float *b,int offset,int M, int n,__global float *V)
{	

	int f=get_local_id(0);
	__local float d[64][64];
	int arrSize=n;
	for (int i = 0; i < n; i++){
		d[i][f]=b[(i+offset)*M+f+offset];
		if(i==f)
			V[i*n+f]=1.0;
		else
			V[i*n+f]=0.0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);	
	
	float sig;
	for (int i=0;i<n-1;i++){
		sig=sqrt(d[i][i]);
		if(f==0){
			d[i][i]=sig;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if(f>=(i+1))
			d[f][i]=d[f][i]/sig;		
		barrier(CLK_LOCAL_MEM_FENCE);
		for(int k=i+1;k<n&&k<arrSize;k++){
		  if(f>=(i+1))
		  	d[k][f]=d[k][f]-d[k][i]*d[f][i];		  
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}	
	barrier(CLK_LOCAL_MEM_FENCE);
	if(f==0){
		d[n-1][n-1]=sqrt(d[n-1][n-1]);	
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	for(int p=f+1;p<n;p++){
		d[f][p]=0;
	}
	
	
	barrier(CLK_LOCAL_MEM_FENCE);	
	//write to matrix
	for (int i = 0; i < n; i++){
		b[(i+offset)*M+f+offset]=d[i][f];
	}
	barrier(CLK_LOCAL_MEM_FENCE);	
	//if(f==0){	
	float faktor;
	for (int i = 0; i < n; i++){
		if(i>0){
			barrier(CLK_LOCAL_MEM_FENCE);
			for (int j = i; j < n; j++) {
				faktor=d[j][i-1];
				V[j*n+f]=V[j*n+f]-faktor*V[(i-1)*n+f];
				barrier(CLK_LOCAL_MEM_FENCE);
			}
		}
		faktor=d[i][i];
		V[i*n+f]=V[i*n+f]/faktor;
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

__kernel void zeroTop(__global float* a, int M){
	int i=get_global_id(0);
	int j=get_global_id(1);
	if(j>i){
		a[i*M+j]=0;
	}
	
}

__kernel void initChol(__global float* a, __global float* b,  __global float* c, int M){
	int i=get_global_id(0);
	int j=get_global_id(1);
	if(i==j){
		a[i*M+j]=b[i]+c[i*M+j];
	}else{
		a[i*M+j]=c[i*M+j];
	}
}
#define TS3	16
__kernel void leftUpdate(__global float* l,__global float* ap, __global float* temp,int offset,int block,int M, int max_threads){
	
	
	int k;
	const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int i = TS3*get_group_id(0) + row;
    const int j = TS3*get_group_id(1) + col;
	
	__local float Asub[TS3][TS3];
    __local float Bsub[TS3][TS3];
	
	float sum=0;
	const int numTiles = (block+TS3-1)/TS3;
	
	for (int t=0; t<numTiles; t++) {
	
		const int tiledRow = TS3*t + row;
		const int tiledCol = TS3*t + col;
	
		if(i<max_threads && tiledCol<block){
			Asub[col][row] = ap[(i+offset+block)*M+offset+tiledCol];
		}else{
			Asub[col][row] = 0.0;
		}
		
		if(j<block && tiledRow<block){
			Bsub[row][col] = temp[j*block+tiledRow];
		}else{		
			Bsub[row][col] = 0.0;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		for (k=0;k<TS3;k++){
			sum+=Asub[k][row]*Bsub[k][col];			
		}
		barrier(CLK_LOCAL_MEM_FENCE);		
	}
	if(i<max_threads && j<block){
		l[i*block+j]=sum;
	}		
}

__kernel void midUpdate(__global float* l,__global float* ap,int offset,int block,int M, int max_threads){
	
	int k;
	const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int i = TS3*get_group_id(0) + row;
    const int j = TS3*get_group_id(1) + col;

    __local float Asub[TS3+1][TS3+1];
    __local float Bsub[TS3][TS3];
	
	float sum;
	sum=0;
	//if(i< && j<max_threads){
		
	const int numTiles = (block+TS3-1)/TS3;
	for (int t=0; t<numTiles; t++) {
		
		const int tiledRow = TS3*t + row;
		const int tiledCol = TS3*t + col;
		
		if(i<max_threads && tiledCol<max_threads){
			Asub[col][row] = l[i*block+tiledCol];
		}else{
			Asub[col][row] = 0.0;
		}
		
		if(j<max_threads && tiledRow<max_threads){
			Bsub[row][col] = l[j*block+tiledRow ];
		}else{		
			Bsub[row][col] = 0.0;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if(i<max_threads && j<max_threads && i<M && j<M){
			for (k=0;k<TS3;k++){
				sum+=Asub[k][row]*Bsub[k][col];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if(i<max_threads && j<max_threads && i<M && j<M){
		ap[(i+offset+block)*M+offset+block+j]-=sum;	
	}	
	if(i<max_threads && j<block){
		ap[(i+offset+block)*M+offset+j]=l[i*block+j];
	}
}

#define WPT 4
#define RTS	8
__kernel void invprod(__global float *tX,__global float *tXOmX, int P){


    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = TS1*get_group_id(0) + row;
    const int globalCol = TS1*get_group_id(1) + col;
	
    __local float Asub[TS1][TS1];
    __local float Bsub[TS1][TS1];
   
    //float acc = 0.0;
	int k_min;
	float acc[WPT1];
    for (int w=0; w<WPT1; w++) {
        acc[w] = 0.0f;
    }
    const int numTiles = (P+TS1-1)/TS1;
	for (int t=0; t<numTiles; t++) {
		for (int w=0; w<WPT1; w++) {
			const int tiledRow = TS1*t + row;
			const int tiledCol = TS1*t + col;
			
			if(globalRow<P && (tiledCol+w*RTS1)<P){
					Asub[col+w*RTS1][row] = tX[(tiledCol+w*RTS1)*P+globalRow];
			}else{
				Asub[col+w*RTS1][row] = 0.0;
			}
			
			if((globalCol+w*RTS1)<P && tiledRow<P){
					Bsub[row][col+w*RTS1] = tX[tiledRow*P+globalCol+w*RTS1 ];
			}else{		
				Bsub[row][col+w*RTS1] = 0.0;
			}
		}
		
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
		for(int k=0;k<TS1;k++){				
			for (int w=0; w<WPT1; w++) {
				acc[w]+=Asub[k][row]*Bsub[k][col+w*RTS1];
			}
		} 
		
		barrier(CLK_LOCAL_MEM_FENCE);
		  
	}	
	for (int w=0; w<WPT1; w++) {
		if(globalRow<P&&(globalCol+w*RTS1)<P){		
			tXOmX[(globalCol+w*RTS1)*P + globalRow]=acc[w];
		}
	}
 
}
#define TS4	16
__kernel void invprod2(__global float *tX,__global float *tXOmX, int P, int n){


    const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int globalRow = TS4*get_group_id(0) + row;
    const int globalCol = TS4*get_group_id(1) + col;

    __local float Asub[TS4][TS4];
    __local float Bsub[TS4][TS4];
   
    float acc = 0.0;
	int k_min;
    const int numTiles = (n+TS4-1)/TS4;
	for (int t=0; t<numTiles; t++) {
		  
		const int tiledRow = TS4*t + row;
		const int tiledCol = TS4*t + col;
		
		if(globalRow<P && tiledCol<n){
			Asub[col][row] = tX[tiledCol*P+globalRow];
		}else{
			Asub[col][row] = 0.0;
		}
		
		if(globalCol<P && tiledRow<n){
			Bsub[row][col] = tX[tiledRow*P+globalCol ];
		}else{		
			Bsub[row][col] = 0.0;
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
		if(globalCol<P && globalRow<P){
		  for (int k=0; k<TS4; k++) {
			acc+=Asub[k][row]*Bsub[k][col];
		  }
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
		  
	}
	if(globalRow<P && globalCol<P){ 
		tXOmX[globalCol*P + globalRow] = acc;
	}
	
  
}

__kernel void inverse1(__global float* ap,int remainder,int part_size_fixed, int M){
	
	int indeks=get_global_id(0);
	int i=indeks*part_size_fixed;
	int part_size;
	float faktor;
	if(indeks<remainder){
		i+=indeks;
		part_size=part_size_fixed+1;
	}else{
		i+=remainder;
		part_size=part_size_fixed;
	}	
	
	__local float V1[64][64];
	
    for(int p=0;p<part_size;p++){
      for(int r=0;r<part_size;r++){
        if(p==r)
          V1[p][r]=1;
        else
          V1[p][r]=0;
      }
    }
	
    for (unsigned int ii = 0; ii < part_size; ii++){
      if(ii>0){
        for (unsigned int j = ii; j < part_size; j++) {
          faktor=ap[(j+i)*M+i+ii-1];
          for (unsigned int k = 0; k < part_size; k++) {
            V1[j][k]=V1[j][k]-faktor*V1[(ii-1)][k];
          }
        }
	  }
      faktor=ap[(ii+i)*M+ii+i];
      for (unsigned int k = 0; k < part_size; k++) {
        V1[ii][k]=V1[ii][k]/faktor;  
      }
    }
	
    for(int p=0;p<part_size;p++){
      for(int r=0;r<part_size;r++){
        ap[(p+i)*M+i+r]=V1[p][r];
      }
    }
}

#define TS2 32
__kernel void inverse2(__global float* ap,__global int* sizes,__global float* MM, int repeat, int remainder,int part_size_fixed, int M){

	int n=get_global_id(2)*2;
	float sum=0;
	int part_size1=0,part_size2=0;
	int offset_i,offset_j;
	
	for(int r=n*repeat;r<(n+1)*repeat;r++)
		part_size1+= sizes[r];

	for(int r=(n+1)*repeat;r<(n+2)*repeat;r++)
		part_size2+= sizes[r];
	int sizeM=repeat*(part_size_fixed+1);
	offset_i=(n+1)*repeat*part_size_fixed;offset_j=n*repeat*part_size_fixed;
	if(((n+1)*repeat)<=remainder)
		offset_i+=(n+1)*repeat;
	else
		offset_i+=remainder;
		
	if((n*repeat)<=remainder)
		offset_j+=n*repeat;
	else
		offset_j+=remainder;
	
	const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int i = TS2*get_group_id(0) + row;
    const int j = TS2*get_group_id(1) + col;

    __local float Asub[TS2][TS2];
    __local float Bsub[TS2][TS2];
      
    float acc[WPT];
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
	
    const int numTiles = (part_size2+TS2-1)/TS2;
	
	sum=0;
	for (int t=0; t<numTiles; t++) {
		for (int w=0; w<WPT; w++) {
			const int tiledRow = TS2*t + row;
			const int tiledCol = TS2*t + col;
			
			if(i<part_size2 && (tiledCol+w*RTS1)<part_size1){
				Asub[col+w*RTS][row] = ap[(i+offset_i)*M+tiledCol+offset_j+part_size1+w*RTS];
			}else{
				Asub[col+w*RTS][row] = 0.0;
			}
				
			if((j+w*RTS1)<part_size1 && tiledRow<part_size2){
				Bsub[col+w*RTS][row] = ap[(tiledRow+offset_i)*M+j+offset_j+w*RTS];
			}else{		
				Bsub[col+w*RTS][row] = 0.0;
			}
		}
			
		barrier(CLK_LOCAL_MEM_FENCE);
	
		for(int k=0;k<TS2;k++){				
			for (int w=0; w<WPT; w++) {
				acc[w]+=Asub[k][row]*Bsub[col+w*RTS][k];
			}
		} 
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	for (int w=0; w<WPT; w++) {
		if(i<part_size2&&(j+w*RTS)<part_size1){
			MM[(n/2)*(sizeM)*(sizeM)+i*part_size1+j+w*RTS]=acc[w];
		}
	}
}

__kernel void inverse3(__global float* ap,__global int* sizes,__global float* MM, int repeat, int remainder,int part_size_fixed, int M){

	int n=get_global_id(2)*2;
	float sum=0;
	int part_size1=0,part_size2=0;
	int offset_i,offset_j;
	//int i=get_global_id(0);
	//int j=get_global_id(1);
	for(int r=n*repeat;r<(n+1)*repeat;r++)
		part_size1+= sizes[r];

	for(int r=(n+1)*repeat;r<(n+2)*repeat;r++)
		part_size2+= sizes[r];
	
	int sizeM=repeat*(part_size_fixed+1);
	offset_i=(n+1)*repeat*part_size_fixed;offset_j=n*repeat*part_size_fixed;
	if(((n+1)*repeat)<=remainder)
		offset_i+=(n+1)*repeat;
	else
		offset_i+=remainder;
		
	if((n*repeat)<=remainder)
		offset_j+=n*repeat;
	else
		offset_j+=remainder;

	
	const int row = get_local_id(0);
    const int col = get_local_id(1);
    const int i = TS2*get_group_id(0) + row;
    const int j = TS2*get_group_id(1) + col;

    __local float Asub[TS2][TS2];
    __local float Bsub[TS2][TS2];
	
    float acc[WPT];
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
	
    const int numTiles = (part_size1+TS2-1)/TS2;
	
	sum=0;
	for (int t=0; t<numTiles; t++) {
		for (int w=0; w<WPT; w++) {
			const int tiledRow = TS2*t + row;
			const int tiledCol = TS2*t + col;
			if(i<part_size2 && (tiledCol+w*RTS)<part_size1){
				Asub[col+w*RTS][row] = MM[(n/2)*(sizeM)*(sizeM)+i*part_size1+tiledCol+w*RTS];
			}else{
				Asub[col+w*RTS][row] = 0.0;
			}
			if((j+w*RTS)<part_size1 && tiledRow<part_size2){
				Bsub[col+w*RTS][row] = ap[(tiledRow+offset_i-part_size1)*M+j+offset_j+w*RTS];
			}else{		
				Bsub[col+w*RTS][row] = 0.0;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		
		for(int k=0;k<TS2;k++){
			//if(i<part_size2&&j<part_size1){
			for (int w=0; w<WPT; w++) {
				acc[w]+=Asub[k][row]*Bsub[col+w*RTS][k];
			}
			//}
		} 	
		barrier(CLK_LOCAL_MEM_FENCE);		
	}
	for (int w=0; w<WPT; w++) {
		if(i<part_size2&&(j+w*RTS)<part_size1){
			ap[(i+offset_i)*M+j+offset_j+w*RTS]=-acc[w];
		}
	}
	
}

__kernel void dummy(){
	int i=get_global_id(0);
	
}

__kernel void addS2(__global float* a,__global float* b, float s2, int M){
	int i=get_global_id(0);
	int j=get_global_id(1);
	b[i*M+j]=a[i*M+j]*s2;
}	