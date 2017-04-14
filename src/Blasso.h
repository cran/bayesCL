#ifndef __BLASSO_H__
#define __BLASSO_H__


typedef struct bayesreg
{
  unsigned int M;           /* VEDNO ISTI KOT 'M' V RAZREDU 'Blasso'! dimension of these matrices and vectors */
  float *A;               /* inverse of Vb unscaled by s2 */
  float *Ai;              /* inverse of A */
  float *bmu;              /* posterior mean for beta */
  float *ABmu;             /* in R: ABmu = A %*% bmu */
  float *Vb;              /* posterior covariance matrix for beta Vb = s2*Ai */

} BayesReg;

class Blasso
{
  
private:
  unsigned int M;
  unsigned int N;
  unsigned int n;
  unsigned int nf;
  
  int normalize;
  
  double* Xorig;
  double* Xnorm;
  double* Xp;
  double Xnorm_scale;
  double* Xmean;
  
  double* Y;
  double Ymean;
  double*resid;
  double* XtY;
  double YtY;
  
  double lambda2;
  double s2;
  double *tau2i;
  double *beta;
  
  double *rn;               /* vector for N(0,1) draws used to sample beta */
  double *Xbeta_v;          /* untility vector for unnorming beta coefficients */
  double *BtDi;             /* in R syntax: t(B) %*% inv(D) %*% B */
  
  double r;
  bool icept;
  double delta;
  
  double a;
  double b;
  
  BayesReg *breg;
  
  int *pin;                 /* integer list of the columns of X that are in use */
protected:
  void InitX(const unsigned int n, double *X, const bool normalize);
  void InitParams(double *beta, const double lambda2, const double s2, double *tau2i);
  void InitY(const unsigned int N, double *Y);
  void UpdateXY(void);
  void Draw(const unsigned int thin, const bool fixnu,CLptrs *clptr);
  void GetParams(double *mu, double *beta, unsigned int *M, double *s2, double *tau2i, double *lambda2) const;
  
public:
  
  Blasso(const unsigned intM, const unsigned int n, double *X, double *Y,
         double *beta,
         const double lambda2, const double s2, double *tau2i, const double r, const double delta, const double a, const double b,
         const bool normalize,CLptrs *clptr);
  void Rounds(const unsigned int T, const unsigned int thin,
              double *mu, double *beta, unsigned int *M, double *s2,
              double *tau2i, double *lambda2,CLptrs *clptr);
  void Cleanup();
  void Init(void);
  
  double inverseTime,drawBetaTime,part1Time,part2Time;
  cl_mem a_mem,a2_mem,tau_mem,Xp_mem;
  cl_mem temp_mem;
  cl_mem V_mem;
  cl_mem l_mem;
  cl_int status;
  
  cl_mem v1_mem;
  cl_mem a1_mem;
  cl_mem sizes_mem;
  cl_mem MM_mem;
  
  size_t datasize;
  size_t datasize_temp;
  size_t datasize_leftUpdate;
  size_t datasize_Update;
  
  int block;
  int offset;
  int parts;
  
  int part_size_fixed;
  int part_size;
  int remainder;
  cl_event ev1,ev2,ev3,ev4,ev5,ev6,ev7,ev8;
  double time1,time2,time3,time4,time5,time6,time7,time8;
  
};

BayesReg* new_BayesReg(const unsigned int M, const unsigned int n, double **Xp, double *DiXp, CLptrs *clptr);
void delete_BayesReg(BayesReg* breg);
bool compute_BayesReg(const unsigned int M, double *XtY, double *tau2i,
                      const double lambda2, const double s2, BayesReg *breg);
void refresh_Vb(BayesReg *breg, const double s2);
void alloc_rest_BayesReg(BayesReg* breg);

void draw_tau2i_lasso(const unsigned int M, double *tau2i, double *beta,
                      double lambda2, double s2);
double rinvgauss(const double mu, const double lambda);

#endif
