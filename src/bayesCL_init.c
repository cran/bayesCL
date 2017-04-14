#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* FIXME: 
   Check these declarations against the C/Fortran source code.
*/

/* .Call calls */
extern SEXP blassoGPU(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP find_devices(SEXP);
extern SEXP find_platforms();
extern SEXP mult_gibbs(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP mult_gibbs_double(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP prepareCL(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP rpg_devroye(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP rpg_devroye_double(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"blassoGPU",          (DL_FUNC) &blassoGPU,          18},
    {"find_devices",       (DL_FUNC) &find_devices,        1},
    {"find_platforms",     (DL_FUNC) &find_platforms,      0},
    {"mult_gibbs",         (DL_FUNC) &mult_gibbs,         13},
    {"mult_gibbs_double",  (DL_FUNC) &mult_gibbs_double,  13},
    {"prepareCL",            (DL_FUNC) &prepareCL,             5},
    {"rpg_devroye",        (DL_FUNC) &rpg_devroye,         8},
    {"rpg_devroye_double", (DL_FUNC) &rpg_devroye_double,  8},
    {NULL, NULL, 0}
};

void R_init_bayesCL(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
