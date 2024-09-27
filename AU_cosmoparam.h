//    Copyright 2024 Amol Upadhye
//
//    This file is part of FlowsForTheMassesII.
//
//    FlowsForTheMassesII is free software: you can redistribute it and/or 
//    modify it under the terms of the GNU General Public License as published 
//    by the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    FlowsForTheMassesII is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with FlowsForTheMassesII.  If not, see 
//    <http://www.gnu.org/licenses/> .

////////////////////////////////// CONSTANTS ///////////////////////////////////
//All dimensionful quantities are in units of Mpc/h to the appropriate power,
//unless otherwise noted.  Values declared as const double or const int  may be
//modified by the user, while those in #define statements are derived parameters
//which should not be changed.

//conformal hubble today
const double Hc0h = 3.33564095198152e-04; //(1e2/299792.458)
#define Hc0h2 (Hc0h*Hc0h)

//initial scale factor, and max value of eta=ln(a/a_in)
const double aeta_in = 1e-3; 
#define eta_stop (-log(aeta_in))

//neutrino fluid parameters:
//N_TAU = number of HDM flows
//N_MU = number of multipoles to track for each flow
#define N_TAU (GALAQ_N)
#define N_MU (10)

const int SWITCH_HDM_SOURCE_NONLIN = 1; //source hdm growth using nonlin CB

//Time-RG constants
#define N_UI (14)
#define N_PI (17)
#define N_BISPEC (10)
const int aU[] = {0,0,0,0,0,0,0,0, 1,1,1,1,1,1};
const int cU[] = {0,0,0,0,0,0,0,0, 1,1,1,1,1,1};
const int dU[] = {1,1,1,1,1,1,1,1, 1,1,1,1,1,1};
const int bU[] = {0,0,0,0,1,1,1,1, 0,0,0,1,1,1};
const int eU[] = {0,0,1,1,0,0,1,1, 0,0,1,0,0,1};
const int fU[] = {0,1,0,1,0,1,0,1, 0,1,1,0,1,1};
const int JU[] = {8,9,10,11,12,13,14,15,56,57,59,60,61,63};

//total number of equations:
//  N_PI*N_TAU*N_MU*NK (delta theta r for N_TAU streams N_MU moments NK wave#)
//  + 2*NK  (delta and theta for linear CDM+Baryons)
//  + 17*NK (delta, theta, cross, I for non-linear CDM+Baryons)
//  + 10*NK (4 equilateral + 6 isosceles bispectrum components)
//  + NK    (CMB lensing potential power spectrum)
#define N_EQ (N_PI*N_TAU*N_MU*NK + (2+N_PI+N_BISPEC)*NK + NK)

//cosmoparam constants
#define COSMOPARAM_NU_EFF (3.044)
#define COSMOPARAM_NU_MASSIVE (3.044)
#define COSMOPARAM_MAX_REDSHIFTS (1000)
#define COSMOPARAM_MAX_CHAR_LEN (1000)
#define COSMOPARAM_MAX_NQ_NTAU (20000)

const int COSMOPARAM_DEBUG_INIT = 1;

////////////////////////////////////////////////////////////////////////////////
struct cosmoparam{

  //user-defined parameters
  double n_s;
  double sigma_8;
  double h;
  double Omega_m_0;
  double Omega_b_0;
  double Omega_hdm_tot_0;
  double T_CMB_0_K;
  double w0_eos_de;
  double wa_eos_de;
  double T_hdm_0_K;
  double m_hdm_eV;

  //code switches
  int switch_nonlinear;
  int switch_Nmunl;
  int switch_print;
  int switch_transfer_type;

  //inputs and outputs
  double z_nonlinear_initial;
  int num_z_outputs;
  double z_outputs[COSMOPARAM_MAX_REDSHIFTS];
  char file_transfer_function[COSMOPARAM_MAX_CHAR_LEN];
  int num_massive_hdm_approx;
  char file_hdm_distribution[COSMOPARAM_MAX_CHAR_LEN];
  char file_hdm_transfer_root[COSMOPARAM_MAX_CHAR_LEN];
  int num_interp_redshifts;
  double z_interp_redshifts[COSMOPARAM_MAX_REDSHIFTS];

  //fixed or derived parameters
  double Omega_cb_0;
  double Omega_hdm_t_0[N_TAU];
  double Omega_gam_0;
  double Omega_nurel_0;
  double Omega_nugam_0;
  double Omega_rel_0;
  double Omega_de_0;
  double Omega_m_h2_0;
  double Omega_b_h2_0;
  
  int N_tau;
  double N_nu_eff;
  double N_nu_massive;
  double f_cb_0;
  double f_hdm_0;
  
  double w_eos_cdm;
  double w_eos_gam;

  double alpha_G;
  double sound_horiz;
  double Theta_CMB_27_Sq;

  //mode-coupling data for CB
  int initAggcb=0; //set to 1 if Aggcb allocated and 2 if computed
  double *Aggcb; //mode-couplings; dimension nUI*NK
  double *yAgg;  //perturbations at last Agg computation

  //mode-coupling data for neutrinos
  int nAgghdm=0; //number of fluids for which Agghdm computed
  double *Agghdm; //mode-couplings; dimension nAgghdm*nUI*N_MU*NK
  double *Agghdm0; //monopole mode-coupling data; dimension nAgghdm*nUI*NK
};

////////////////////////////////////////////////////////////////////////////////
//functions for using cosmoparam

int nu_frac(double Onh2, double mnu[3]){
  const double Dm22_eV2 = 7.42e-5, Dm23_eV2 = 2.517e-3; //2007.14792
  double mu_nu_eV = 93.25, M_eV = Onh2 * mu_nu_eV, m1_eV, m2_eV, m3_eV;
  
  //special case of M under minimum: ignore measured mass splittings
  //Note that CLASS complains for Omega_nu,i,0*h^2 less than about 1e-5,
  //corresponding to individual mass ~ 1e-5*mu_nu_eV = 9.325e-4, so don't
  //allow any nu mass to drop below this.
  if(M_eV < 3e-5 * mu_nu_eV) M_eV = 3e-5 * mu_nu_eV; 
  if(M_eV <= 0.06){
    m1_eV = mu_nu_eV*1e-5;
    m2_eV = (M_eV-m1_eV) * sqrt(Dm22_eV2) / (sqrt(Dm22_eV2) + sqrt(Dm23_eV2));
    if(m2_eV < m1_eV) m2_eV = m1_eV;
    m3_eV = M_eV - m1_eV - m2_eV;
    
    mnu[0] = m1_eV;
    mnu[1] = m2_eV;
    mnu[2] = m3_eV;
    return 0;
  }
  
  //Iterate to find the masses in the normal hierarchy
  m1_eV = M_eV / 3;
  double m1_eV_old = M_eV;
  int iter = 0;
  while( fabs(m1_eV/m1_eV_old-1.0)>1e-9 && ++iter <= 100000){
    m1_eV_old = m1_eV;
    m1_eV = M_eV / (1.0 + sqrt(1.0 + Dm22_eV2/(m1_eV_old*m1_eV_old))
                    + sqrt(1.0 + Dm23_eV2/(m1_eV_old*m1_eV_old)));
  }
  m2_eV = sqrt(m1_eV*m1_eV + Dm22_eV2);
  m3_eV = sqrt(m1_eV*m1_eV + Dm23_eV2);
  
  mnu[0] = m1_eV;
  mnu[1] = m2_eV;
  mnu[2] = m3_eV;
  return iter>=100000;
}

int print_cosmoparam(const struct cosmoparam C, int verbosity){
  
  if(verbosity > 0){
    printf("#cosmoparam: n_s=%g, sigma_8=%g, h=%g, Omega_m_0=%g, Omega_b_0=%g, Omega_hdm_tot_0=%g, T_CMB_0_K=%g, w0_eos_de=%g, wa_eos_de=%g\n",
      C.n_s, C.sigma_8, C.h, C.Omega_m_0, C.Omega_b_0, C.Omega_hdm_tot_0,
      C.T_CMB_0_K, C.w0_eos_de, C.wa_eos_de);
    printf("#cosmoparam:HDM: t   Omega_{hdm,t,0}\n");
    for(int t=0; t<C.N_tau; t++)
      printf("#cosmoparam:HDM: %i %1.18g\n", t, C.Omega_hdm_t_0[t]);
    fflush(stdout);
  }

  if(verbosity > 1){
    printf("#cosmoparam: switch_nonlinear=%i, switch_Nmunl=%i, switch_print=%i, switch_transfer_type=%i\n",
      C.switch_nonlinear, C.switch_Nmunl, C.switch_print, 
      C.switch_transfer_type);
    fflush(stdout);
  }

  if(verbosity > 2){
    printf("#cosmoparam: z_nonlinear_initial=%g\n",C.z_nonlinear_initial);
    printf("#cosmoparam: z_outputs[%i]:", C.num_z_outputs);
    for(int i=0; i<C.num_z_outputs; i++) printf(" %g",C.z_outputs[i]);
    printf("\n");
    fflush(stdout);
  }

  return 0;
}

int alloc_cosmoparam_A(struct cosmoparam *C){
  if(!C->initAggcb){
    C->initAggcb = 1; //allocated
    C->nAgghdm = 0;//C->N_tau; //leave 0 until we actually compute Agghdm
    
    C->Aggcb = (double *)malloc(N_UI*NK*sizeof(double));
    for(int i=0; i<N_UI*NK; i++) C->Aggcb[i] = 0;

    C->yAgg = (double *)malloc(N_EQ*sizeof(double));
    for(int i=0; i<N_EQ; i++) C->yAgg[i] = 0;

    C->Agghdm = (double *)malloc(N_TAU*N_UI*N_MU*NK*sizeof(double));
    for(int i=0; i<N_TAU*N_UI*N_MU*NK; i++) C->Agghdm[i] = 0;

    C->Agghdm0 = (double *)malloc(N_TAU*N_UI*NK*sizeof(double));
    for(int i=0; i<N_TAU*N_UI*NK; i++) C->Agghdm0[i] = 0;
  }
  return 0;
}

int alloc_cosmoparam_Acb(struct cosmoparam *C){
  if(!C->initAggcb){
    C->initAggcb = 1;
    C->Aggcb = (double *)malloc(N_UI*NK*sizeof(double));
    C->yAgg = (double *)malloc(N_EQ*sizeof(double));
  }
  return 0;
}

int free_cosmoparam_A(struct cosmoparam *C){
  if(C->initAggcb){
    C->initAggcb = 0;
    free(C->Aggcb);
    free(C->yAgg);
  }
  if(C->nAgghdm > 0){
    C->nAgghdm = 0;
    free(C->Agghdm);
    free(C->Agghdm0);
  }
  return 0;
}

int initialize_Omega_hdm_t_0(struct cosmoparam *C, const char *dist, int N_tau){

  double sum_wef = 0;
  tabulated_function f_hdm(dist, COSMOPARAM_MAX_NQ_NTAU, 2, 0, 1);
 
  if(GALAQ_q[0]<=f_hdm.min1x() || GALAQ_q[N_tau-1]>=f_hdm.max1x()){
    printf("\nERROR: requested q range [%g,%g] outside of input distribution function range [%g,%g].  Please provide a distribution function file covering a larger range", GALAQ_q[0], GALAQ_q[N_tau-1], f_hdm.min1x(), f_hdm.max1x());
    if(GALAQ_q[N_tau-1]>=f_hdm.max1x()) 
      printf(" or reduce N_TAU in AU_cosmoparam.h");
    printf(".  Quitting.\n\n");
    fflush(stdout);
    abort();
  }

  for(int t=0; t<N_tau; t++){
    double pre = sq(GALAQ_q[t]) * exp(GALAQ_q[t]);
    C->Omega_hdm_t_0[t] = pre * GALAQ_w[t] * f_hdm(GALAQ_q[t]);
    sum_wef += C->Omega_hdm_t_0[t];
  }

  double Ohdmfac = C->Omega_hdm_tot_0 / sum_wef;
  for(int t=0; t<N_tau; t++) C->Omega_hdm_t_0[t] *= Ohdmfac;

  return 0;
}

int initialize_cosmoparam(struct cosmoparam *C, const char *params, int N_tau){
  FILE *fp;
  if( (fp=fopen(params,"r")) == NULL ){
    printf("ERROR: File %s not found.  Quitting.\n",params);
    exit(1);
  }

  if(COSMOPARAM_DEBUG_INIT)
    printf("#initialize_cosmoparam: reading param file %s\n", params);

  char buf[1000], buf2[1000], *pbuf = buf;
  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%lg",&C->n_s);

  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%lg",&C->sigma_8);
  
  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%lg",&C->h);
  
  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%lg",&C->Omega_m_0);
  
  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%lg",&C->Omega_b_0);
  
  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%lg",&C->Omega_hdm_tot_0);
  
  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%lg",&C->T_CMB_0_K);
  
  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%lg",&C->w0_eos_de);
  
  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%lg",&C->wa_eos_de);

  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%i",&C->switch_nonlinear);

  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%i",&C->switch_Nmunl);

  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%i",&C->switch_print);

  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%i",&C->switch_transfer_type);

  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%lg",&C->z_nonlinear_initial);

  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%i",&C->num_z_outputs);

  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  for(int i=0; i<C->num_z_outputs; i++){
    sscanf(pbuf,"%s",buf2);
    sscanf(buf2,"%lg",&C->z_outputs[i]);
    pbuf += strlen(buf2)+1;
  }

  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%s",C->file_transfer_function);

  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%i",&C->num_massive_hdm_approx);
  if(C->num_massive_hdm_approx != 1){
    printf("ERROR: num_massive_hdm_approx != 1.  Only mflr supported.\n");
    exit(1);
  }

  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%lg",&C->T_hdm_0_K);
  
  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%lg",&C->m_hdm_eV);

  do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
  sscanf(buf,"%s",C->file_hdm_distribution);

  //Shouldn't need to use this here.
  if(C->num_massive_hdm_approx == 0){
    do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
    sscanf(buf,"%s",C->file_hdm_transfer_root);

    do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
    sscanf(buf,"%i",&C->num_interp_redshifts);
  
    do{ fgets(buf, sizeof buf, fp); } while(*buf=='#' || *buf=='\n');
    for(int i=0; i<C->num_interp_redshifts; i++){
      sscanf(pbuf,"%s",buf2);
      sscanf(buf2,"%lg",&C->z_interp_redshifts[i]);
      pbuf += strlen(buf2)+1;
    }
  }
  
  //set fixed/derived parameters
  C->N_tau = N_tau;
  C->N_nu_eff = COSMOPARAM_NU_EFF;
  C->N_nu_massive = COSMOPARAM_NU_MASSIVE;
  //nu_frac(C->Omega_nu_tot_0*C->h*C->h, C->m_nu_eV);

  initialize_Omega_hdm_t_0(C, C->file_hdm_distribution, N_tau);

  C->Omega_cb_0 = C->Omega_m_0 - C->Omega_hdm_tot_0;
  C->Omega_gam_0 = 4.46911743913795e-07 * pow(C->T_CMB_0_K,4) / (C->h*C->h);
  C->Omega_nurel_0 = 0.227107317660239
    * (C->N_nu_eff-C->N_nu_massive) * C->Omega_gam_0;
  C->Omega_nugam_0 = (1.0+0.227107317660239*C->N_nu_eff)*C->Omega_gam_0;
  C->Omega_rel_0 = C->Omega_gam_0 + C->Omega_nurel_0;
  C->Omega_de_0 = 1.0 - C->Omega_cb_0 - C->Omega_hdm_tot_0 - C->Omega_rel_0;
  C->Omega_m_h2_0 = C->Omega_m_0 * C->h	* C->h;
  C->Omega_b_h2_0 = C->Omega_b_0 * C->h * C->h;
  
  C->w_eos_cdm = 0;
  C->w_eos_gam = 0.333333333333333333;
  C->f_cb_0 = C->Omega_cb_0 / C->Omega_m_0;
  C->f_hdm_0 = C->Omega_hdm_tot_0 / C->Omega_m_0;

  C->sound_horiz = 55.234*C->h /  
    ( pow(C->Omega_cb_0 * C->h * C->h,0.2538)
      * pow(C->Omega_b_h2_0,0.1278)
      * pow(1.0 + C->Omega_hdm_tot_0 * C->h * C->h, 0.3794) );
  double rbm = C->Omega_b_0 / C->Omega_m_0;
  C->alpha_G = 1.0 - 0.328*log(431.0*C->Omega_m_h2_0) * rbm
    + 0.38 * log(22.3*C->Omega_m_h2_0) * rbm*rbm;
  C->Theta_CMB_27_Sq = pow(C->T_CMB_0_K/2.7,2);

  C->initAggcb = 0;
  C->nAgghdm = 0;
  
  if(COSMOPARAM_DEBUG_INIT) print_cosmoparam(*C,COSMOPARAM_DEBUG_INIT); 

  fclose(fp);
  return 0;
}
  
int copy_cosmoparam(const struct cosmoparam B, struct cosmoparam *C){

  C->n_s = B.n_s;
  C->sigma_8 = B.sigma_8;
  C->h = B.h;
  C->Omega_m_0 = B.Omega_m_0;
  C->Omega_b_0 = B.Omega_b_0;
  C->Omega_hdm_tot_0 = B.Omega_hdm_tot_0;
  C->T_CMB_0_K = B.T_CMB_0_K;
  C->w0_eos_de = B.w0_eos_de;
  C->wa_eos_de = B.wa_eos_de;
  C->T_hdm_0_K = B.T_hdm_0_K;
  C->m_hdm_eV  = B.m_hdm_eV;
  
  C->switch_nonlinear = B.switch_nonlinear;
  C->switch_Nmunl = B.switch_Nmunl;
  C->switch_print = B.switch_print;
  C->switch_transfer_type = B.switch_transfer_type;

  C->z_nonlinear_initial = B.z_nonlinear_initial;
  C->num_z_outputs = B.num_z_outputs;
  for(int i=0; i<B.num_z_outputs; i++) C->z_outputs[i] = B.z_outputs[i];
  strcpy(C->file_transfer_function,B.file_transfer_function);
  C->num_massive_hdm_approx = B.num_massive_hdm_approx;
  strcpy(C->file_hdm_distribution, B.file_hdm_distribution);
  strcpy(C->file_hdm_transfer_root, B.file_hdm_transfer_root);
  C->num_interp_redshifts = B.num_interp_redshifts;
  for(int i=0; i<B.num_interp_redshifts; i++)
    C->z_interp_redshifts[i] = B.z_interp_redshifts[i];
  
  //set fixed/derived parameters
  C->N_tau = B.N_tau;
  C->N_nu_eff = B.N_nu_eff;
  C->N_nu_massive = B.N_nu_massive;
  C->Omega_cb_0 = B.Omega_cb_0;
  C->Omega_gam_0 = B.Omega_gam_0;
  C->Omega_nurel_0 = B.Omega_nurel_0;
  C->Omega_nugam_0 = B.Omega_nugam_0;
  C->Omega_rel_0 = B.Omega_rel_0;
  C->Omega_de_0 = B.Omega_de_0;
  C->Omega_m_h2_0 = B.Omega_m_h2_0;
  C->Omega_b_h2_0 = B.Omega_b_h2_0;
  for(int t=0; t<B.N_tau; t++) C->Omega_hdm_t_0[t] = B.Omega_hdm_t_0[t];

  C->w_eos_cdm = 0;
  C->w_eos_gam = 0.333333333333333333;
  C->f_cb_0 = B.f_cb_0;
  C->f_hdm_0 = B.f_hdm_0;

  C->sound_horiz = B.sound_horiz;
  C->alpha_G = B.alpha_G;
  C->Theta_CMB_27_Sq = B.Theta_CMB_27_Sq;

  C->initAggcb = B.initAggcb;
  C->nAgghdm = B.nAgghdm;
  if(B.initAggcb){
    alloc_cosmoparam_A(C);

    if(B.initAggcb>1)
      for(int i=0; i<N_UI*NK; i++) C->Aggcb[i] = B.Aggcb[i];

    if(B.nAgghdm > 0){
      for(int i=0; i<C->nAgghdm*N_UI*N_MU*NK; i++) C->Agghdm[i]  = B.Agghdm[i];
      for(int i=0; i<C->nAgghdm*N_UI*NK; i++)      C->Agghdm0[i] = B.Agghdm0[i];
    }  
  }
  
  return 0;
}

int copy_cosmoparam_linear(const struct cosmoparam B, struct cosmoparam *C){

  C->n_s = B.n_s;
  C->sigma_8 = B.sigma_8;
  C->h = B.h;
  C->Omega_m_0 = B.Omega_m_0;
  C->Omega_b_0 = B.Omega_b_0;
  C->Omega_hdm_tot_0 = B.Omega_hdm_tot_0;
  C->T_CMB_0_K = B.T_CMB_0_K;
  C->w0_eos_de = B.w0_eos_de;
  C->wa_eos_de = B.wa_eos_de;
  C->T_hdm_0_K = B.T_hdm_0_K;
  C->m_hdm_eV  = B.m_hdm_eV;
  
  C->switch_nonlinear = B.switch_nonlinear;
  C->switch_Nmunl = B.switch_Nmunl;
  C->switch_print = B.switch_print;
  C->switch_transfer_type = B.switch_transfer_type;

  C->z_nonlinear_initial = B.z_nonlinear_initial;
  C->num_z_outputs = B.num_z_outputs;
  for(int i=0; i<B.num_z_outputs; i++) C->z_outputs[i] = B.z_outputs[i];
  strcpy(C->file_transfer_function,B.file_transfer_function);
  C->num_massive_hdm_approx = B.num_massive_hdm_approx;
  strcpy(C->file_hdm_distribution, B.file_hdm_distribution);
  strcpy(C->file_hdm_transfer_root, B.file_hdm_transfer_root);
  C->num_interp_redshifts = B.num_interp_redshifts;
  for(int i=0; i<B.num_interp_redshifts; i++)
    C->z_interp_redshifts[i] = B.z_interp_redshifts[i];
  
  //set fixed/derived parameters
  C->N_tau = B.N_tau;
  C->N_nu_eff = B.N_nu_eff;
  C->N_nu_massive = B.N_nu_massive;
  C->Omega_cb_0 = B.Omega_cb_0;
  C->Omega_gam_0 = B.Omega_gam_0;
  C->Omega_nurel_0 = B.Omega_nurel_0;
  C->Omega_nugam_0 = B.Omega_nugam_0;
  C->Omega_rel_0 = B.Omega_rel_0;
  C->Omega_de_0 = B.Omega_de_0;
  C->Omega_m_h2_0 = B.Omega_m_h2_0;
  C->Omega_b_h2_0 = B.Omega_b_h2_0;
  for(int t=0; t<B.N_tau; t++) C->Omega_hdm_t_0[t] = B.Omega_hdm_t_0[t];

  C->w_eos_cdm = 0;
  C->w_eos_gam = 0.333333333333333333;
  C->f_cb_0 = B.f_cb_0;
  C->f_hdm_0 = B.f_hdm_0;

  C->sound_horiz = B.sound_horiz;
  C->alpha_G = B.alpha_G;
  C->Theta_CMB_27_Sq = B.Theta_CMB_27_Sq;

  C->initAggcb = 0;
  C->nAgghdm = 0;
  
  return 0;
}

double cosmoparam_fdiff(double x, double y){
  return 2.0 * fabs(x-y) / (fabs(x) + fabs(y) + 1e-100);
}

int isequal_cosmoparam(const struct cosmoparam B, const struct cosmoparam C){
  int equal = (B.switch_nonlinear == C.switch_nonlinear);
  equal = equal && ( B.switch_Nmunl == C.switch_Nmunl );
  equal	= equal	&& ( B.switch_print == C.switch_print );
  equal = equal && ( B.switch_transfer_type == C.switch_transfer_type );
  equal = equal && ( B.num_z_outputs == C.num_z_outputs );
  equal = equal && ( B.num_massive_hdm_approx == C.num_massive_hdm_approx );
  equal = equal && ( B.N_tau == C.N_tau );
  equal = equal && ( B.initAggcb == C.initAggcb );
  equal = equal && ( B.nAgghdm == C.nAgghdm );
  if(!equal) return 0;
  if(strcmp(B.file_transfer_function, C.file_transfer_function) != 0) return 0;
  if(strcmp(B.file_hdm_transfer_root, C.file_hdm_transfer_root) != 0) return 0;
  if(strcmp(B.file_hdm_distribution, C.file_hdm_distribution) != 0) return 0;

  double fdmax = cosmoparam_fdiff(B.n_s,C.n_s);
  fdmax = fmax( fdmax, cosmoparam_fdiff(B.sigma_8,C.sigma_8) );
  fdmax	= fmax( fdmax, cosmoparam_fdiff(B.h,C.h) );
  fdmax = fmax( fdmax, cosmoparam_fdiff(B.Omega_m_0,C.Omega_m_0) );
  fdmax = fmax( fdmax, cosmoparam_fdiff(B.Omega_b_0,C.Omega_b_0) );
  fdmax = fmax( fdmax, cosmoparam_fdiff(B.Omega_hdm_tot_0,C.Omega_hdm_tot_0) );
  fdmax = fmax( fdmax, cosmoparam_fdiff(B.T_CMB_0_K,C.T_CMB_0_K) );
  fdmax = fmax( fdmax, cosmoparam_fdiff(B.w0_eos_de,C.w0_eos_de) );
  fdmax = fmax( fdmax, cosmoparam_fdiff(B.wa_eos_de,C.wa_eos_de) );
  fdmax = fmax( fdmax, cosmoparam_fdiff(B.T_hdm_0_K,C.T_hdm_0_K) );
  fdmax = fmax( fdmax, cosmoparam_fdiff(B.m_hdm_eV,C.m_hdm_eV) );

  for(int t=0; t<B.N_tau; t++)
    fdmax = fmax(fdmax,cosmoparam_fdiff(B.Omega_hdm_t_0[t],C.Omega_hdm_t_0[t]));

  for(int i=0; i<B.num_z_outputs; i++)
    fdmax = fmax( fdmax, cosmoparam_fdiff(B.z_outputs[i],C.z_outputs[i]) );

  for(int i=0; i<N_EQ; i++)
    fdmax = fmax( fdmax, cosmoparam_fdiff(B.yAgg[i],C.yAgg[i]) );

  if(B.initAggcb>1){
    for(int i=0; i<N_UI*NK; i++)
      fdmax = fmax( fdmax, cosmoparam_fdiff(B.Aggcb[i],C.Aggcb[i]) );
  }

  if(B.nAgghdm>0){
    for(int i=0; i<B.nAgghdm*N_UI*N_MU*NK; i++)
      fdmax = fmax( fdmax, cosmoparam_fdiff(B.Agghdm[i],C.Agghdm[i]) );
    for(int i=0; i<B.nAgghdm*N_UI*NK; i++)
      fdmax = fmax( fdmax, cosmoparam_fdiff(B.Agghdm0[i],C.Agghdm0[i]) );
  }
    
  return (fdmax < 1e-6);
}

