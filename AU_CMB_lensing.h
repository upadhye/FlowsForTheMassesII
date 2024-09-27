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

//////////////////////////////////////////////////////////////////////////////////arrays for k and ell
const double ellkapmax = 25000, dlnell=log(ellkapmax)/(NK-1);
//double lnkArr[NK], kArr[NK], lnellArr[NK], ellArr[NK];

int init_k_ell_arrays(void){
  for(int i=0; i<NK; i++){
    lnkArr[i] = LNKMIN + DLNK*i;
    kArr[i] = exp(lnkArr[i]);

    if(i==0) ellArr[i] = 1;
    else if(i==NK-1) ellArr[i] = ellkapmax;
    else{
      double Dellkap_i = log(ellkapmax/ellArr[i-1]) / (NK-i);
      ellArr[i] = floor(1 + ellArr[i-1]*exp(Dellkap_i));
    }
    
    lnellArr[i] = log(ellArr[i]);
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////////
//comoving distance

double H0chi_integrand(double z, void *params){
  struct cosmoparam *C = (struct cosmoparam *)params;
  double aeta = 1.0 / (1.0 + z), eta = log(aeta/aeta_in);
  double H_H0_eta = sqrt(Hc2_Hc02_eta(eta, *C)) / aeta;
  return 1.0 / H_H0_eta;
}

int H0chi_eta_init(int N_H0CHI, double *eta_chi_i, double *H0chi_i, 
                   const struct cosmoparam C){

  double zmin=1e-4, zmax=1e4, dlnz = log(zmax/zmin) / (N_H0CHI-1), zlast=0;
  struct cosmoparam C2;
  copy_cosmoparam(C,&C2);
  
  for(int i=0; i<N_H0CHI; i++){
    double z = zmin * exp(dlnz * i), aeta = 1.0/(1.0+z), DH0chi, dum;
    eta_chi_i[N_H0CHI-1-i] = log(aeta / aeta_in);

    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
    gsl_function F;
    F.function = &H0chi_integrand;
    F.params = &C2;
    gsl_integration_qag(&F, zlast, z, 0, 1e-4, 1000, 6, w, &DH0chi, &dum);

    zlast = z;
    H0chi_i[N_H0CHI-1-i] = (i==0 ? DH0chi : DH0chi + H0chi_i[N_H0CHI-i]);
  }
  return 0;
}

double H0chi(double eta, const struct cosmoparam C){
  const int N_H0CHI = 1000;
  static double eta_chi_i[N_H0CHI], H0chi_i[N_H0CHI];

  double aeta = aeta_in * exp(eta), zaeta = 1.0/aeta - 1.0;
  if(zaeta <= 1e-4) return zaeta;

  static int init = 0;
  if(!init){ 
    H0chi_eta_init(N_H0CHI, eta_chi_i, H0chi_i, C); 
    init=1; 
  }

  static tabulated_function H0chiInterp(N_H0CHI, eta_chi_i, H0chi_i);
  return H0chiInterp(eta);
}

