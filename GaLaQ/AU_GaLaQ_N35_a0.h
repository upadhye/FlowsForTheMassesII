//    Copyright 2024 Amol Upadhye
//
//    This file is part of FlowsForTheMassesII.
//
//    FlowsForTheMassesII is free software: you can redistribute
//    it and/or modify it under the terms of the GNU General
//    Public License as published by the Free Software
//    Foundation, either version 3 of the License, or (at
//    your option) any later version.
//
//    FlowsForTheMassesII is distributed in the hope that it
//    will be useful, but WITHOUT ANY WARRANTY; without
//    even the implied warranty of MERCHANTABILITY or
//    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
//    Public License for more details.
//
//    You should have received a copy of the GNU General
//    Public License along with FlowsForTheMassesII.  If not,
//    see <http://www.gnu.org/licenses/>.

// Gauss-Legendre Quadrature with N=35, alpha=0
const int GALAQ_N = 35;

const double GALAQ_alpha = 0;

const double GALAQ_q[] = {4.072920906171344918e-02,2.146874527351445572e-01,5.280103843193437729e-01,9.813861734590779706e-01,1.575725947577451880e+00,2.312228296513212378e+00,3.192397939493119896e+00,4.218064125029716394e+00,5.391402734673823360e+00,6.714963279915739491e+00,8.191701758837620417e+00,9.825020497092705085e+00,1.161881638890733726e+01,1.357753935730780448e+01,1.570626339576277708e+01,1.801077328783512499e+01,2.049767110714662977e+01,2.317450799779925319e+01,2.604994871037023429e+01,2.913397920954442810e+01,3.243817183767793466e+01,3.597602876989402176e+01,3.976343410480026819e+01,4.381926011860667103e+01,4.816619797401877889e+01,5.283192505801562788e+01,5.785079502213444158e+01,6.326637367538476298e+01,6.913541388364815532e+01,7.553443513472991810e+01,8.257140552358264074e+01,9.040852386440074895e+01,9.931297365944135436e+01,1.097959909449064781e+02,1.231732531753759190e+02};

const double GALAQ_w[] = {1.003622374192361943e-01,1.964823833617123927e-01,2.260145669343438524e-01,1.962716678964447070e-01,1.376007420131272196e-01,8.003040046414028330e-02,3.912577142888327281e-02,1.618672652612587748e-02,5.685325571774348975e-03,1.697223299125987959e-03,4.304850681416785884e-04,9.263405306365354961e-05,1.687026486931441144e-05,2.591672147427984968e-06,3.344610957266338777e-07,3.607785438691556486e-08,3.233569032210222515e-09,2.391322981592160871e-10,1.447332399038879687e-11,7.101355277811633596e-13,2.793369509321614133e-14,8.694840214330673469e-16,2.108845936777164086e-17,3.912910811923232624e-19,5.432719406706797734e-21,5.493694724075139474e-23,3.912692565636768339e-25,1.880986928220191972e-27,5.775155104394526630e-30,1.051140746780541074e-32,1.021277013187401728e-35,4.527347312081493173e-39,7.080232733926100721e-43,2.397494633365148340e-47,5.106723813700605722e-53};

