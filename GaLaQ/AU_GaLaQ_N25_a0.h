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

// Gauss-Legendre Quadrature with N=25, alpha=0
const int GALAQ_N = 25;

const double GALAQ_alpha = 0;

const double GALAQ_q[] = {5.670477545270547620e-02,2.990108985869885383e-01,7.359095554350161272e-01,1.369183116035193715e+00,2.201326053721467613e+00,3.235675803558037078e+00,4.476496615073834562e+00,5.929083762700448013e+00,7.599899309956749782e+00,9.496749220932434454e+00,1.162901491177875357e+01,1.400795797654506814e+01,1.664712559728878460e+01,1.956289801146905560e+01,2.277524198683504153e+01,2.630877239096888687e+01,3.019429116331610530e+01,3.447109757192203006e+01,3.919060880393742252e+01,4.442234933616202142e+01,5.026457499383354133e+01,5.686496717394017253e+01,6.446667061595412918e+01,7.353423479210015046e+01,8.526015556249595306e+01};

const double GALAQ_w[] = {1.375260142293442955e-01,2.516452737649096938e-01,2.561760028097561093e-01,1.862154903624367586e-01,1.031998481075206420e-01,4.471416112993354536e-02,1.530523288639546765e-02,4.152414632877078973e-03,8.920990732596821050e-04,1.511560191642398032e-04,2.006553180193290943e-05,2.067774396431873941e-06,1.634652022291141601e-07,9.766015062124489395e-09,4.327720794184915195e-10,1.389600963389519624e-11,3.138922792539935349e-13,4.802614822604131685e-15,4.735885364807260498e-17,2.814205379843014883e-19,9.164954395991364010e-22,1.418940009497260814e-24,8.273651944099044399e-28,1.168881711542660551e-31,1.315831500059174607e-36};

