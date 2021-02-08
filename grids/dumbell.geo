// Gmsh project created on Mon Feb  8 12:57:54 2021
SetFactory("OpenCASCADE");
//+

L = 3.141592653589793;

Point(1) = {0, 0, 0, 1.0};
Point(2) = {0, 3*L/8, 0, 1.0};
Point(3) = {0, 5*L/8, 0, 1.0};
Point(4) = {0, L, 0, 1.0};//+

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
//+
Extrude {L, 0, 0} {
  Curve{3}; Curve{2}; Curve{1}; Layers{5}; Recombine;
}
//+
Extrude {L/4, 0, 0} {
  Curve{8}; Layers{1}; Recombine;
}
//+
Point(11) = {L+L/4, L, 0, 1.0};
//+
Point(12) = {L+L/4, 0, 0, 1.0};
//+
Line(14) = {12, 9};
//+
Line(15) = {10, 11};
//+
Extrude {L, 0, 0} {
  Curve{15}; Curve{13}; Curve{14}; Layers{5}; Recombine;
}