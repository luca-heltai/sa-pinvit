// Gmsh project created on Mon Feb  8 12:30:49 2021
SetFactory("OpenCASCADE");

L = 1.0;

//+
Point(1) = {-L, -L, 0, L};
Point(2) = {0, -L, 0, L};
Point(3) = {L, -L, 0, L};

Line(1) = {1, 2};
Line(2) = {2, 3};

Extrude {0, L, 0} {
  Curve{1}; Curve{2}; Layers{1}; Recombine;
}
Extrude {0, L, 0} {
  Curve{5}; Layers{1}; Recombine;
}
