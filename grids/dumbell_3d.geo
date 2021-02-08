Include "dumbell.geo";

//+
Extrude {0, 0, L} {
  Surface{1};  Surface{2};  Surface{3};  Surface{4};  Surface{5}; 
  Surface{6};  Surface{7};
  Layers{5}; Recombine;
}

//+
Physical Volume("material", 1) = {3, 2, 1, 4, 7, 6, 5};