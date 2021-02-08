Include "fichera.geo";

//+
Extrude {0, 0, L} {
  Surface{1};
  Surface{2};
  Surface{3}; 
  
  Layers{1}; 
  Recombine;
}

Physical Volume("material", 1) = {1, 2, 3};