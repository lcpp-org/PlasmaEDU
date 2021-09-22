Include "EM01.par";

Group {
  Boundary1  = Region[10];
  Boundary2  = Region[15];
  FreeSpace  = Region[12];
  Dielectric = Region[13];
  WaveDomain = Region[{FreeSpace,Dielectric}];
  AllDomain  = Region[{WaveDomain,Boundary1,Boundary2}];
}


Function {
  nu[ AllDomain ] = nu0;
  epsilon[ FreeSpace  ] = Tensor[ eps0,0,0, 0,eps0,0, 0,0,eps0 ];
  epsilon[ Dielectric ] = Tensor[ eps1,0,0, 0,eps1,0, 0,0,eps1 ];
  FunTime[] = Sin[Omega*$Time + Phase];
  SaveFct[] = !($TimeStep % 1);
}


Constraint {
  { Name ValueOfEfield ; Type Assign;
    Case {
      { Region Boundary1    ; Value 1.0 ; TimeFunction FunTime[]; }
      { Region Boundary2    ; Value 0.0 ; }
    }
  }
}


Jacobian {
  { Name JVol ; Case { { Region All ; Jacobian Vol ; } } }
  { Name JSur ; Case { { Region All ; Jacobian Sur ; } } }
}


Integration {
  { Name IntGauss ;
    Case {
      { Type Gauss ;
        Case {
      	  { GeoElement Point       ; NumberOfPoints  1 ; }
      	  { GeoElement Line        ; NumberOfPoints  3 ; }
      	  { GeoElement Triangle    ; NumberOfPoints  4 ; }
      	  { GeoElement Quadrangle  ; NumberOfPoints  4 ; }
      	  { GeoElement Tetrahedron ; NumberOfPoints  4 ; }
      	  { GeoElement Hexahedron  ; NumberOfPoints  6 ; }
      	  { GeoElement Prism       ; NumberOfPoints  6 ; }
      	}
      }
    }
  }
}


FunctionSpace {
  { Name Space_E_1D; Type Form1P;
    BasisFunction {
      { Name sn; NameOfCoef en; Function BF_PerpendicularEdge;
        Support AllDomain; Entity NodesOf[All]; }
    }
    Constraint {
      { NameOfCoef en;  EntityType NodesOf ; NameOfConstraint ValueOfEfield; }
    }
  }
}


Formulation {
  { Name FEM_E_1D; Type FemEquation;
    Quantity {
      { Name E;  Type Local; NameOfSpace Space_E_1D; }
    }
    Equation {
      Galerkin { DtDt [ epsilon[]  * Dof{E} , {E} ];
            In WaveDomain; Integration IntGauss; Jacobian JVol;  }
      Galerkin { [ nu[] * Dof{Curl E} , {Curl E} ];
            In WaveDomain; Integration IntGauss; Jacobian JVol;  }
    }
  }
}


Resolution {

  { Name Efield1D_Newmark;
    System { { Name SOL; NameOfFormulation FEM_E_1D; } }
    Operation {
      InitSolution[SOL] ;
      InitSolution[SOL] ;
      TimeLoopNewmark[t0,t1,dt,0.25,0.5]
      { Generate[SOL] ; Solve[SOL] ; Test[ SaveFct[] ]{ SaveSolution[SOL] ; }
      }
    }
  }

  { Name Efield1D_Eigenvalues;
    System { { Name SOL; NameOfFormulation FEM_E_1D; Type Complex; } }
    Operation {
        GenerateSeparate[SOL] ;
        EigenSolve[SOL,20,1e6,100e6];
        SaveSolution[SOL] ; }
  }

}


PostProcessing {

  { Name Etime1D; NameOfFormulation FEM_E_1D; NameOfSystem SOL;
    Quantity {
      { Name E;   Value{ Local{ [ {E} ] ; In AllDomain; } } }
      { Name Ez;  Value{ Local{ [ CompZ[{E}] ] ; In AllDomain; } } }
    }
  }

}


PostOperation {

  { Name E_field_1D; NameOfPostProcessing Etime1D ;
    Operation {
      Print[ E,  OnElementsOf AllDomain , File "E.pos" ] ;
      Print[ Ez,  OnElementsOf AllDomain , File "Ez.pos" ] ;
      Print[ Ez,  OnLine { {a,0,0} {2*b,0,0} } {150}, Format Table, File "Ez.dat" ];
    }
  }

}
