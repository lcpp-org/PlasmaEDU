function SY = yamamura( ENERGY_eV, PROJECTILE, TARGET )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This function implements the semi-empirical formulas obtained by
%    Yamamura and Tawara for the calculation of ion-induced Sputtering Yield
%    from monatomic solids at normal incidence.
%
% YASUNORI YAMAMURA and HIRO TAWARA, ENERGY DEPENDENCE OF ION-INDUCED
%    SPUTTERING YIELDS FROM MONATOMIC SOLIDS AT NORMAL INCIDENCE,
%    ATOMIC DATA AND NUCLEAR DATA TABLES 62, 149–253 (1996)
%
% Input legend
%
%    ENERGY_eV  :  Energy [eV] of the incident projectile (es: [10:1:100] )
%    PROJECTILE :  Chemical symbol of the projectile (es: 'He3', helium-3)
%    TARGET     :  Chemical symbol (IUPAC) of the target (es: 'Be', berillium)
%
% Example of usage, for He --> Be
%
%    SY = yamamura( ENERGY_eV, PROJECTILE, TARGET )
%
%    Calculation of the sputtering yield of Helium-4 ions (Z1=2, M1=4.00)
%    incident on a Berillium-9 target (Z2=4, M2=9.01), done for energies
%    from 10eV to 1000eV with a step of 100eV:
%
%    SY = yamamura( [10:100:1000], 'He', 'Be');
%
% Author
%
%    Davide Curreli
%    University of Illinois at Urbana-Champaign
%
% Errata Corrige
%
%    Please send errata corrige to:
%    dcurreli at illinois dot edu
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch PROJECTILE
    %    M1  :  Mass [amu] of the projectile (es: M1 = 3.016, helium-3)
    %    Z1  :  Atomic number of projectile (es: Z1 = 2, helium)
    case 'H',   M1 = 1.008;  Z1 = 1;
    case 'D',   M1 = 2.014;  Z1 = 1;
    case 'He3', M1 = 3.016;  Z1 = 2;
    case 'He',  M1 = 4.002;  Z1 = 2;
    case 'Li',  M1 = 6.941;  Z1 = 3;
    case 'Be',  M1 = 9.012;  Z1 = 4;
    case 'B',   M1 = 10.81;  Z1 = 5;
    case 'C',   M1 = 12.01;  Z1 = 6;
    case 'N',   M1 = 14.007; Z1 = 7;
    case 'O',   M1 = 15.999; Z1 = 8;
    case 'F',   M1 = 18.998; Z1 = 9;
    case 'Ne',  M1 = 20.179; Z1 = 10;
    case 'Na',  M1 = 22.989; Z1 = 11;
    case 'Al',  M1 = 26.982; Z1 = 13;
    case 'Si',  M1 = 28.085; Z1 = 14;
    case 'P',   M1 = 30.974; Z1 = 15;
    case 'S',   M1 = 32.060; Z1 = 16;
    case 'Cl',  M1 = 35.453; Z1 = 17;
    case 'Ar',  M1 = 39.949; Z1 = 18;
    case 'K',   M1 = 39.098; Z1 = 19;
    case 'Ni',  M1 = 58.700; Z1 = 28;
    case 'Zn',  M1 = 65.380; Z1 = 30;
    case 'As',  M1 = 74.922; Z1 = 33;
    case 'Se',  M1 = 78.960; Z1 = 34;
    case 'Br',  M1 = 79.904; Z1 = 35;
    case 'Kr',  M1 = 83.798; Z1 = 36;
    case 'Cd',  M1 = 112.41; Z1 = 48;
    case 'Sb',  M1 = 121.75; Z1 = 51;
    case 'Te',  M1 = 127.60; Z1 = 52;
    case 'Xe',  M1 = 131.29; Z1 = 54;
    case 'Cs',  M1 = 132.91; Z1 = 55;
    case 'Hg',  M1 = 200.59; Z1 = 80;
    case 'Tl',  M1 = 204.37; Z1 = 81;
    case 'Pb',  M1 = 207.20; Z1 = 82;
    case 'Bi',  M1 = 208.98; Z1 = 83;
end

switch TARGET
    %   M2   Mass [amu] of the target (es: M2 = 9.01, berillium)
    %   Z2   Atomic number of the target (es: Z2 = 4, berillium)
    % Yamamura best-fit parameters for target (Table I):
    %   Us   Surface binding energy of the target in [eV]
    %   Q    Yamamura fitting parameter for Eq. 15
    %   W    Yamamura factor for the Gamma expression Eq. 16
    %   s    Yamamura exponent in Eq. 15
    case 'Be', M2 = 9.012;  Z2 = 4;  Us = 3.32; Q = 1.66; W = 2.32; s = 2.50;
    case 'B',  M2 = 10.81;  Z2 = 5;  Us = 5.77; Q = 2.62; W = 4.39; s = 2.50;
    case 'C',  M2 = 12.01;  Z2 = 6;  Us = 7.37; Q = 1.70; W = 1.84; s = 2.50;
    case 'Al', M2 = 26.98;  Z2 = 13; Us = 3.39; Q = 1.00; W = 2.17; s = 2.50;
    case 'Si', M2 = 28.08;  Z2 = 14; Us = 4.63; Q = 0.66; W = 2.32; s = 2.50;
    case 'Ti', M2 = 47.90;  Z2 = 22; Us = 4.85; Q = 0.54; W = 2.57; s = 2.50;
    case 'V',  M2 = 50.94;  Z2 = 23; Us = 5.31; Q = 0.72; W = 2.39; s = 2.50;
    case 'Cr', M2 = 51.99;  Z2 = 24; Us = 4.10; Q = 0.93; W = 1.44; s = 2.50;
    case 'Mn', M2 = 54.94;  Z2 = 25; Us = 2.92; Q = 0.95; W = 0.88; s = 2.50;
    case 'Fe', M2 = 55.85;  Z2 = 26; Us = 4.28; Q = 0.75; W = 1.20; s = 2.50;
    case 'Co', M2 = 58.93;  Z2 = 27; Us = 4.39; Q = 1.02; W = 1.54; s = 2.50;
    case 'Ni', M2 = 58.70;  Z2 = 28; Us = 4.44; Q = 0.94; W = 1.33; s = 2.50;
    case 'Cu', M2 = 63.55;  Z2 = 29; Us = 3.49; Q = 1.00; W = 0.73; s = 2.50;
    case 'Ge', M2 = 72.59;  Z2 = 32; Us = 3.85; Q = 0.59; W = 2.08; s = 2.50;
    case 'Zr', M2 = 91.22;  Z2 = 40; Us = 6.25; Q = 0.54; W = 2.50; s = 2.80;
    case 'Nb', M2 = 92.91;  Z2 = 41; Us = 7.57; Q = 0.93; W = 2.65; s = 2.80;
    case 'Mo', M2 = 95.94;  Z2 = 42; Us = 6.82; Q = 0.85; W = 2.39; s = 2.80;
    case 'Ru', M2 = 101.07; Z2 = 44; Us = 6.74; Q = 1.31; W = 2.36; s = 2.50;
    case 'Rh', M2 = 102.91; Z2 = 45; Us = 5.75; Q = 1.14; W = 2.59; s = 2.50;
    case 'Pd', M2 = 106.40; Z2 = 46; Us = 3.89; Q = 0.85; W = 1.36; s = 2.50;
    case 'Ag', M2 = 107.87; Z2 = 47; Us = 2.95; Q = 1.08; W = 1.03; s = 2.80;
    case 'Sn', M2 = 118.69; Z2 = 50; Us = 3.14; Q = 0.47; W = 0.88; s = 2.50;
    case 'Tb', M2 = 158.93; Z2 = 65; Us = 4.05; Q = 0.90; W = 1.42; s = 2.50;
    case 'Tm', M2 = 168.93; Z2 = 69; Us = 2.42; Q = 0.65; W = 0.85; s = 2.50;
    case 'Hf', M2 = 178.49; Z2 = 72; Us = 6.44; Q = 0.65; W = 2.25; s = 2.50;
    case 'Ta', M2 = 180.95; Z2 = 73; Us = 8.10; Q = 0.56; W = 2.84; s = 2.80;
    case 'W',  M2 = 183.84; Z2 = 74; Us = 8.90; Q = 0.72; W = 2.14; s = 2.80;
    case 'Re', M2 = 186.21; Z2 = 75; Us = 8.03; Q = 1.03; W = 2.81; s = 2.50;
    case 'Os', M2 = 190.20; Z2 = 76; Us = 8.17; Q = 1.11; W = 2.86; s = 2.50;
    case 'Ir', M2 = 192.22; Z2 = 77; Us = 6.94; Q = 0.96; W = 2.43; s = 2.50;
    case 'Pt', M2 = 195.09; Z2 = 78; Us = 5.84; Q = 1.03; W = 3.21; s = 2.50;
    case 'Au', M2 = 196.97; Z2 = 79; Us = 3.81; Q = 1.08; W = 1.64; s = 2.80;
    case 'Th', M2 = 232.04; Z2 = 90; Us = 6.20; Q = 0.63; W = 2.79; s = 2.50;
    case 'U',  M2 = 238.03; Z2 = 92; Us = 5.55; Q = 0.66; W = 2.78; s = 2.50;

    otherwise

    fprintf('\n');
    fprintf('Warning: Target not in the Yamamura-Tawara database of \n\n');
    fprintf('   "YASUNORI YAMAMURA and HIRO TAWARA, ENERGY DEPENDENCE\n');
    fprintf('   OF ION-INDUCED SPUTTERING YIELDS FROM MONATOMIC SOLIDS AT\n');
    fprintf('   NORMAL INCIDENCE, ATOMIC DATA AND NUCLEAR DATA TABLES 62,\n');
    fprintf('   149-253 (1996)"\n\n');
    fprintf('Evaluating the sputtering yield with approximated values \n');
    fprintf('   for the target, as recommended by Yamamura and Tawara: \n\n');
    fprintf('   Q=1.0, s=2.5, W=0.35*Us,\n\n');
    fprintf('Please enter the target info :\n\n');
    M2 = input('   Mass [amu] of the target = ');
    Z2 = input('   Atomic number Z of the target = ');
    Us = input('   Heat of sublimation of the target in [eV] = ');
    fprintf('\n\n');
    Q = 1.0; s = 2.5; W = 0.35*Us;

end

% Correct zeros
I = find(ENERGY_eV==0);
ENERGY_eV(I) = 10*eps;

% Reduced energy
epsE = 0.03255/Z1/Z2/((Z1^(2/3)+Z2^(2/3))^0.5)*M2/(M1+M2)*ENERGY_eV;
sqrtE = sqrt(epsE);

% Lindhard electronic stopping coefficient
ke = 0.079*(M1+M2)^1.5/M1^1.5/M2^0.5*Z1^(2/3)*Z2^0.5/((Z1^(2/3)+Z2^(2/3))^(3/4));

% Reduced nuclear stopping power based on the Thomas-Fermi potential
s_n_TF = 3.441*sqrtE.*log(epsE+2.1718)./(1.0+6.355*sqrtE+epsE.*(6.882*sqrtE-1.708));

% Nuclear stopping cross section
Sn = 84.78*Z1*Z2/((Z1^(2/3)+Z2^(2/3))^0.5)*M1/(M1+M2)*s_n_TF;

% Small gamma
gamma = 4*M1*M2/(M1+M2)/(M1+M2);

% Fitting alpha and sputtering threshold
if (M1<=M2)
    alpha_star = 0.249*(M2/M1)^0.56 + 0.0035*(M2/M1)^1.5;
    E_th = Us * (1.0+5.7*(M1/M2))/gamma;
else
    alpha_star = 0.0875*(M2/M1)^-0.15 + 0.165*(M2/M1);
    E_th = Us * 6.7/gamma;
end

% Capital Gamma
Gamma = W/(1.0+(M1/7)^3.0);

% Yamamura Sputtering Yield
SY = 0.042*Q*alpha_star/Us*Sn./(1.0+Gamma*ke*epsE.^0.3).*(1.0-sqrt(E_th./ENERGY_eV)).^s;
