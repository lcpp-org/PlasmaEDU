import numpy as np

def yamamura(ion, target, energy_eV):
    '''
    Semi-empirical Yamamura sputtering yield formula for normal incidence.

    References:
    
    [1] Y. Yamamura, H. Tawara, Energy Dependence of ion-induced  
        sputtering yields from monoatomic solids at normal incidence, 
        Atomic Data and Nuclear Data Tables, Vol. 62, No. 2, 149-253 (1996)
        https://doi.org/10.1006/adnd.1996.0005 

    [2] Erratum: Volume 62, No. 2 (1995) Energy Dependence of Ion-Induced 
        Sputtering Yields from Monatomic Solids at Normal Incidence, 
        YASUNORI YAMAMURA, HIRO TAWARA, Atimic Data and Nuclear Data Tables,
        Vol. 63, 353 (1996)
        https://doi.org/10.1006/adnd.1996.0016 

    Args:
    ----
        ion (dict): 
            A dictionary with the fields: 
                    Z (atomic number)
                    m (atomic mass)
        target (dict): 
            A dictionary with the fields:
                    Z (atomic number)
                    m (atomic mass)
                    Es (surface binding energy)
                    Q (Yamamura coefficient)
                    W (Yamamura coefficient)
                    s (Yamamura coefficient)
        energy_eV (float): 
            Energy in electron-volts
    Returns:
    -------
        Y (float): 
            Sputtering yield in atoms/ion
        Eth (float):
            Sputtering threshold in eV
    '''
    # Properties of the ion
    M1 = ion['m']
    Z1 = ion['Z']

    # Properties of the target
    M2 = target['m']
    Z2 = target['Z']
    Us = target['Es']
    Q  = target['Q']
    W  = target['W']
    s  = target['s']

    # Reduced masses of the ion and target
    Mr_1 = M1 / (M1 + M2)
    Mr_2 = M2 / (M1 + M2)
    
    # Fractions 
    F12 = 1.0 / 2.0
    F23 = 2.0 / 3.0
    F34 = 3.0 / 4.0

    # Lindhard screening length, Eq. (22) Ref. [1]
    a_L = 0.03255 / ((Z1**F23 + Z2**F23)**F12)

    # Lindhard reduced energy in the Thomas-Fermi form, Eq. (22) Ref. [1]
    epsE = 1.0/Z1/Z2 * Mr_2 * energy_eV * a_L
    sqrtE = np.sqrt(epsE)

    # Lindhard-Scharff electronic stopping coefficient, Eq. (20) Ref. [1]
    ke = 0.079 * (M1 + M2)**1.5 / M1**1.5 / M2**F12 * Z1**F23 * Z2**F12 / (Z1**F23 + Z2**F23)**F34
    se = ke * sqrtE

    # Lindhard-Scharff-Schiott reduced nuclear  
    # cross section based on the Thomas-Fermi potential, Eq. (4) Ref. [1]
    sn_TF = 3.441 * sqrtE * np.log(epsE + 2.718) / (1.0 + 6.355*sqrtE + epsE*(6.882*sqrtE - 1.708))

    # Nuclear stopping cross section, Eq. (21) Ref. [1]
    # Note that Eq. (21) in the original paper has a typo, 
    # the coefficient 8.478 should read 84.78, as noted in the Erratum Ref. [2] 
    # published in the same journal, https://doi.org/10.1006/adnd.1996.0016 
    Sn = 84.78 * Z1 * Z2 / ((Z1**F23 + Z2**F23)**F12) * Mr_1 * sn_TF

    # Small gamma, energy transfer factor in elastic collisions, Eq. (19) Ref. [1]
    gamma = 4.0 * M1 * M2 / (M1 + M2)**2

    # Yamamura empirical alpha Eq. (17) Ref. [1]
    # and sputtering threshold Eq. (18) Ref. [1]
    if (M1<=M2):
        alpha_star = 0.249*(M2/M1)**0.56 + 0.0035*(M2/M1)**1.5
        Eth = Us * (1.0 + 5.7*(M1/M2)) / gamma
    else:
        alpha_star = 0.0875*(M2/M1)**-0.15 + 0.165*(M2/M1)
        Eth = Us * 6.7 / gamma

    # Capital Gamma, Eq. (16) Ref. [1]
    Gamma = W/(1.0+(M1/7.0)**3.0)

    # Yamamura Sputtering Yield, Eq. (15) Ref. [1] 
    SY = 0.042 * Q * alpha_star / Us * Sn/(1.0+Gamma*ke*epsE**0.3) * (1.0-np.sqrt(Eth/energy_eV))**s
    SY = np.real(SY)
    
    return SY, Eth
