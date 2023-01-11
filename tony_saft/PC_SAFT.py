# -*- coding: utf-8 -*-
"""
Spyder Editor
# Author: Antonio Cavalcante de Lima Neto
# Github: https://github.com/tonyCLN
# Date: 14-10-2022
# Updated: 14-11-2022

References:
    
Gross, J., & Sadowski, G. (2001). 
Perturbed-Chain SAFT: An Equation of State 
Based on a Perturbation Theory for Chain Molecules
Joachim. Industrial and Engineering Chemistry Research, 40, 1244–1260.
 https://doi.org/10.1021/ie0003887
    
Gross, J., & Sadowski, G. (2019). Reply to Comment on 
“perturbed-Chain SAFT: An Equation of State Based on 
a Perturbation Theory for Chain Molecules.” Industrial 
and Engineering Chemistry Research, 58(14), 5744–
5745. https://doi.org/10.1021/acs.iecr.9b01515

Michelsen, M. L., & Hendriks, E. M. (2001). 
Physical properties from association models. 
Fluid Phase Equilibria, 180(1–2), 165–174. 
https://doi.org/10.1016/S0378-3812(01)00344-2

X association algorithm:
Tan, S. P., Adidharma, H., & Radosz, M. (2004).
Generalized Procedure for Estimating the Fractions of 
Nonbonded Associating Molecules and Their Derivatives in 
Thermodynamic Perturbation Theory. Industrial and Engineering 
Chemistry Research, 43(1), 203–208. https://doi.org/10.1021/ie034041q

Delta funtion:
Tan, S. P., Adidharma, H., & Radosz, M. (2004). Generalized Procedure 
for Estimating the Fractions of Nonbonded Associating Molecules and 
Their Derivatives in Thermodynamic Perturbation Theory. Industrial 
and Engineering Chemistry Research, 43(1), 203–208. 
https://doi.org/10.1021/ie034041q

"""
import numpy as np
from scipy.constants import k as kb  # constante de Boltzmann J /K
from scipy.constants import Avogadro as Navo  # numero de avogadro mol^-1
from scipy.constants import pi
from scipy import optimize
import numpy as np
import numdifftools as nd


class PC_SAFT_EOS():
    def __init__(self, m, sigma, epsilon_k, M=None, kbi=None, kAB_k=None, eAB_k=None, S=None,deltasimplified=True):
        if type(m) is np.ndarray:
            self.ncomp = len(m)
        else:
            self.ncomp = 1
        self.M = M
        self.m = m
        self.sigma = sigma
        self.epsilon_k = epsilon_k
        self.eAB_k = eAB_k
        self.kAB_k = kAB_k
        self.S = S
        if kbi is None:
            self.kbi = np.zeros([self.ncomp, self.ncomp])
        else:
            self.kbi = kbi

        ap = np.zeros([7, 3])
        bp = np.zeros([7, 3])

        # table 1
        ap[0, 0] = 0.91056314451539e0
        ap[0, 1] = -0.30840169182720e0
        ap[0, 2] = -0.09061483509767e0
        ap[1, 0] = 0.63612814494991e0
        ap[1, 1] = 0.18605311591713e0
        ap[1, 2] = 0.45278428063920e0
        ap[2, 0] = 2.68613478913903e0
        ap[2, 1] = -2.50300472586548e0
        ap[2, 2] = 0.59627007280101e0
        ap[3, 0] = -26.5473624914884e0
        ap[3, 1] = 21.4197936296668e0
        ap[3, 2] = -1.72418291311787e0
        ap[4, 0] = 97.7592087835073e0
        ap[4, 1] = -65.2558853303492e0
        ap[4, 2] = -4.13021125311661e0
        ap[5, 0] = -159.591540865600e0
        ap[5, 1] = 83.3186804808856e0
        ap[5, 2] = 13.7766318697211e0
        ap[6, 0] = 91.2977740839123e0
        ap[6, 1] = -33.7469229297323e0
        ap[6, 2] = -8.67284703679646e0

        bp[0, 0] = 0.72409469413165e0
        bp[0, 1] = -0.57554980753450e0
        bp[0, 2] = 0.09768831158356e0
        bp[1, 0] = 1.11913959304690e0 * 2.e0
        bp[1, 1] = 0.34975477607218e0 * 2.e0
        bp[1, 2] = -0.12787874908050e0 * 2.e0
        bp[2, 0] = -1.33419498282114e0 * 3.e0
        bp[2, 1] = 1.29752244631769e0 * 3.e0
        bp[2, 2] = -3.05195205099107e0 * 3.e0
        bp[3, 0] = -5.25089420371162e0 * 4.e0
        bp[3, 1] = -4.30386791194303e0 * 4.e0
        bp[3, 2] = 5.16051899359931e0 * 4.e0
        bp[4, 0] = 5.37112827253230e0 * 5.e0
        bp[4, 1] = 38.5344528930499e0 * 5.e0
        bp[4, 2] = -7.76088601041257e0 * 5.e0
        bp[5, 0] = 34.4252230677698e0 * 6.e0
        bp[5, 1] = -26.9710769414608e0 * 6.e0
        bp[5, 2] = 15.6044623461691e0 * 6.e0
        bp[6, 0] = -50.8003365888685e0 * 7.e0
        bp[6, 1] = -23.6010990650801e0 * 7.e0
        bp[6, 2] = -4.23812936930675e0 * 7.e0

        self.ap = ap
        self.bp = bp

        self.deltasimplified = deltasimplified
        return

    # EQ A.9 ok!
    def PC_SAFT_d_T(self, T):
        d_T = np.zeros(self.ncomp)
        sigma = self.sigma
        epsilon_k = self.epsilon_k

        for i in range(self.ncomp):
            d_T[i] = sigma[i]*(1.0-0.12*np.exp(-3.*epsilon_k[i]/(T)))

        return d_T

    # EQ A.8 
    def PC_SAFT_csi(self, dens, T, x):
        d_T = self.PC_SAFT_d_T(T)
        rho = self.PC_SAFT_rho(dens)

        csi = np.zeros(4)

        for i in range(4):

            soma = 0
            for j in range(self.ncomp):
                soma = soma + x[j]*self.m[j]*d_T[j]**i

            csi[i] = soma*pi*rho/6

        return csi

    # EQ A.7 ok!
    def PC_SAFT_ghs(self, dens, T, x):
        csi = self.PC_SAFT_csi(dens, T, x)
        d_T = self.PC_SAFT_d_T(T)

        ghs = np.zeros([self.ncomp, self.ncomp])

        for i in range(self.ncomp):
            for j in range(self.ncomp):
                ghs[i, j] = 1/(1-csi[3]) + (d_T[i]*d_T[j]/(d_T[i]+d_T[j]))*3*csi[2]/(
                    1-csi[3])**2 + ((d_T[i]*d_T[j]/(d_T[i]+d_T[j]))**2)*2*csi[2]**2/(1-csi[3])**3

        return ghs

    # EQ A.26 ok!
    def PC_SAFT_Zhs(self, dens, T, x):
        csi = self.PC_SAFT_csi(dens, T, x)

        P1 = csi[3]/(1-csi[3])
        P2 = 3*csi[1]*csi[2]/(csi[0]*(1-csi[3])**2)
        P3 = (3*csi[2]**3 - csi[3]*csi[2]**3)/(csi[0]*(1-csi[3])**3)

        Zhs = P1+P2+P3

        return Zhs

    # EQ A.5 ok!
    def PC_SAFT_mmed(self, x):
        mmed = 0
        for i in range(self.ncomp):
            mmed += x[i]*self.m[i]

        return mmed

    # EQ A.27 ok!
    def PC_SAFT_rho_dghsd_drho(self, dens, T, x):

        csi = self.PC_SAFT_csi(dens, T, x)
        d_T = self.PC_SAFT_d_T(T)

        rho_dghsd_drho = np.zeros([self.ncomp, self.ncomp])

        for i in range(self.ncomp):
            for j in range(self.ncomp):

                rho_dghsd_drho[i, j] = csi[3]/(1-csi[3])**2 + (d_T[i]*d_T[j]/(d_T[i]+d_T[j]))*(3*csi[2]/(1-csi[3])**2 + 6*csi[2]*csi[3]/(
                    1-csi[3])**3) + (d_T[i]*d_T[j]/(d_T[i]+d_T[j]))**2*(4*csi[2]**2/(1-csi[3])**3 + 6*csi[2]**2*csi[3]/(1-csi[3])**4)

        return rho_dghsd_drho

    # EQ A.25 ok!
    def PC_SAFT_Zhc(self, dens, T, x):
        mmed = self.PC_SAFT_mmed(x)
        Zhs = self.PC_SAFT_Zhs(dens, T, x)
        ghs = self.PC_SAFT_ghs(dens, T, x)
        rho_dghsd_drho = self.PC_SAFT_rho_dghsd_drho(dens, T, x)

        soma = 0

        for i in range(self.ncomp):
            soma += x[i]*(self.m[i]-1)/ghs[i, i]*rho_dghsd_drho[i, i]

        Zhc = mmed*Zhs - soma

        return Zhc

    # EQ A.18 AND A.19  ok!
    def PC_SAFT_a_e_b(self, x):
        ap = self.ap
        bp = self.bp
        mmed = self.PC_SAFT_mmed(x)

        a = np.zeros(7)
        b = np.zeros(7)

        for i in range(7):

            a[i] = ap[i, 0] + (mmed-1)*ap[i, 1]/mmed + \
                (1-1/mmed)*(1-2/mmed)*ap[i, 2]
            b[i] = bp[i, 0] + (mmed-1)*bp[i, 1]/mmed + \
                (1-1/mmed)*(1-2/mmed)*bp[i, 2]

        return a, b

    # EQ A.11 ok! -> its not inverted in the paper
    def PC_SAFT_C1(self, dens, T, x):
        mmed = self.PC_SAFT_mmed(x)
        eta = self.PC_SAFT_csi(dens, T, x)[3]

        C1 = (1 + mmed*(8*eta-2*eta**2)/(1-eta)**4 + (1-mmed)*(20*eta -
              27*eta**2 + 12*eta**3 - 2*eta**4)/((1-eta)*(2-eta))**2)**-1

        return C1

    # EQ A.31 ok!
    def PC_SAFT_C2(self, dens, T, x):
        C1 = self.PC_SAFT_C1(dens, T, x)
        mmed = self.PC_SAFT_mmed(x)
        eta = self.PC_SAFT_csi(dens, T, x)[3]

        C2 = -C1**2*(mmed*(-4*eta**2 + 20*eta + 8)/(1-eta)**5 + (1-mmed)
                     * (2*eta**3+12*eta**2-48*eta+40)/((1-eta)*(2-eta))**3)

        return C2

    # A.16 and A.17 ok!
    def PC_SAFT_I1_e_I2(self, dens, T, x):
        a, b = self.PC_SAFT_a_e_b(x)
        eta = self.PC_SAFT_csi(dens, T, x)[3]

        I1 = 0
        I2 = 0

        for i in range(7):
            I1 += a[i]*eta**i
            I2 += b[i]*eta**i

        return I1, I2

    # EQ A.29 and A.30 ok!
    def PC_SAFT_detaI1_deta_e_detaI2_deta(self, dens, T, x):
        a, b = self.PC_SAFT_a_e_b(x)
        eta = self.PC_SAFT_csi(dens, T, x)[3]

        detaI1_deta = 0
        detaI2_deta = 0

        for i in range(7):
            detaI1_deta += a[i]*(i+1)*eta**i
            detaI2_deta += b[i]*(i+1)*eta**i

        return detaI1_deta, detaI2_deta

    # EQ A.14 ok!
    def PC_SAFT_MAT_sigma(self, x):
        sigma = self.sigma
        MAT_sigma = np.zeros([self.ncomp, self.ncomp])

        for i in range(self.ncomp):
            for j in range(self.ncomp):
                MAT_sigma[i, j] = (sigma[i]+sigma[j])/2

        return MAT_sigma

    # EQ A.15 ok!
    def PC_SAFT_MAT_epsilon_k(self, x):
        epsilon_k = self.epsilon_k
        kbi = self.kbi
        MAT_epsilon_k = np.zeros([self.ncomp, self.ncomp])

        for i in range(self.ncomp):
            for j in range(self.ncomp):
                MAT_epsilon_k[i, j] = (
                    epsilon_k[i]*epsilon_k[j])**(1/2)*(1-kbi[i, j])

        return MAT_epsilon_k

    # EQ A.12 and A.13 ok!
    def PC_SAFT_m_2esig_3_e_m_2e_2sig_3(self, T, x):
        m = self.m

        MAT_epsilon_k = self.PC_SAFT_MAT_epsilon_k(x)
        MAT_sigma = self.PC_SAFT_MAT_sigma(x)
        m_2esig_3 = 0
        m_2e_2sig_3 = 0

        for i in range(self.ncomp):
            for j in range(self.ncomp):
                m_2esig_3 += x[i]*x[j]*m[i]*m[j] * \
                    MAT_epsilon_k[i, j]/T*MAT_sigma[i, j]**3
                m_2e_2sig_3 += x[i]*x[j]*m[i]*m[j] * \
                    (MAT_epsilon_k[i, j]/T)**2*MAT_sigma[i, j]**3

        return m_2esig_3, m_2e_2sig_3

    # EQ A.26 ok!
    def PC_SAFT_Zdisp(self, dens, T, x):
        mmed = self.PC_SAFT_mmed(x)
        detaI1_deta, detaI2_deta = self.PC_SAFT_detaI1_deta_e_detaI2_deta(
            dens, T, x)
        C1 = self.PC_SAFT_C1(dens, T, x)
        C2 = self.PC_SAFT_C2(dens, T, x)
        eta = self.PC_SAFT_csi(dens, T, x)[3]
        _, I2 = self.PC_SAFT_I1_e_I2(dens, T, x)
        m_2esig_3, m_2e_2sig_3 = self.PC_SAFT_m_2esig_3_e_m_2e_2sig_3(T, x)
        rho = self.PC_SAFT_rho(dens)

        Zdisp = -2*pi*rho*detaI1_deta*m_2esig_3 - pi * \
            rho*mmed*(C1*detaI2_deta+C2*eta*I2)*m_2e_2sig_3

        return Zdisp

    # EQ A.24 ok!
    def PC_SAFT_Z(self, dens, T, x):
        Zhc = self.PC_SAFT_Zhc(dens, T, x)
        Zdisp = self.PC_SAFT_Zdisp(dens, T, x)
        if self.S is None:
            Z = 1 + Zhc + Zdisp

        else:
            da_ass_deta = self.PC_SAFT_da_ass_deta(dens,T,x)
            Zassoc = dens*da_ass_deta
            Z = 1 + Zhc + Zdisp + Zassoc
        return Z

    # EQ A.23 ok! -> there is no avogadro number in the paper, only volume convertion in intead
    def PC_SAFT_Pressure(self, dens, T, x):
        Z = self.PC_SAFT_Z(dens, T, x)
        P = Z*kb*T*dens*Navo
        return P

    # EQ A.21 ok!
    def PC_SAFT_rho(self, dens):
        rho = dens*Navo/1.0e30
        return rho

    # EQ -- ok!
    def PC_SAFT_dens(self, T, P, x, phase=None, opt=False, method=None,real=False):
        m = self.m
        d_T = self.PC_SAFT_d_T(T)
        soma = 0
        # for liquid
        etaguessL = 0.5
        # for gas
        etaguessV = 1e-10
        for i in range(self.ncomp):
            soma += x[i]*m[i]*d_T[i]**3

        densL0 = 6/pi*etaguessL/soma*1e30/Navo
        densV0 = 6/pi*etaguessV/soma*1e30/Navo

        def residuo(dens):
            densi, = dens
            # res0 = 1 - (self.PC_SAFT_Pressure(densi, T, x))/P
            res0 = ((self.PC_SAFT_Pressure(densi, T, x))/P) -1
            f = [res0]

            return f
        # aqui otimiza
        if opt is True:
            def fobjL(dens):
                f = ((P - self.PC_SAFT_Pressure(dens, T, x)))**2 - min(0,1/(6/pi*(pi/(3/np.sqrt(2)))/soma*1e30/Navo)- dens)
                return f

            def fobjV(dens):
                f = ((P - self.PC_SAFT_Pressure(dens, T, x)))**2
                return f
            
            
            if phase is None or phase == 'liq':
                ans = optimize.minimize(fobjL, densL0, method=method )
                densL_1 = ans["x"][0]
                if phase == 'liq':
                    return densL_1

            if phase is None or phase == 'vap':
                ans = optimize.minimize(fobjV, densV0, method=method)
                densV_1 = ans["x"][0]
                if phase == 'vap':
                    return densV_1

        # aqui usa o solver
        else:
            def residuo_log(dens_ad):  # escalar

                densi = dens_ad[0]
                pcalc = self.PC_SAFT_Pressure(densi, T, x)
                res0 = np.log(pcalc / P)
                f = [res0]

                return f
            
            if method is None:
                method = 'hybr'

            if phase is None or phase == 'liq':
                ans = optimize.root(residuo, [densL0, ], method=method)
                densL_1 = ans["x"][0]

                if phase == 'liq':
                    return densL_1



            if phase is None or phase == 'vap':
                ans = optimize.root(residuo_log, [densV0, ], method=method,tol=None)
                densV_1_ad = ans["x"][0]
                densV_1 = densV_1_ad
                if phase == 'vap':
                    return densV_1
            
        return densL_1, densV_1

        if phase is None or phase == 'vap':
            ans = optimize.root(residuo, [densV0, ])
            densV_1_ad = ans["x"][0]
            densV_1 = densV_1_ad
            if phase == 'vap':
                return densV_1
            
        return densL_1, densV_1


    
    # EQ A.6 ok!
    def PC_SAFT_a_hs(self, dens, T, x):
        csi = self.PC_SAFT_csi(dens, T, x)
        # print(csi)
        a_hs = (3*csi[1]*csi[2]/(1-csi[3]) + csi[2]**3/(csi[3]*(1-csi[3])
                ** 2) + (csi[2]**3/csi[3]**2 - csi[0])*np.log(1-csi[3]))/csi[0]

        return a_hs

    # EQ A.4 ok!
    def PC_SAFT_a_hc(self, dens, T, x):
        mmed = self.PC_SAFT_mmed(x)
        m = self.m
        ghs = self.PC_SAFT_ghs(dens, T, x)
        a_hs = self.PC_SAFT_a_hs(dens, T, x)

        soma = 0

        for i in range(self.ncomp):
            soma += -x[i]*(m[i]-1)*np.log(ghs[i, i])

        a_hc = mmed*a_hs + soma

        return a_hc

    # EQ A.10 ok!
    def PC_SAFT_a_disp(self, dens, T, x):
        I1, I2 = self.PC_SAFT_I1_e_I2(dens, T, x)
        m_2esig_3, m_2e_2sig_3 = self.PC_SAFT_m_2esig_3_e_m_2e_2sig_3(T, x)
        rho = self.PC_SAFT_rho(dens)
        mmed = self.PC_SAFT_mmed(x)
        C1 = self.PC_SAFT_C1(dens, T, x)

        a_disp = -2*pi*rho*I1*m_2esig_3 - pi*rho*mmed*C1*I2*m_2e_2sig_3
        return a_disp

    # EQ A.3 ok!
    def PC_SAFT_a_res(self, dens, T, x):
        if self.S is None:

            a_res = self.PC_SAFT_a_hc(dens, T, x) + \
                self.PC_SAFT_a_disp(dens, T, x)
        else:
            a_res = self.PC_SAFT_a_hc(dens, T, x) + self.PC_SAFT_a_disp(dens, T, x) + self.PC_SAFT_a_ass(dens, T, x)
        
        # import pcsaft
        # a_res = pcsaft.pcsaft_ares(T,dens,x,params)
        return a_res

    # EQ A.34 ok!
    def PC_SAFT_mat_dcsi_dxk(self, dens, T):
        m = self.m
        d_T = self.PC_SAFT_d_T(T)
        mat_dcsi_dxk = np.zeros([4, self.ncomp])
        rho = self.PC_SAFT_rho(dens)

        for k in range(self.ncomp):
            for i in range(4):
                mat_dcsi_dxk[i, k] = pi/6 * rho*m[k]*d_T[k]**i

        return mat_dcsi_dxk

    # EQ A.37 derivative confirmed ok!
    def PC_SAFT_dghs_dx(self, dens, T, x):
        mat_dcsi_dxk = self.PC_SAFT_mat_dcsi_dxk(dens, T)
        csi = self.PC_SAFT_csi(dens, T, x)

        d_T = self.PC_SAFT_d_T(T)

        dghs_dx = []
        for k in range(self.ncomp):
            dghs_dx.append(np.zeros([self.ncomp, self.ncomp]))

        for k in range(self.ncomp):
            P1 = mat_dcsi_dxk[3, k]/(1-csi[3])**2
            P2 = 3*mat_dcsi_dxk[2, k]/(1-csi[3])**2 + \
                6*(csi[2]*mat_dcsi_dxk[3, k])/(1-csi[3])**3
            P3 = 4*csi[2]*mat_dcsi_dxk[2, k] / \
                (1-csi[3])**3 + 6*csi[2]**2*mat_dcsi_dxk[3, k]/(1-csi[3])**4
            for i in range(self.ncomp):
                for j in range(self.ncomp):
                    dghs_dx[k][i, j] = P1 + (d_T[i]*d_T[j]/(d_T[i]+d_T[j])) * \
                        P2 + (d_T[i]*d_T[j]/(d_T[i]+d_T[j]))**2*P3

        return dghs_dx



    # EQ A.36 derivative confirmed ok!
    def PC_SAFT_da_hs_dx(self, dens, T, x):
        mat_dcsi_dxk = self.PC_SAFT_mat_dcsi_dxk(dens, T)
        a_hs = self.PC_SAFT_a_hs(dens, T, x)
        csi = self.PC_SAFT_csi(dens, T, x)
        da_hs_dx = np.zeros([self.ncomp])

        for i in range(self.ncomp):
            P1 = - mat_dcsi_dxk[0, i]/csi[0]*a_hs
            P2 = 3*(mat_dcsi_dxk[1, i]*csi[2] + csi[1]
                    * mat_dcsi_dxk[2, i])/(1-csi[3])
            P3 = 3*csi[1]*csi[2]*mat_dcsi_dxk[3, i]/(1-csi[3])**2
            P4 = 3*csi[2]**2*mat_dcsi_dxk[2, i]/(csi[3]*(1-csi[3])**2)
            P5 = csi[2]**3*mat_dcsi_dxk[3, i] * \
                (3*csi[3]-1)/(csi[3]**2*(1-csi[3])**3)
            P6 = ((3*csi[2]**2*mat_dcsi_dxk[2, i]*csi[3] - 2*csi[2]**3 *
                  mat_dcsi_dxk[3, i])/csi[3]**3 - mat_dcsi_dxk[0, i])*np.log(1-csi[3])
            P7 = (csi[0]-csi[2]**3/csi[3]**2)*mat_dcsi_dxk[3, i]/(1-csi[3])
            da_hs_dx[i] = P1 + (P2 + P3 + P4 + P5 + P6 + P7)/csi[0]

        return da_hs_dx

    # EQ A.35 ------------------- it's not correcly written in the paper
    def PC_SAFT_da_hc_dx(self, dens, T, x):
        m = self.m
        mmed = self.PC_SAFT_mmed(x)
        a_hs = self.PC_SAFT_a_hs(dens, T, x)
        ghs = self.PC_SAFT_ghs(dens, T, x)
        dghs_dx = self.PC_SAFT_dghs_dx(dens, T, x)
        da_hs_dx = self.PC_SAFT_da_hs_dx(dens, T, x)

        da_hc_dx = np.zeros(self.ncomp)
        somai = np.zeros(self.ncomp)

        for k in range(self.ncomp):
            for i in range(self.ncomp):
                somai[k] += x[i]*(m[i]-1)/ghs[i, i] * dghs_dx[k][i, i]

            da_hc_dx[k] = m[k]*a_hs + mmed*da_hs_dx[k] - \
                somai[k] - (m[k] - 1)*np.log(ghs[k, k])

        return da_hc_dx


        
    # EQ A.44 and A.45 derivative confirmed ok!
    def PC_SAFT_dai_dx_e_dbi_dx(self, x):
        m = self.m
        mmed = self.PC_SAFT_mmed(x)
        ap = self.ap
        bp = self.bp

        dai_dx = []
        dbi_dx = []

        for i in range(self.ncomp):
            dai_dx.append(np.zeros([7]))
            dbi_dx.append(np.zeros([7]))

        for k in range(self.ncomp):
            for i in range(7):
                dai_dx[k][i] = (m[k]/mmed**2)*ap[i, 1] + m[k] / \
                    (mmed**2)*(3-4/mmed)*ap[i, 2]
                dbi_dx[k][i] = (m[k]/mmed**2)*bp[i, 1] + m[k] / \
                    (mmed**2)*(3-4/mmed)*bp[i, 2]

        return dai_dx, dbi_dx


    # EQ A.42 and A.43 derivative confirmed ok!
    def PC_SAFT_dI1_dx_e_dI2_dx(self, dens, T, x):
        a, b = self.PC_SAFT_a_e_b(x)
        eta = self.PC_SAFT_csi(dens, T, x)[3]
        dai_dx, dbi_dx = self.PC_SAFT_dai_dx_e_dbi_dx(x)
        mat_dcsi_dxk = self.PC_SAFT_mat_dcsi_dxk(dens, T)
        dI1_dx = np.zeros(self.ncomp)
        dI2_dx = np.zeros(self.ncomp)

        for k in range(self.ncomp):
            for i in range(7):
                dI1_dx[k] += a[i]*i*mat_dcsi_dxk[3, k] * \
                    eta**(i-1) + dai_dx[k][i]*eta**i
                dI2_dx[k] += b[i]*i*mat_dcsi_dxk[3, k] * \
                    eta**(i-1) + dbi_dx[k][i]*eta**i

        return dI1_dx, dI2_dx

    # EQ A.41  derivative confirmed ok!
    def PC_SAFT_dC1_dx(self, dens, T, x):
        m = self.m
        eta = self.PC_SAFT_csi(dens, T, x)[3]
        mat_dcsi_dxk = self.PC_SAFT_mat_dcsi_dxk(dens, T)
        C1 = self.PC_SAFT_C1(dens, T, x)
        C2 = self.PC_SAFT_C2(dens, T, x)
        dC1_dx = np.zeros(self.ncomp)

        for k in range(self.ncomp):
            P1 = C2*mat_dcsi_dxk[3, k]
            P2 = m[k]*(8*eta-2*eta**2)/(1-eta)**4
            P3 = - m[k]*(20*eta-27*eta**2+12*eta**3 -
                         2*eta**4)/((1-eta)*(2-eta))**2
            dC1_dx[k] = P1 - C1**2*(P2 + P3)

        return dC1_dx

    # EQ A.41 derivative confirmed ok!
    def PC_SAFT_dC1_dx_num(self, dens, T, x):
        step = 1e-5

        dC1_dx = []

        for k in range(self.ncomp):
            xmais = x*1
            xmenos = x*1

            xmais[k] = xmais[k] + step
            xmenos[k] = xmenos[k] - step

            mmedmais = self.PC_SAFT_mmed(xmais)
            etamais = self.PC_SAFT_csi(dens, T, xmais)[3]

            mmedmenos = self.PC_SAFT_mmed(xmenos)
            etamenos = self.PC_SAFT_csi(dens, T, xmenos)[3]

            C1mais = (1 + mmedmais*(8*etamais-2*etamais**2)/(1-etamais)**4 + (1-mmedmais)*(20 *
                      etamais - 27*etamais**2 + 12*etamais**3 - 2*etamais**4)/((1-etamais)*(2-etamais))**2)**-1
            C1menos = (1 + mmedmenos*(8*etamenos-2*etamenos**2)/(1-etamenos)**4 + (1-mmedmenos)*(
                20*etamenos - 27*etamenos**2 + 12*etamenos**3 - 2*etamenos**4)/((1-etamenos)*(2-etamenos))**2)**-1

            dC1_dx.append((C1mais - C1menos)/(2*step))

        return dC1_dx

    # EQ A.39 and A.40 derivative confirmed ok!
    def PC_SAFT_dm_2esig_3_dx_e_dm_2e_2sig_3_dx(self, dens, T, x):
        m = self.m
        #epsilon_k = self.epsilon_k
        dm_2esig_3_dx = np.zeros(self.ncomp)
        dm_2e_2sig_3_dx = np.zeros(self.ncomp)
        MAT_epsilon_k = self.PC_SAFT_MAT_epsilon_k(x)
        MAT_sigma = self.PC_SAFT_MAT_sigma(x)

        for k in range(self.ncomp):
            for i in range(self.ncomp):
                dm_2esig_3_dx[k] += 2*m[k]*x[i]*m[i] * \
                    (MAT_epsilon_k[k, i]/T)*MAT_sigma[k, i]**3
                dm_2e_2sig_3_dx[k] += 2*m[k]*x[i]*m[i] * \
                    (MAT_epsilon_k[k, i]/T)**2*MAT_sigma[k, i]**3

        return dm_2esig_3_dx, dm_2e_2sig_3_dx


    # EQ A.38  derivative confirmed ok!
    def PC_SAFT_da_disp_dx(self, dens, T, x):
        m = self.m
        dI1_dx, dI2_dx = self.PC_SAFT_dI1_dx_e_dI2_dx(dens, T, x)
        dm_2esig_3_dx, dm_2e_2sig_3_dx = self.PC_SAFT_dm_2esig_3_dx_e_dm_2e_2sig_3_dx(dens, T, x)
        dC1_dx = self.PC_SAFT_dC1_dx(dens, T, x)
        C1 = self.PC_SAFT_C1(dens, T, x)
        I1, I2 = self.PC_SAFT_I1_e_I2(dens, T, x)
        da_disp_dx = np.zeros(self.ncomp)
        m_2esig_3, m_2e_2sig_3 = self.PC_SAFT_m_2esig_3_e_m_2e_2sig_3(T, x)
        mmed = self.PC_SAFT_mmed(x)
        rho = self.PC_SAFT_rho(dens)

        for k in range(self.ncomp):
            P1 = -2*pi*rho*(dI1_dx[k]*m_2esig_3 + I1*dm_2esig_3_dx[k])
            P2 = (m[k]*C1*I2 + mmed*dC1_dx[k]*I2+mmed*C1*dI2_dx[k])*m_2e_2sig_3
            P3 = mmed*C1*I2*dm_2e_2sig_3_dx[k]
            da_disp_dx[k] = P1 - pi*rho*(P2+P3)
        
        
        return da_disp_dx

    
    
    def PC_SAFT_da_ass_dx_num(self, dens, T, x):
        nsite = len(self.S)
        step = 1e-4
        da_ass_dx = []

        for k in range(self.ncomp):

            xmais = x*1
            xmenos = x*1
            
            xmais[k] = xmais[k] + step
            xmenos[k]= xmenos[k] - step
            
            X_A_mais = self.PC_SAFT_X_tan(dens, T, xmais)
            X_A_menos = self.PC_SAFT_X_tan(dens, T, xmenos)
            
            a_ass_mais = 0
            a_ass_menos = 0
            
            for i in range(self.ncomp):
                somamais=0
                somamenos=0
                for j in range(nsite):
                    somamais += (np.log(X_A_mais[j, i]) - X_A_mais[j, i]/2 )
                    somamenos += (np.log(X_A_menos[j, i]) - X_A_menos[j, i]/2 )
                    
                a_ass_mais += xmais[i]*(somamais+ sum(self.S[:,i])*0.5)
                a_ass_menos += xmenos[i]*(somamenos+ sum(self.S[:,i])*0.5)
            
            da_ass_dx.append((a_ass_mais-a_ass_menos)/(2*step))


        return da_ass_dx

    def PC_SAFT_dg_ij_drho(self, dens, T, x):
        m = self.m
        d_T = self.PC_SAFT_d_T(T)
        dg_ij_drho = np.zeros((self.ncomp,self.ncomp))
        csi = self.PC_SAFT_csi(dens, T, x)
        C = np.einsum('i,i,i->', x,m,d_T**2)
        D = np.einsum('i,i,i->', x,m,d_T**3)
        for i in range(self.ncomp):
            for j in range(self.ncomp):
            
                dg_ij_drho[i,j] =  pi/6.*((D)/(1-csi[3])/(1-csi[3]) + 3*d_T[i]*d_T[j]/
                        (d_T[i]+d_T[j])*(C/(1-csi[3])/(1-csi[3])+2*(D)*csi[2]/(1-csi[3])**3) 
                        + 2*((d_T[i]*d_T[j]/(d_T[i]+d_T[j]))**2)*(2*C*csi[2]/(1-csi[3])**3
                        +3*(D)*csi[2]*csi[2]
                        /(1-csi[3])**4))
                
        
        return dg_ij_drho
    
    #confirmado pois ddelta_drho_i * rho = ddelta_dx_i !
    def PC_SAFT_ddelta_dx_k(self, dens, T, x):
        nsite = len(self.S)
        MAT_sigma = self.PC_SAFT_MAT_sigma(x)
        eAiBj_k = self.PC_SAFT_eAiBj_k()
        kAiBj_k = self.PC_SAFT_kAiBj_k(x)
        ddelta_dx_k = np.zeros((nsite, self.ncomp, nsite, self.ncomp,self.ncomp))
        
        dg_ij_dx_k = self.PC_SAFT_dg_ij_dx_k(dens, T, x)
        
        for m in range(self.ncomp):
            for i in range(self.ncomp):
                for j in range(nsite):
                    for k in range(self.ncomp):
                        for l in range(nsite):
                            ddelta_dx_k[j,i,l,k,m] = MAT_sigma[k,i]**3*dg_ij_dx_k[k,i,m]*(np.exp(eAiBj_k[k,i]/T) - 1 )*kAiBj_k[k,i]
        
        return ddelta_dx_k
    
    def PC_SAFT_ddelta_dx_k_num(self, dens, T, x):
        step = 1e-5
        dens= dens+step
        nsite = len(self.S)
        eAiBj_k = self.PC_SAFT_eAiBj_k()

        delta_mais = np.zeros((nsite, self.ncomp, nsite, self.ncomp))
        delta_menos = np.zeros((nsite, self.ncomp, nsite, self.ncomp))
        ddelta_dx_k = []
        
        for m in range(self.ncomp):
            xmais = x*1
            xmenos = x*1

            xmais[m] = xmais[m] + step
            xmenos[m] = xmenos[m] - step            
            
            MAT_sigma_mais = self.PC_SAFT_MAT_sigma(xmais)
            MAT_sigma_menos = self.PC_SAFT_MAT_sigma(xmenos)
            
            kAiBj_k_mais = self.PC_SAFT_kAiBj_k(xmais)
            kAiBj_k_menos = self.PC_SAFT_kAiBj_k(xmenos)
            
            ghs_mais = self.PC_SAFT_ghs(dens, T, xmais)
            ghs_menos = self.PC_SAFT_ghs(dens, T,xmenos)
            
            for i in range(self.ncomp):
                for j in range(nsite):
                    for k in range(self.ncomp):
                        for l in range(nsite):
                            delta_mais[j,i,l,k] = MAT_sigma_mais[k,i]**3*ghs_mais[k,i]*(np.exp(eAiBj_k[k,i]/T) - 1 )*kAiBj_k_mais[k,i]
                            delta_menos[j,i,l,k] = MAT_sigma_menos[k,i]**3*ghs_menos[k,i]*(np.exp(eAiBj_k[k,i]/T) - 1 )*kAiBj_k_menos[k,i]
        
            ddelta_dx_k.append((delta_mais - delta_menos)/(2*step))
        return ddelta_dx_k
    
    

    def PC_SAFT_ddelta_dn_k_num(self, dens, T, x):
        step = 1e-4
        nsite = len(self.S)
        eAiBj_k = self.PC_SAFT_eAiBj_k()
        delta_mais=np.zeros((nsite, self.ncomp, nsite, self.ncomp))
        delta_menos = np.zeros((nsite, self.ncomp, nsite, self.ncomp))
        ddelta_dn_k = np.zeros((nsite, self.ncomp, nsite, self.ncomp,self.ncomp))
        for m in range(self.ncomp):
            
            xmais = x*1
            xmenos = x*1

            xmais[m] = xmais[m] + step
            xmenos[m] = xmenos[m] - step
         
            nmais = 1 + step
            nmenos = 1 - step
            
            xmais = xmais[:]/np.sum(xmais[:])
            xmenos = xmenos[:]/np.sum(xmenos[:])
            
            MAT_sigma_mais = self.PC_SAFT_MAT_sigma(xmais)
            MAT_sigma_menos = self.PC_SAFT_MAT_sigma(xmenos)
            
            kAiBj_k_mais = self.PC_SAFT_kAiBj_k(xmais)
            kAiBj_k_menos = self.PC_SAFT_kAiBj_k(xmenos)
            
            ghs_mais = self.PC_SAFT_ghs(dens, T, xmais)
            ghs_menos = self.PC_SAFT_ghs(dens, T, xmenos)
            
            for i in range(self.ncomp):
                for j in range(nsite):
                    for k in range(self.ncomp):
                        for l in range(nsite):
                            if j != l:
                                delta_mais[j,i,l,k] = MAT_sigma_mais[k,i]**3*ghs_mais[k,i]*(np.exp(eAiBj_k[k,i]/T) - 1 )*kAiBj_k_mais[k,i]
                                delta_menos[j,i,l,k] = MAT_sigma_menos[k,i]**3*ghs_menos[k,i]*(np.exp(eAiBj_k[k,i]/T) - 1 )*kAiBj_k_menos[k,i]
        
            ddelta_dn_k[m] = (nmais*delta_mais - nmenos*delta_menos)/(2*step)
        return ddelta_dn_k

    def PC_SAFT_mu_assoc_kT(self, dens, T, x):
        X_A = self.PC_SAFT_X_tan(dens, T, x)
        mu_assoc_kT = np.zeros(self.ncomp)
        rho = self.PC_SAFT_rho(dens)
        dX_A_drho_k = self.PC_SAFT_dX_A_dx_k_num(dens, T, x)/rho
        
        sum1 = np.einsum('ji->i', np.log(X_A) -X_A*0.5 +0.5*self.S) 
        sum3 = np.einsum('i,jim,ji->m', rho*x,dX_A_drho_k,(1/X_A-0.5))
        
        mu_assoc_kT  = sum1 + sum3
        
        return mu_assoc_kT
    
    # Essa EQ nao tem no paper ok!
    def PC_SAFT_da_res_dx(self, dens, T, x):
        da_hc_dx = self.PC_SAFT_da_hc_dx(dens, T, x)
        da_disp_dx = self.PC_SAFT_da_disp_dx(dens, T, x)

        if self.S is None:
            da_res_dx = da_hc_dx + da_disp_dx
        else:
            da_ass_dx = self.PC_SAFT_da_ass_dx_num(dens, T, x)
            da_res_dx = da_hc_dx + da_disp_dx + da_ass_dx
        return da_res_dx

    # EQ A.33 ok!
    def PC_SAFT_mu_res_kT(self, dens, T, x):
        mu_res_kT = np.zeros(self.ncomp)
        mu_hc = np.zeros(self.ncomp)
        mu_disp  = np.zeros(self.ncomp)
        mu_assoc_kT = self.PC_SAFT_mu_assoc_kT( dens, T, x)
        Zhc = self.PC_SAFT_Zhc(dens, T, x)
        Zdisp = self.PC_SAFT_Zdisp(dens, T, x)
        da_hc_dx = self.PC_SAFT_da_hc_dx(dens, T, x)
        da_disp_dx = self.PC_SAFT_da_disp_dx(dens, T, x)
        ares_hc = self.PC_SAFT_a_hc(dens, T, x) 
        ares_disp = self.PC_SAFT_a_disp(dens, T, x)
        somahc = np.einsum('j,j->', x,da_hc_dx)
        somadisp = np.einsum('j,j->', x,da_disp_dx)

        
        for i in range(self.ncomp):
            mu_hc[i] = ares_hc + Zhc + da_hc_dx[i] - somahc
            mu_disp[i] = ares_disp + Zdisp + da_disp_dx[i] - somadisp
            mu_res_kT[i] = mu_hc[i] + mu_disp[i] + mu_assoc_kT[i]
            
        return mu_res_kT

    # EQ A.32 ok!
    def PC_SAFT_phi(self, dens, T, x):
        mu = self.PC_SAFT_mu_res_kT(dens, T, x)
        Z = self.PC_SAFT_Z(dens, T, x)
        lnphi = np.zeros(self.ncomp)

        for i in range(self.ncomp):
            lnphi[i] = mu[i] - np.log(Z)

        phi = np.exp(lnphi)
        return phi

    def delta(self):
        delta = []
        for k in range(self.ncomp):
            delta.append(np.zeros([self.nsite, self.nsite]))
            
    def PC_SAFT_eAiBj_k(self):
        eAB_k = self.eAB_k
        eAiBj_k = np.zeros((self.ncomp, self.ncomp))

        for i in range(self.ncomp):
            for j in range(self.ncomp):
                eAiBj_k[i, j] = (eAB_k[i]+eAB_k[j])/2
                
        # print(eAiBj_k)
        return eAiBj_k

    def PC_SAFT_kAiBj_k(self, x):
        kAB_k = self.kAB_k
        kAiBj_k = np.zeros((self.ncomp, self.ncomp))
        MAT_sigma = self.PC_SAFT_MAT_sigma(x)
        for i in range(self.ncomp):
            for j in range(self.ncomp):
                kAiBj_k[i, j] = np.sqrt(kAB_k[i]*kAB_k[j])*(np.sqrt(MAT_sigma[i, i]*MAT_sigma[j, j])/((MAT_sigma[i, i] + MAT_sigma[j, j])/2))**3 #*(1-self.kbi[i,j])
        # print(kAiBj_k)
        return kAiBj_k

    def PC_SAFT_delt(self, dens, T, x):
        nsite = len(self.S)
        MAT_sigma = self.PC_SAFT_MAT_sigma(x)
        eAiBj_k = self.PC_SAFT_eAiBj_k()
        kAiBj_k = self.PC_SAFT_kAiBj_k(x)
        ghs = self.PC_SAFT_ghs(dens, T, x)
        delt = np.zeros((nsite, self.ncomp, nsite, self.ncomp))
        d_T = self.PC_SAFT_d_T(T)
        
        if self.deltasimplified == False:
            for i in range(self.ncomp):
                for j in range(nsite):
                    for k in range(self.ncomp):
                        for l in range(nsite):
                            if j != l:
                                delt[j,i,l,k] = (d_T[i]*d_T[k]/(d_T[i]+d_T[k]))**3*ghs[i,k]*(np.exp(eAiBj_k[i,k]/T) - 1 )*kAiBj_k[i,k]
        else:
            for i in range(self.ncomp):
                for j in range(nsite):
                    for k in range(self.ncomp):
                        for l in range(nsite):
                            if j != l:
                                delt[j,i,l,k] = MAT_sigma[k,i]**3*ghs[k,i]*(np.exp(eAiBj_k[k,i]/T) - 1 )*kAiBj_k[k,i]
        # print(delt)
        return delt

    def PC_SAFT_X_tan(self, dens, T, x):
        delta = self.PC_SAFT_delt(dens, T, x)
        S = self.S
        nsite = len(S)
        ncomp = self.ncomp
        rho = self.PC_SAFT_rho(dens)
        X_A = np.ones([nsite, ncomp])*0.5
        res = 1.
        it = 0
        itMAX = 3000
        while (res > 1e-9 and it < itMAX):
            it += 1
            X_A_old = X_A*1.
            sum1 = np.einsum('k,lk,lk,lkji-> ji', x,S,X_A_old,delta)
            X_A = 1./(1. + rho*sum1)
            dif = np.abs((X_A-X_A_old)/X_A)
            res = np.max(dif)
        if it == itMAX:
            X_A = X_A*np.nan
        return X_A

    #confirmei com o do zach fazendo dX_A_dx_k/rho = dX_A_drho_k
    def PC_SAFT_dX_A_dx_k_num(self, dens, T, x):
        nsite = len(self.S)
        step = 1e-8
        dX_A_dx_k = np.ones((nsite, self.ncomp, self.ncomp))


        for k in range(self.ncomp):

            xmais = x*1
            xmenos = x*1
            
            xmais[k] = xmais[k] + step
            xmenos[k]= xmenos[k] - step
            
            X_A_mais = self.PC_SAFT_X_tan(dens, T, xmais)
            X_A_menos = self.PC_SAFT_X_tan(dens, T, xmenos)
            
            dX_A_dx_k[:,:,k]=((X_A_mais-X_A_menos)/(2*step))
        
        return dX_A_dx_k


    def PC_SAFT_a_ass(self, dens, T, x):
        nsite = len(self.S)
        X_A = self.PC_SAFT_X_tan(dens, T, x)
        a_ass = 0
        s1=0

        for i in range(self.ncomp):
            s1=0
            for j in range(nsite):
                s1 += (np.log(X_A[j, i]) - X_A[j, i]/2 )
            a_ass += x[i]*(s1+ sum(self.S[:,i])*0.5)
        
        return a_ass

    def PC_SAFT_da_ass_deta(self, dens, T, x):
        def f(dens,T,x):
            a_ass = self.PC_SAFT_a_ass(dens, T, x)
            return a_ass
        da_ass_deta = nd.Derivative(f,n=1)
        
        return da_ass_deta(dens,T,x)
    
    
    def PC_SAFT_da_res_ddens(self,dens,T,x):
        def f(dens,T,x):
            a_res = self.PC_SAFT_a_res(dens, T, x)
            return a_res
        da_res_deta = nd.Derivative(f,n=1)
        
        return da_res_deta(dens,T,x)
    
    
    

    def PC_SAFT_Psat(self, T, guessP):
        x = np.array([1])
        dens_L0, dens_V0 = self.PC_SAFT_dens(T, guessP, x)

        def residuo(var):
            Psat = var[0]
            dens_L = var[1]
            dens_V = var[2]

            Pl = self.PC_SAFT_Pressure(dens_L, T, x)
            Pv = self.PC_SAFT_Pressure(dens_V, T, x)
            phiL = self.PC_SAFT_phi(dens_L, T, x)
            phiV = self.PC_SAFT_phi(dens_V, T, x)

            res1 = (1-phiL/phiV)**2
            res2 = ((Pl-Psat)/Pl)**2
            res3 = ((Pv-Psat)/Pv)**2

            f = res1 + res2 + res3
            return f

        ans = optimize.minimize(
            residuo, [guessP, dens_L0, dens_V0], method='Nelder-Mead')
        Psat = ans["x"][0]
        dens_L = ans["x"][1]
        dens_V = ans["x"][2]

        return Psat, dens_L, dens_V

    def PC_SAFT_Psat2(self, T, iguess_P):  # ,index):
        RES = 1
        TOL = 1e-7
        MAX = 100
        P = iguess_P
        i = 0
        while(RES > TOL and i < MAX):
            x = np.array([1.])
            dens_L, dens_V = self.PC_SAFT_dens(T, P, x)
            if np.abs(dens_L-dens_V) < 1e-9:
                print('solução trivial')
                return np.nan, -1
            phiL = self.PC_SAFT_phi(dens_L, T, x)
            phiV = self.PC_SAFT_phi(dens_V, T, x)
            P = P*(phiL/phiV)
            RES = np.abs(phiL/phiV-1.)
            i = i+10
        return P[0]

    # kg/m³
    def PC_SAFT_massdens(self, T, P, x, phase=None, method=None, opt=False):

        M = self.M
        if phase is None:
            densl, densv = self.PC_SAFT_dens(T, P, x, method=method, opt=opt)
            massdens = np.zeros(2)
            for i in range(self.ncomp):
                massdens[0] += x[i]*M[i]*densl/1e+3
                massdens[1] += x[i]*M[i]*densv/1e+3
        else:
            dens = self.PC_SAFT_dens(
                T, P, x, phase=phase, method=method, opt=opt)
            massdens = 0
            for i in range(self.ncomp):
                massdens += x[i]*M[i]*dens/1e+3

        return massdens

    def PC_SAFT_gamma_w(self, dens, T, x):
        da_res_dx = self.PC_SAFT_da_res_dx(dens, T, x)
        gammaw = np.exp(da_res_dx)

        return gammaw

    def PC_SAFT_da_dT_num(self, dens, T, x):
        step = 1e-5

        T_mais = T + step
        T_menos = T - step

        a_res_mais = self.PC_SAFT_a_res(dens, T_mais, x)
        a_res_mais_mais = self.PC_SAFT_a_res(dens, T_mais+step, x)
        a_res_menos = self.PC_SAFT_a_res(dens, T_menos, x)
        a_res_menos_menos = self.PC_SAFT_a_res(dens, T_menos-step, x)
        
        da_dT = (-a_res_mais_mais +8*a_res_mais - 8*a_res_menos+a_res_menos_menos)/(12*step)

        return da_dT
    
    def PC_SAFT_H_res_RT(self, dens, T, x):
        da_dT = self.PC_SAFT_da_dT_num(dens, T, x)
        Z = self.PC_SAFT_Z(dens, T, x)

        H_res_RT = -T*da_dT + (Z-1)

        return H_res_RT

    def PC_SAFT_S_res_R(self, dens, T, x):
        a_res = self.PC_SAFT_a_res(dens, T, x)
        da_dT = self.PC_SAFT_da_dT_num(dens, T, x)
        Z = self.PC_SAFT_Z(dens, T, x)

        S_res_R = -T*(da_dT) - a_res + np.log(Z)

        return S_res_R

    def PC_SAFT_G_res(self, dens, T, x):
        a_res = self.PC_SAFT_a_res(dens, T, x)
        Z = self.PC_SAFT_Z(dens, T, x)
        G_res_RT = (a_res + (Z - 1) - np.log(Z))*kb*Navo*T

        return G_res_RT

    def PC_SAFT_ddens_dT(self, T, P, x):
        step = 1e-5

        T_mais = T + step
        T_menos = T - step

        dens_mais = self.PC_SAFT_massdens(T_mais, P, x, phase='vap', method=None)
        dens_menos = self.PC_SAFT_massdens(T_menos, P, x, phase='vap', method=None)

        ddens_dT = (dens_mais - dens_menos)/(2*step)

        return ddens_dT

    def PC_SAFT_kappat(self, T, P, x, phase):
        opt=False
        method=None
        dens = self.PC_SAFT_dens(T, P, x,phase)
        ddens_dP = self.PC_SAFT_ddens( T, P, x, phase, opt, method)
        kappat = ddens_dP /dens

        return kappat

    def PC_SAFT_alphap(self, T, P, x):
        ddens_dT = self.PC_SAFT_ddens_dT(T, P, x)
        dens = self.PC_SAFT_massdens(T, P, x, phase='vap')
        alphap = -ddens_dT/dens

        return alphap

    def PC_SAFT_Cp_res(self,T,P,x,phase):
        step = 1e-4
        T_mais = T+step
        T_menos = T-step
        dens_mais = self.PC_SAFT_dens(T_mais, P, x,phase=phase)
        dens_menos = self.PC_SAFT_dens(T_menos, P, x,phase=phase)
        H_res_mais  = self.PC_SAFT_H_res_RT(dens_mais, T_mais, x)*kb*Navo*T
        H_res_menos  = self.PC_SAFT_H_res_RT(dens_menos, T_menos, x)*kb*Navo*T
        
        Cp = (H_res_mais-H_res_menos)/(2*step) 
        
        return Cp
    
    def PC_SAFT_Cv_res(self,T,P,x,phase):
        step = 1e-4
        T_mais = T+step
        T_menos = T-step
        dens_mais = self.PC_SAFT_dens(T_mais, P, x,phase=phase)
        dens_menos = self.PC_SAFT_dens(T_menos, P, x,phase=phase)
        dens = self.PC_SAFT_dens(T, P, x,phase=phase)
        H_res_mais  = self.PC_SAFT_H_res_RT(dens, T_mais, x)*kb*Navo*T 
        H_res_menos  = self.PC_SAFT_H_res_RT(dens, T_menos, x)*kb*Navo*T 
        
        dH_dt_res = (H_res_mais-H_res_menos)/(2*step)
        
        P_mais = self.PC_SAFT_Pressure(dens, T_mais, x)
        P_menos = self.PC_SAFT_Pressure(dens, T_menos, x)
        
        dP_dT = (P_mais-P_menos)/(2*step)
        Cv = dH_dt_res + dP_dT/dens 

        return Cv

    def PC_SAFT_ddens(self, T, P, x, phase, opt, method):
        def f(P,T,x,phase,opt,method):
            dens = self.PC_SAFT_dens(T, P, x, phase, opt, method)
            return dens
        ddens_dP = nd.Derivative(f)
        
        return ddens_dP( P,T, x,phase,opt,method)#,d2dens_dP(self, T,P, x,phase,opt,method)
    
    def PC_SAFT_da_assoc_ddens(self,dens,T,x):
        nsite = len(self.S)
        dX_tan_ddens = self.PC_SAFT_dX_tan_ddens(dens, T, x)
        X_A = self.PC_SAFT_X_tan(dens, T, x)
        da_assoc_ddens = 0

        for i in range(self.ncomp):
            for j in range(nsite):
                da_assoc_ddens += x[i]*(1/X_A[j, i] - 1/2)*dX_tan_ddens[j,i]
        return da_assoc_ddens
    
    
    def PC_SAFT_Zassoc(self,dens,T,x):
        mu_assoc_kT = self.PC_SAFT_mu_assoc_kT(dens,T,x)
        a_ass = self.PC_SAFT_a_ass(dens,T,x)
        Zassoc = 0
        for i in range(self.ncomp):
            Zassoc +=  x[i]*mu_assoc_kT[i]
        Zassoc += a_ass
        # print('Z assoc',Zassoc)
        return Zassoc
    
    def PC_SAFT_dahs_dv(self, dens, T, x):
        mmed = self.PC_SAFT_mmed(x)
        eta = self.PC_SAFT_csi(dens, T, x)[3]
        dahsdv = -dens*eta*(mmed*(4-2*eta)/(1-eta)**3 -mmed*(5-2*eta)/(1-eta)/(2-eta) +(5-2*eta)/(1-eta)/(2-eta))
        return dahsdv
    
    def PC_SAFT_dadisp_dv(self, dens, T, x):
        I1, I2 = self.PC_SAFT_I1_e_I2(dens, T, x)
        eta = self.PC_SAFT_csi(dens, T, x)[3]
        detaI1_deta, detaI2_deta = self.PC_SAFT_detaI1_deta_e_detaI2_deta(dens, T, x)
        mmed = self.PC_SAFT_mmed(x)
        C1 = self.PC_SAFT_C1(dens, T, x)
        m_2esig_3,m_2e_2sig_3= self.PC_SAFT_m_2esig_3_e_m_2e_2sig_3(T, x)
        dC1dv = self.PC_SAFT_dC1_dv(dens, T, x)
        adisp = self.PC_SAFT_a_disp(dens, T, x)
        Fv = -dens*eta*(2*pi*detaI1_deta/eta*m_2esig_3 + pi*C1*mmed*detaI2_deta/eta*m_2e_2sig_3 + pi*mmed*dC1dv*I2*m_2e_2sig_3)
        dadispdv = dens*(Fv-adisp/(kb*Navo*T))
        return dadispdv
    
    def PC_SAFT_dC1_dv(self, dens, T, x):
        def f(dens, T, x):
            C1 = self.PC_SAFT_C1(dens, T, x)
            return C1
        dC1dv = nd.Derivative(f,n=1)
        
        return dC1dv(dens,T,x)
    
    def PC_SAFT_daass_dv(self, dens, T, x):
        S = self.S
        nsite = len(S)
        dlggdv = self.PC_SAFT_dlgghs_ddens( dens, T, x)
        X_A = self.PC_SAFT_X_tan(dens, T, x)
        dQdv = 0
        for i in range(self.ncomp):
            for j in range(nsite):
                dQdv += 0.5*dens*(1-1/dens*dlggdv)*x[i]*(1-X_A[j,i])
        daassdv = kb*Navo*dQdv
        
        return daassdv
    
    def PC_SAFT_dlgghs_ddens(self, dens, T, x):
        def f(dens, T, x):
            lgghs = np.log(self.PC_SAFT_ghs(dens, T, x))
            return lgghs
        dlgghs_ddens = nd.Derivative(f,n=1)
        return dlgghs_ddens(dens,T,x)
            