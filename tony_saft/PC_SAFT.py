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
            da_ass_deta = self.PC_SAFT_da_ass_deta(dens, T, x)
            Z = 1 + Zhc + Zdisp + dens*da_ass_deta
            # print(da_ass_deta,eta)
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
            res0 = 1 - (self.PC_SAFT_Pressure(densi, T, x))/P
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
                ans = optimize.minimize(fobjL, densL0, method=method)
                densL_1 = ans["x"][0]
                if phase == 'liq':
                    print('aaaa')
                    return densL_1

            if phase is None or phase == 'vap':
                ans = optimize.minimize(fobjV, densV0, method=method)
                densV_1 = ans["x"][0]
                if phase == 'vap':
                    return densV_1

        # aqui usa o solver
        else:
            if method is None:
                method = 'hybr'

            if phase is None or phase == 'liq':
                ans = optimize.root(residuo, [densL0, ], method=method)
                densL_1 = ans["x"][0]

                if phase == 'liq':
                    return densL_1

            def residuo_log(dens_ad):  # escalar

                densi = dens_ad[0]
                pcalc = self.PC_SAFT_Pressure(densi, T, x)
                res0 = np.log(pcalc / P)
                f = [res0]

                return f

            if phase is None or phase == 'vap':
                ans = optimize.root(residuo_log, [densV0, ], method=method,tol=None)
                densV_1_ad = ans["x"][0]
                densV_1 = densV_1_ad
                if phase == 'vap':
                    return densV_1
            
        if real is True:
            Gl_res = self.PC_SAFT_G_res(densL_1, T, x)
            Gv_res = self.PC_SAFT_G_res(densV_1, T, x)
            if Gl_res < Gv_res:
                return densL_1
            if Gl_res > Gv_res:
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
            a_res = self.PC_SAFT_a_hc(
                dens, T, x) + self.PC_SAFT_a_disp(dens, T, x) + self.PC_SAFT_a_ass(dens, T, x)

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
            P3 = - m[k]*(20*eta-27**eta**2+12*eta**3 -
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
        dm_2esig_3_dx, dm_2e_2sig_3_dx = self.PC_SAFT_dm_2esig_3_dx_e_dm_2e_2sig_3_dx(
            dens, T, x)
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
        step = 1e-5
        da_ass_dx = []

        for k in range(self.ncomp):

            xmais = x*1
            xmenos = x*1

            xmais[k] = xmais[k] + step
            xmenos[k] = xmenos[k] - step

            a_ass_mais = self.PC_SAFT_a_ass(dens, T, xmais)
            a_ass_menos = self.PC_SAFT_a_ass(dens, T, xmenos)

            da_ass_dx.append((a_ass_mais-a_ass_menos)/(2*step))

        return da_ass_dx

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
        a_res = self.PC_SAFT_a_res(dens, T, x)
        Z = self.PC_SAFT_Z(dens, T, x)
        da_res_dx = self.PC_SAFT_da_res_dx(dens, T, x)
        mu_res_kT = np.zeros(self.ncomp)
        somaj = np.zeros(self.ncomp)

        for k in range(self.ncomp):
            # for j in [x for x in range(self.ncomp) if x != k]:
            for j in range(self.ncomp):
                somaj[k] += da_res_dx[j]*x[j]

            mu_res_kT[k] = a_res + (Z-1) + da_res_dx[k] - somaj[k]

        return mu_res_kT

    # EQ A.32 ok!
    def PC_SAFT_phi(self, dens, T, x):
        #dens= self.PC_SAFT_dens(self,T,P,x)
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
        return eAiBj_k

    def PC_SAFT_kAiBj_k(self, x):
        kAB_k = self.kAB_k
        kAiBj_k = np.zeros((self.ncomp, self.ncomp))
        MAT_sigma = self.PC_SAFT_MAT_sigma(x)
        for i in range(self.ncomp):
            for j in range(self.ncomp):
                kAiBj_k[i, j] = np.sqrt(kAB_k[i]*kAB_k[j])*(np.sqrt(
                    MAT_sigma[i, i]*MAT_sigma[j, j])/((MAT_sigma[i, i] + MAT_sigma[j, j])/2))**3
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
                                delt[j,i,l,k] = pi/6*MAT_sigma[i,k]**3*ghs[i,k]*(np.exp(eAiBj_k[i,k]/T) - 1 )*kAiBj_k[i,k]
        return delt

    def PC_SAFT_X_tan(self, dens, T, x):
        delta = self.PC_SAFT_delt(dens, T, x)
        S = self.S
        nsite = len(S)
        ncomp = self.ncomp
        rho = self.PC_SAFT_rho(dens)
    # Sji is the number of association sites of type j in molecule of componente i
    # example molecule = AABCC, then Sji = 2,1,2
        X_A = np.zeros([nsite, ncomp])
        dif = np.zeros([nsite, ncomp])
        X_A[:, :] = .5
        res = 1.
        it = 0
        itMAX = 1000
        # Iterations with tolerance of 10-9
        while (res > 1e-9 and it < itMAX):
            it += 1
            X_A_old = X_A*1.
            for i in range(ncomp):
                for j in range(nsite):
                    sum1 = 0
                    for k in range(ncomp):
                        sum2 = 0
                        for l in range(nsite):
                            sum2 += S[l, k]*X_A_old[l, k]*delta[l, k, j, i]
                        sum1 += x[k]*sum2
                    X_A[j, i] = 1./(1. + rho*sum1)
                    dif[j, i] = np.abs((X_A[j, i]-X_A_old[j, i])/X_A[j, i])
            res = np.max(dif)
        if it == itMAX:
            print('too many steps in X_tan')
            X_A = X_A*np.nan
    #     print('xtan=',X_A.flatten())
        return X_A

    def PC_SAFT_a_ass(self, dens, T, x):
        nsite = len(self.S)
        X_A = self.PC_SAFT_X_tan(dens, T, x)
        a_ass = 0

        for i in range(self.ncomp):
            for j in range(nsite):
                a_ass += x[i]*(np.log(X_A[j, i]) - X_A[j, i]/2 + 0.5*self.S[j, i])
        return a_ass

    def PC_SAFT_da_ass_deta(self, dens, T, x):
        step = 1e-5

        dens_mais = dens + step
        dens_menos = dens - step

        a_ass_mais = self.PC_SAFT_a_ass(dens_mais, T, x)
        a_ass_menos = self.PC_SAFT_a_ass(dens_menos, T, x)

        da_ass_deta = (a_ass_mais - a_ass_menos)/(2*step)
        def f(P,T,x,phase,opt,method):
            dens = self.PC_SAFT_dens(T, P, x, phase, opt, method)
            return dens
        
        return da_ass_deta

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
        a_res_menos = self.PC_SAFT_a_res(dens, T_menos, x)

        da_dT = (a_res_mais - a_res_menos)/(2*step)

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

    def PC_SAFT_Cp(self,T,P,x,phase):
        step = 1e-4
        T_mais = T+step
        T_menos = T-step
        dens_mais = self.PC_SAFT_dens(T_mais, P, x,phase=phase)
        dens_menos = self.PC_SAFT_dens(T_menos, P, x,phase=phase)
        H_res_mais  = self.PC_SAFT_H_res_RT(dens_mais, T_mais, x)*kb*Navo*T
        H_res_menos  = self.PC_SAFT_H_res_RT(dens_menos, T_menos, x)*kb*Navo*T
        
        Cp = (H_res_mais-H_res_menos)/(2*step) + kb*Navo*(5.457+ 1.045*1e-3*T -1.157*1e+5/(T**2))
        
        return Cp
    
    def PC_SAFT_Cv(self,T,P,x,phase):
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
        Cv = dH_dt_res + dP_dT/dens + kb*Navo*(5.457+ 1.045*1e-3*T -1.157*1e+5/(T**2)) - 2*kb*Navo

        return Cv

    def PC_SAFT_ddens(self, T, P, x, phase, opt, method):
        def f(P,T,x,phase,opt,method):
            dens = self.PC_SAFT_dens(T, P, x, phase, opt, method)
            return dens
        ddens_dP = nd.Derivative(f)
        
        return ddens_dP( P,T, x,phase,opt,method)#,d2dens_dP(self, T,P, x,phase,opt,method)
        
    
    def PC_SAFT_triP(self,T, P, x):
        def raiz(var):
            T = var[0]
            P = var[1]
            r = self.PC_SAFT_ddens(T, P, x, opt=False, method=None, real = True)
            return r 
        ans = optimize.root(raiz, [T,P])
        return ans["x"]
            