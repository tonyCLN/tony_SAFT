# -*- coding: utf-8 -*-
"""
Spyder Editor
# Author: Antonio Cavalcante de Lima Neto
# Github: 
# Date: 14-10-2022
# Updated: 


"""
import numpy as np
from scipy.constants import k as kb  # constante de Boltzmann J /K
from scipy.constants import Avogadro as Navo  # numero de avogadro mol^-1
from scipy.constants import pi
from scipy import optimize
import numpy as np
# from pcsaft import pcsaft_den, pcsaft_ares, pcsaft_fugcoef, pcsaft_gres, pcsaft_osmoticC
import numdifftools as nd

class PC_SAFT_EOS():
    def __init__(self, m, sigma, epsilon_k, M=None, kbi=None, kAB_k=None, eAB_k=None, S=None):
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

        return

    # EQ A.9 ok!
    def PC_SAFT_d_T(self, T):
        d_T = np.zeros(self.ncomp)
        sigma = self.sigma
        epsilon_k = self.epsilon_k

        for i in range(self.ncomp):
            d_T[i] = sigma[i]*(1.0-0.12*np.exp(-3.*epsilon_k[i]/(T)))

        return d_T

    # EQ A.8 ---- problemas-----
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

    # A.16 and A.17 ok! -> only for I2
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

    # EQ A.35 ------------------- it's not correcly written in thw paper
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


    # EQ A.42 and A.43    derivative confirmed ok!
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

    def PC_SAFT_BubblePy(self, T, x, guessP, guessy):
        # estimativa inicial de Pbolha
        P = guessP*1.
        y = guessy*1
        j = 0
        res_loopP = 1
        soma_y = np.sum(y)
        y = y/soma_y  # guess
        soma_y = 1.
        tol_loopP = 1e-6
        jmax = 100
        while (res_loopP > tol_loopP and j < jmax):
            # fugacidade do liquido
            dens_L, dens_V = self.PC_SAFT_dens(T, P, x)
            # print(Vm_L,Vm_V)
            phi_L = self.PC_SAFT_phi(dens_L, T, x)
            f_L = phi_L*x*P  # vetorial
            res_loopY = 1  # qq coisa maior q 1e-6
            tol_loopy = 1e-6
            i = 0
            imax = 100
            while (res_loopY > tol_loopy and i < imax):
                # fugacidade do vapor
                dens_L, dens_V = self.PC_SAFT_dens(T, P, y/soma_y)
                #print (Vm_L,Vm_V )
                phi_V = self.PC_SAFT_phi(dens_V, T, y/soma_y)
                f_V = phi_V*y*P  # vetorial
                # calculando yi'
                y_novo = y*f_L/f_V
                res = y_novo-y  # vetorial
                res_loopY = np.linalg.norm(res)
                # normalizacao dos y p/ entrar na eq de estado como uma fracao
                soma_y = np.sum(y_novo)
                y = y_novo*1  # cópia do array essencial para o método numérico #salvando o novo como o 'y velho' da iteracao anterior
                i = i+1
            res_loopP = abs(soma_y-1)
            P = P*soma_y  # atualiza P
            j = j+1
        if np.linalg.norm(y/soma_y - x/np.sum(x)) < 1e-4:
            #print ('solução trivial')
            P = np.nan
            y[:] = np.nan
        return P, y, i, j

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
    #         print(it,X_A.flatten())
            X_A_old = X_A*1.
            for i in range(ncomp):
                for j in range(nsite):
                    sum1 = 0
                    for k in range(ncomp):
                        sum2 = 0
                        for l in range(nsite):
                            #                         if (l==j): continue #cycle #ESSA LINHA PROHIBE A AUTO ASSOCIAÇÃO
                            #                         sum2 += S[j,i]*S[l,k]*X_A_old[l,k]*delta[l,k,j,i]
                            sum2 += S[l, k]*X_A_old[l, k]*delta[l, k, j, i]
                            # aqui era só S[l,k] eu troquei para S[j,i]*S[l,k]
                            # (tem que dar zero se pelo menos um dos sitios nao existir - eqv delta[l,k,j,i]==0)
                            # em segunda análise usando apenas S[l,k] bate com a solução analítica para o 3B
                            # incusive ao adicionar um sítio [2] com multiplicidade zero S[2,0]=0
                            # se o sitio j não existir eu vou estar calculando um X estacionário para ele
                            # usando as interações virtuais com os demais sitios k que existam
    #                     end do
                        sum1 += x[k]*sum2
    #                 end do
                    X_A[j, i] = 1./(1. + rho*sum1)
                    dif[j, i] = np.abs((X_A[j, i]-X_A_old[j, i])/X_A[j, i])
    #             end do
    #         end do
            res = np.max(dif)
    #         print(res)
    #     end do
    #     print('X_tan ',it, flush=True)
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

    def PC_SAFT_kappat(self, T, P, x):
        phase='liq'
        opt=False
        method=None
        dens = self.PC_SAFT_dens(T, P, x,phase)
        ddens_dP = self.PC_SAFT_ddens( T, P, x, phase, opt, method)
        # dens = self.PC_SAFT_massdens(T, P, x,phase='vap',method = None)
        kappat = ddens_dP /dens

        return kappat

    def PC_SAFT_alphap(self, T, P, x):
        ddens_dT = self.PC_SAFT_ddens_dT(T, P, x)
        dens = self.PC_SAFT_massdens(T, P, x, phase='vap')
        alphap = -ddens_dT/dens

        return alphap

    def PC_SAFT_kappat2(self, T, P, x):
        step = 1
        P_mais = P+step
        P_menos = P-step
        dens_mais = self.PC_SAFT_dens(T, P_mais, x, phase='vap')
        dens_menos = self.PC_SAFT_dens(T, P_menos, x, phase='vap')
        dens = self.PC_SAFT_dens(T, P, x, phase='vap')
        Z_mais = self.PC_SAFT_Z(dens_mais, T, x)
        Z_menos = self.PC_SAFT_Z(dens_menos, T, x)
        Z = self.PC_SAFT_Z(dens, T, x)

        dZ_dP = (Z_mais - Z_menos)/(2*step)
        kappat = 1/P - 1/Z*dZ_dP

        print(dZ_dP, Z, kappat)
        return kappat

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
            
# def testptn():
#     import numpy as np
#     from pcsaft import pcsaft_den,pcsaft_ares,pcsaft_fugcoef,pcsaft_gres,pcsaft_osmoticC,pcsaft_p
#     # Binary mixture: water-acetic acid
#     #0 = water, 1 = lysozyme
#     t = 298.15 # K
#     p = 100000 # Pa
#     xa=.99 #9.99997485e-01
#     x = np.array([xa, 1-xa])
#     M = np.array([18.015, 14400 ])
#     m = np.asarray([1.2047, 882.27])
#     s = np.asarray([0,  2.566])
#     e = np.asarray([353.95, 347.67])
#     volAB = np.asarray([0.0451, 1.00])
#     eAB  = np.asarray([2425.67, 1773.84])
#     k_ij = np.asarray([[0, -0.0890],
#                         [-0.0890, 0]])

#     # s[0] = 3.8395 + 1.2828*np.exp(-0.0074944*t) - 1.3939*np.exp(-0.00056029*t)
#     s[0] = 2.7927 + 10.11*np.exp(-0.01775*t) - 1.417 *np.exp(-0.01146*t)
#     params = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB, 'k_ij':k_ij}

#     eos=PC_SAFT_EOS(m,sigma=s,epsilon_k=e,kbi=k_ij,kAB_k=volAB,eAB_k=eAB,S=np.array([[1, 28],[1, 11 ]]),M = M)
#     eos2 = PC_SAFT_EOS(m,sigma=s,epsilon_k=e,kbi=k_ij,M=M)

#     denst1 = eos.PC_SAFT_massdens(t, p, x,phase='liq')
#     denst2 = eos2.PC_SAFT_massdens(t, p, x,phase='liq')
#     print('dens tony  of water + Lysozyme w/ association at {} K: {} '.format(t, denst1))
#     print('dens tony  of water + Lysozyme at {} K: {} '.format(t, denst2))

#     a_rest1 = eos.PC_SAFT_a_res(denst1, t, x)
#     a_rest2 = eos2.PC_SAFT_a_res(denst2, t, x)
#     # x_A = eos.PC_SAFT_X_tan(rho,t,x)
#     Z1 = eos.PC_SAFT_Z(denst1,t,x)
#     Z2 = eos2.PC_SAFT_Z(denst2,t,x)
#     print('ares tony of water + Lysozyme w/ association at {} K: {} J /mol'.format(t, a_rest1))
#     print('ares tony of water + Lysozyme at {} K: {} J /mol'.format(t, a_rest2))


#     # print('xa tony of water + acetic at {} K: {}'.format(t, x_A))
#     print('Z tony of water + Lysozyme w/ association at {} K: {} '.format(t, Z1))
#     print('Z tony of water + Lysozyme at  {} K: {} '.format(t, Z2))

#     densz = pcsaft_den(t,p,x,params)
#     phiz = pcsaft_fugcoef(t, densz, x, params)
#     dens = eos.PC_SAFT_dens(t, p, x,phase = 'liq')
#     phi1 = eos.PC_SAFT_phi(dens, t, x)
#     print('phi zmeri of water + Lysozyme w/ association at {} K: {} '.format(t, phiz))
#     print('phi tony of water + Lysozyme at  {} K: {} '.format(t, phi1))

#     Pz = pcsaft_p(t, densz, x, params)
#     Pt = eos.PC_SAFT_Pressure(dens, t, x)
#     print('P zmeri of water + Lysozyme w/ association at {} K: {} Pa'.format(t, Pz))
#     print('P tony of water + Lysozyme at  {} K: {} '.format(t, Pt))

#     osmotic = eos.PC_SAFT_osmotic(dens,t,x)
#     print('osmotic coeficient of Lysozyme at  {} K: {} '.format(t, osmotic))
#     gresz = pcsaft_gres(t, densz, x, params)
#     grest = eos.PC_SAFT_G_res(dens, t, x)
#     print('gres zmeri of water + Lysozyme w/ association at {} K: {} '.format(t, gresz))
#     print('gres tony of water + Lysozyme at  {} K: {} '.format(t, grest))

#     t = 298.15
#     x0 = 0.1e-3/(0.1e-3+1000/M[0])
#     xf = 0.8e-3/(0.8e-3+1000/M[0])

#     vx = np.linspace(x0, xf,20)
#     vdens = vx*1
#     c = vx*1
#     for i in range(len(vx)):
#         x = np.array([1-vx[i], vx[i]])
#         vdens[i] = eos.PC_SAFT_massdens(t, p, x,phase='liq')
#         c[i] = vx[i]/(x[0]/1000*M[0])*1e+3
#     import matplotlib.pyplot as plt

#     print(vdens,c)
#     plt.plot(c,vdens,linestyle='-',alpha = 1)
# testptn()


# def testamino():
    # import numpy as np
    # from pcsaft import pcsaft_den,pcsaft_ares,pcsaft_fugcoef
    # # Binary mixture: water-acetic acid
    # #0 = water, 1 = glycine
    # t = 298.15 # K
    # p = 101325 # Pa
    # x = np.asarray([0.5,0.5])
    # m = np.asarray([1.2047, 4.8495])
    # s = np.asarray([0, 2.3270])
    # e = np.asarray([353.95,  216.96])
    # volAB = np.asarray([0.0451,  0.0393])
    # eAB = np.asarray([2425.67,  2598.06])
    # k_ij = np.asarray([[0, -6.12/100],
    #                     [-6.12/100, 0]])

    # s[0] = 3.8395 + 1.2828*np.exp(-0.0074944*t) - 1.3939*np.exp(-0.00056029*t)
    # params = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB, 'k_ij':k_ij}

    # eos=PC_SAFT_EOS(m,sigma=s,epsilon_k=e,kbi=k_ij,kAB_k=volAB,eAB_k=eAB,S=np.array([[1, 1],[1, 1]]))


# testamino()


# def testassc():
#     import numpy as np
#     from pcsaft import pcsaft_den,pcsaft_ares,pcsaft_fugcoef
#     # Binary mixture: water-acetic acid
#     #0 = water, 1 = acetic acid
#     t = 298.15 # K
#     p = 101325 # Pa
#     x = np.asarray([0.5,0.5])
#     m = np.asarray([1.2047, 1.3403])
#     s = np.asarray([0, 3.8582])
#     e = np.asarray([353.95, 211.59])
#     volAB = np.asarray([0.0451, 0.075550])
#     eAB = np.asarray([2425.67, 3044.4])
#     k_ij = np.asarray([[0, -0.127],
#                         [-0.127, 0]])

#     s[0] = 3.8395 + 1.2828*np.exp(-0.0074944*t) - 1.3939*np.exp(-0.00056029*t)
#     params = {'m':m, 's':s, 'e':e, 'e_assoc':eAB, 'vol_a':volAB, 'k_ij':k_ij}

#     eos=PC_SAFT_EOS(m,sigma=s,epsilon_k=e,kbi=k_ij,kAB_k=volAB,eAB_k=eAB,S=np.array([[1, 1],[1, 1]]))
#     eos2 = PC_SAFT_EOS(m,sigma=s,epsilon_k=e,kbi=k_ij)

#     densz = pcsaft_den(t, p, x, params, phase='liq')
#     denst1 = eos.PC_SAFT_dens(t, p, x,phase = 'liq')
#     denst2 = eos2.PC_SAFT_dens(t, p, x,phase = 'liq')
#     print('dens zmeri of water + acetic at {} K: {} '.format(t, densz))
#     print('dens tony  of water + acetic w/ association at {} K: {} '.format(t, denst1))
#     print('dens tony  of water + acetic at {} K: {} '.format(t, denst2))

#     a_resz  = pcsaft_ares(t, densz, x, params)
#     a_rest1 = eos.PC_SAFT_a_res(denst1, t, x)
#     a_rest2 = eos2.PC_SAFT_a_res(denst2, t, x)
#     # x_A = eos.PC_SAFT_X_tan(rho,t,x)
#     Z1 = eos.PC_SAFT_Z(denst1,t,x)
#     Z2 = eos2.PC_SAFT_Z(denst2,t,x)
#     print('ares zmeri of water + acetic at {} K: {} J /mol'.format(t, a_resz))
#     print('ares tony of water + acetic w/ association at {} K: {} J /mol'.format(t, a_rest1))
#     print('ares tony of water + acetic at {} K: {} J /mol'.format(t, a_rest2))

#     # print('xa tony of water + acetic at {} K: {}'.format(t, x_A))
#     print('Z tony of water + acetic w/ association at {} K: {} '.format(t, Z1))
#     print('Z tony of water + acetic at  {} K: {} '.format(t, Z2))

#     phiz = pcsaft_fugcoef(t, densz, x, params)
#     phi1 = eos.PC_SAFT_phi(denst1, t, x)
#     print('phi zmeri of water + acetic w/ association at {} K: {} '.format(t, phiz))
#     print('phi tony of water + acetic at  {} K: {} '.format(t, phi1))

# testassc()
# def test():
#     ncomp=1
#     cnames=["methane", "ethane"]
#     import matplotlib.pyplot as plt
#     x = np.asarray([1])#,0.5])
#     M = np.array([16.043])#, 30.07])
#     m = np.array([1.])#, 1.6069])
#     sigma = s = np.array([3.7039])#,3.5206])
#     epsilon_k = e = np.array([150.03])#,191.42])
#     kbi = k_ij = np.zeros([ncomp,ncomp])
#     # kbi[0,1] =   0.001
#     # kbi[1,0] =  0.001

#     tab= np.loadtxt("dadosmetano.txt")
#     eos=PC_SAFT_EOS(ncomp,m,sigma,epsilon_k,M,kbi)
#     # eos=PC_SAFT_EOS(ncomp,m,sigma,epsilon_k,M,kbi,kAB,eAB)
#     vT = np.linspace(109.38, 190,18)
#     vP = vT*1
#     for i in range(len(vT)):
#         vP[i] = eos.PC_SAFT_Psat2(vT[i],1e+6)[0]
#     plt.plot(vT,vP, color ="r",linestyle='-',alpha = 0.3)
#     plt.scatter(tab[:,0],tab[:,1]*101325)
# test()

# def test():
# # Toluene
#     import numpy as np
#     from pcsaft import pcsaft_den,pcsaft_ares
#     ncomp = 1
#     x = np.asarray([1.])
#     m = np.asarray([2.8149])
#     s =sigma = np.asarray([3.7169])
#     e = epsilon_k = np.asarray([285.69])
#     pyargs = {'m':m, 's':s, 'e':e}
#     eos=PC_SAFT_EOS(ncomp,m,sigma,epsilon_k)
#     T = 273 # K
#     P = 101325 # Pa
#     den = pcsaft_den(T, P, x, pyargs, phase='liq')
#     print('Density of toluene at {} K: {} mol m^-3'.format(T, den))
#     dens,_ = eos.PC_SAFT_dens(T, P, x)
#     print("PC-SAFT-Tony_dens = ",dens)
#     rho = eos.PC_SAFT_rho(dens)
#     print('rho of toluene at {} K: {} molecule A^-3'.format(T, rho))
#     rhõ = eos.PC_SAFT_rhõ(x,dens=dens)
#     print('reduced rho of toluene at {} K: {} '.format(T, rhõ))
#     ares = pcsaft_ares(T, dens, x, pyargs)
#     print('ares zmeri of toluene at {} K: {} J /mol'.format(T, ares))
#     arest = eos.PC_SAFT_a_res(dens, T, x)
#     print('ares tony of toluene at {} K: {} J /mol'.format(T, arest))
#     ãres = eos.PC_SAFT_ã_res(rhõ, T, x)
#     print('reduced ares of toluene at {} K: {}'.format(T, ãres))
#     mu_res = eos.PC_SAFT_mu_res_kT(dens, T, x)
#     print('mu_res of toluene at {} K: {}'.format(T, mu_res))
#     admu_res = eos.PC_SAFT_admu_res(dens, T, x)
#     print('reduced mu res of toluene at {} K: {}'.format(T, admu_res))

# test()
#     T=t=400.
#     dens = rho = 15033.11420899
#     P = 1e7 #10e+7
#     pyargs = {'m':m, 's':s, 'e':e,'k_ij':k_ij}
#     eos=PC_SAFT_EOS(ncomp,m,sigma,epsilon_k,M,kbi)

#     # print("PC-SAFT-Tony_dens = ",eos.PC_SAFT_dens(T, P, x))
#     print("PC-SAFT-zmeri_dens=",pcsaft.pcsaft_den(t, P, x, pyargs))

#     print("PC-SAFT-Tony_ares = ",eos.PC_SAFT_a_res(dens, T, x))
#     print("PC-SAFT-zmeri_ares=",pcsaft.pcsaft_ares(t, rho, x, pyargs))

#     print("PC-SAFT-Tony_phi = ", eos.PC_SAFT_phi(dens,T,x))
#     print("PC-SAFT-zmeri_phi=",pcsaft.pcsaft_fugcoef(t, rho, x, pyargs))


#     print("dcsi", eos.PC_SAFT_mat_dcsi_dxk(dens,T),"\r\n\n\n",
#     "dghs", eos.PC_SAFT_dghs_dx(dens,T,x),"\r\n\n\n",
#     "dghs_num", eos.PC_SAFT_dghs_dx_num(dens,T,x),"\r\n\n\n",
#     "da_hs", eos.PC_SAFT_da_hs_dx(dens,T,x),"\r\n\n\n",
#     "da_hs_num", eos.PC_SAFT_da_hs_dx_num(dens,T,x),"\r\n\n\n",
#     "da_hc", eos.PC_SAFT_da_hc_dx(dens,T,x),"\r\n\n\n",
#     "da_hc num", eos.PC_SAFT_da_hc_dx_num(dens,T,x),"\r\n\n\n",
#     "dai e dbi _dx", eos.PC_SAFT_dai_dx_e_dbi_dx(x),"\r\n\n\n",
#     "dai e dbi _dx num", eos.PC_SAFT_dai_dx_e_dbi_dx_num(x),"\r\n\n\n",
#     "dI1 e dI2", eos.PC_SAFT_dI1_dx_e_dI2_dx(dens,T,x),"\r\n\n\n",
#     "dI1 e dI2 num", eos.PC_SAFT_dI1_dx_e_dI2_dx_num(dens,T,x),"\r\n\n\n",
#     "dC1",eos.PC_SAFT_dC1_dx(dens,T,x),"\r\n\n\n",
#     "dC1 num",eos.PC_SAFT_dC1_dx_num(dens,T,x),"\r\n\n\n",
#     "dm_2esig_3 e dm_2e_2sig_3 _dx" ,eos.PC_SAFT_dm_2esig_3_dx_e_dm_2e_2sig_3_dx(dens,T,x),"\r\n\n\n"
#     "dm_2esig_3 e dm_2e_2sig_3 _dx num" ,eos.PC_SAFT_dm_2esig_3_dx_e_dm_2e_2sig_3_dx_num(dens,T,x),"\r\n\n\n"
#     "da_disp",eos.PC_SAFT_da_disp_dx(dens,T,x),"\r\n\n\n",
#     "da_disp num",eos.PC_SAFT_da_disp_dx_num(dens,T,x),"\r\n\n\n",
#     "da_res",eos.PC_SAFT_da_res_dx(dens,T,x),"\r\n\n\n",
#     "mu_res",eos.PC_SAFT_mu_res_kT(dens,T,x),"\r\n\n\n",
#     "phi",eos.PC_SAFT_phi(dens,T,x))

#     print(eos.PC_SAFT_BubblePy(400,np.array([0.5,0.5]),1.83e+6,np.array([0.5,0.5])))

# test()

# %%
# def test()
#     vP = np.linspace(1e+6,4e+6, 200)
#     vdenTonyL_1=np.zeros(len(vP))
#     vdenTonyV_1=np.zeros(len(vP))
#     vdenzmeri = np.zeros(len(vP))
#     for i in range(len(vP)):
#         vdenTonyL_1[i],vdenTonyV_1[i]=eos.PC_SAFT_dens(T, vP[i],x)
#         #vdenTonyL_1[i],vdenTonyV_1[i]=eos.PC_SAFT_dens2(T, vP[i],x)
#         #vdenTonyL_1[i],vdenTonyV_1[i]=eos.PC_SAFT_dens3(T, vP[i],x) #minimizando A+P0V para monofásico
#         vdenzmeri[i]=pcsaft.pcsaft_den(T,vP[i],x,pyargs)
#         print(round(i/len(vP)*100),"%")
#     import matplotlib.pyplot as plt
#     plt.plot(1/vdenTonyL_1,vP,color='b',linewidth=0.5,linestyle='--',alpha=1)
#     plt.plot(1/vdenTonyV_1,vP,color='c', linewidth=0.5,linestyle=':',alpha=1)

#     plt.plot(1/vdenzmeri,vP, color ="r",linestyle='-',alpha = 0.3)

# # test()

# #calcular fugacidade -> calcular x,y -> calcular P - > calcular dens


# %%
#     import matplotlib.pyplot as plt

#     npontos = 10
#     yguess= np.array([0.1,0.9])
#     vx=np.linspace(0.5,0.999, npontos)
#     vP = vx*1
#     vy = []
#     for i in range(npontos):
#         vP[i],vyi= eos.curva_BubblePy(T,np.array([vx[i],1-vx[i]]),P,yguess)
#         vy.append(vyi)
#         print(vy)
#     plt.plot(vx,vP,color='b',linewidth=0.5,linestyle='--',alpha=1)
#     plt.plot(vy,vP,color='c', linewidth=0.5,linestyle=':',alpha=1)

# test()

# def test2():
#     import matplotlib.pyplot as plt
#     import numpy as np
#     #butane and methane
#     ncomp = 2
#     x = np.asarray([0.5,0.5])
#     M = np.array([ 58.123,16.043])
#     m = np.array([2.3316,1.])
#     sigma = s = np.array([ 3.7086,3.7039])
#     epsilon_k = e = np.array([ 222.88,150.03])
#     kbi = k_ij = np.zeros([ncomp,ncomp])
#     kbi[0,1] =   0.022
#     kbi[1,0] =  0.022

#     T=t=300.
#     dens = rho = 15033.11420899
#     P = 1e7 #10e+7
#     pyargs = {'m':m, 's':s, 'e':e,'k_ij':k_ij}
#     eos=PC_SAFT_EOS(m,sigma,epsilon_k,M,kbi)

#     npontos = 100
#     T = 230

#     vx=np.linspace(0.2,0.999, npontos)
#     vP = vx*1
#     vy = []
#     for i in range(npontos):
#         vP[i],vyi = eos.curva_BubblePy(T,np.array([vx[i],1-vx[i]]),1e+6,0.4)
#         vy.append(vyi[0])
#     plt.plot(vx,vP,color='b',linewidth=0.5,linestyle='--',alpha=1)
#     plt.plot(vy,vP,color='c', linewidth=0.5,linestyle=':',alpha=1)


# test2()

# def testmetano():
#     import matplotlib.pyplot as plt
#     from pcsaft import pcsaft_den,pcsaft_ares
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     tab=pd.read_excel('dados.xlsx',sheet_name='metano')
#     T = tab.T.values
#     densv = tab.densv.values
#     P = tab.P.values
#     densl = tab.densl.values

#     #butane and methane
#     ncomp = 2
#     x = np.asarray([0,1])
#     M = np.array([ 58.123,16.043])
#     m = np.array([2.3316,1.])
#     sigma = s = np.array([ 3.7086,3.7039])
#     epsilon_k = e = np.array([ 222.88,150.03])
#     kbi = k_ij = np.zeros([ncomp,ncomp])
#     kbi[0,1] =   0.022
#     kbi[1,0] =  0.022

#     pyargs = {'m':m, 's':s, 'e':e,'k_ij':k_ij}
#     eos=PC_SAFT_EOS(m,sigma,epsilon_k,M,kbi)
#     denslc = P*1
#     densvc = P*1
#     # for i in range(len(P)):
#     #     denslc[i],densvc[i] = eos.PC_SAFT_massdens(T[0][i],P[i],x)
#     i = len(P)-1
#     print(T[0][i], P[i])
#     print(eos.PC_SAFT_triP(T[0][i], P[i], x))
    # fig = plt.figure()
    # ax =fig.add_subplot(111)
    # ax.set_xlabel(r'$\rho$ [kg/m³]')
    # ax.set_ylabel(r'P [Pa]')
    # ax.set_title('densidades do metano saturado, HS + Disp')
    # plt.plot(denslc,P,color='b',linewidth=1,linestyle='-',alpha=1)
    # plt.plot(densvc,P,color='r', linewidth=1,linestyle='-',alpha=1)
    # plt.scatter(densv,P,color='r')
    # plt.scatter(densl,P,color='b')

# testmetano()

# def testarg():
#     import matplotlib.pyplot as plt
#     from pcsaft import pcsaft_den,pcsaft_ares
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     ncomp = 1
#     x = np.asarray([1])
#     M = np.array([ 39.948])
#     m = np.array([0.98])
#     sigma = s = np.array([3.401])
#     epsilon_k = e = np.array([273.15-123.12])
#     kbi = k_ij = np.zeros([ncomp,ncomp])
#     # kbi[0,1] =   0.001
#     # kbi[1,0] =  0.001
#     eos=PC_SAFT_EOS(m,sigma=s,epsilon_k=e,kbi=k_ij,M=M)#,kAB_k=volAB,eAB_k=eAB,S=np.array([[1, 28],[1, 11 ]]),M = M)

#     import pandas as pd
#     tab=pd.read_excel('dados.xlsx',sheet_name='psatarg')
#     T = tab.Temp.values
#     densv = tab.densv.values
#     P = tab.P.values
#     densl = tab.densl.values


#     P= P+0.0
#     Pcalc = P*1
#     densvc = P*1
#     denslc = P*1


#     for i in range(len(P)):
#         denslc[i],densvc[i] = eos.PC_SAFT_massdens(T[i], P[i], x)

#     plt.plot(denslc,P,color='b',linewidth=0.5,linestyle='-',alpha=1)
#     plt.plot(densvc,P,color='r', linewidth=0.5,linestyle='-',alpha=1)
#     plt.scatter(densv,P,color='r')
#     plt.scatter(densl,P,color='b')

# testarg()

# def testbutano():
#     import matplotlib.pyplot as plt
#     from pcsaft import pcsaft_den, pcsaft_ares
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     tab = pd.read_excel('dados.xlsx', sheet_name='butano')
#     T = tab.Temp.values
#     densv = tab.densv.values
#     P = tab.P.values
#     densl = tab.densl.values
#     P = P/10.0

#     #butane and methane
#     ncomp = 2
#     x = np.asarray([1, 0])
#     M = np.array([58.123, 16.043])
#     m = np.array([2.3316, 1.])
#     sigma = s = np.array([3.7086, 3.7039])
#     epsilon_k = e = np.array([222.88, 150.03])
#     kbi = k_ij = np.zeros([ncomp, ncomp])
#     kbi[0, 1] = 0.022
#     kbi[1, 0] = 0.022

#     pyargs = {'m': m, 's': s, 'e': e, 'k_ij': k_ij}
#     eos = PC_SAFT_EOS(m, sigma, epsilon_k, M, kbi)
#     denslc = P*1
#     densvc = P*1
#     for i in range(len(P)):
#         denslc[i], densvc[i] = eos.PC_SAFT_massdens(T[i], P[i], x)

#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_ylabel(r'P [Pa]')
#     ax.set_xlabel(r'$\rho$ [Kg/m³]')
#     ax.set_title(r'denslidade do butano saturado, HS + disp')
#     ax.plot(denslc, P, color='b', linewidth=1, linestyle='-', alpha=1)
#     ax.plot(densvc, P, color='r', linewidth=1, linestyle='-', alpha=1)
#     ax.scatter(densv, P, color='r',alpha=0.7)
#     ax.scatter(densl, P, color='b',alpha=0.7)

# testbutano()

# def testwater():
#     import matplotlib.pyplot as plt
#     from pcsaft import pcsaft_den,pcsaft_ares
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     tab=pd.read_excel('dados.xlsx',sheet_name='water')
#     T = tab.Temp.values
#     densv = tab.densv.values
#     P = tab.P.values
#     kappat = tab.kappat.values
#     P=P+0.0
#     densl = tab.densl.values
#     # Binary mixture: water-acetic acid
#     #0 = water, 1 = ethanol

#     x = np.array([1, 0])
#     M = np.array([18.015, 46.069 ])
#     m = np.asarray([1.2047, 2.38267])
#     s = np.asarray([0,  3.17706])
#     e = np.asarray([353.95, 198.237])
#     volAB = np.asarray([0.0451,  0.03238])
#     eAB  = np.asarray([2425.67,  2653.39])
#     k_ij = np.asarray([[0, -0.185],
#                         [-0.185, 0]])
#     denslc = P*1
#     densvc = P*1
#     for i in range(len(P)):
#         t = T[i]
#         s[0] = 2.7927 + 10.11*np.exp(-0.01775*t) - 1.417 *np.exp(-0.01146*t)
#         eos=PC_SAFT_EOS(m,sigma=s,epsilon_k=e,M=M,kbi=k_ij,kAB_k=volAB,eAB_k=eAB,S=np.array([[1, 2],[1, 2]]))
#         denslc[i],densvc[i] = eos.PC_SAFT_massdens(T[i],P[i],x)

#     fig = plt.figure()
#     ax =fig.add_subplot(111)
#     ax.set_ylabel(r'P [Pa]')
#     ax.set_xlabel(r'$\rho$ [Kg/m³]')
#     ax.set_title(r'denslidade da água líquida saturada ')
#     plt.plot(denslc,P,color='b',linewidth=1,linestyle='-',alpha=1)
#     plt.scatter(densl,P,color='b',alpha=0.7)

#     fig = plt.figure()
#     ax =fig.add_subplot(111)
#     ax.set_title(r'denslidade vapor de água saturada ')
#     ax.set_ylabel(r'P [Pa]')
#     ax.set_xlabel(r'$\rho$ [Kg/m³]')
#     plt.plot(densvc,P,color='r', linewidth=1,linestyle='-',alpha=1)
#     plt.scatter(densv,P,color='r',alpha=0.7)


# testwater()

# def testwater2():
#     import matplotlib.pyplot as plt
#     from pcsaft import pcsaft_den,pcsaft_ares
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     tab=pd.read_excel('dados.xlsx',sheet_name='water')
#     T = tab.Temp.values
#     densv = tab.densv.values
#     P = tab.P.values
#     kappat = tab.kappat.values
#     P=P+0.0
#     densl = tab.densl.values
#     # Binary mixture: water-acetic acid
#     #0 = water, 1 = ethanol

#     x = np.array([1, 0])
#     M = np.array([18.015, 46.069 ])
#     m = np.asarray([1.2047, 2.38267])
#     s = np.asarray([0,  3.17706])
#     e = np.asarray([353.95, 198.237])
#     volAB = np.asarray([0.0451,  0.03238])
#     eAB  = np.asarray([2425.67,  2653.39])
#     k_ij = np.asarray([[0, -0.185],
#                         [-0.185, 0]])
#     denslc = P*1
#     densvc = P*1
#     for i in range(len(P)):
#         t = T[i]
#         s[0] = 2.7927 + 10.11*np.exp(-0.01775*t) - 1.417 *np.exp(-0.01146*t)
#         eos=PC_SAFT_EOS(m,sigma=s,epsilon_k=e,M=M,kbi=k_ij,kAB_k=volAB,eAB_k=eAB,S=np.array([[1, 2],[1, 2]]))
#         denslc[i],densvc[i] = eos.PC_SAFT_massdens(T[i],P[i],x,opt=False,method='hybr')

#     fig = plt.figure()
#     ax =fig.add_subplot(111)
#     ax.set_ylabel(r'P [Pa]')
#     ax.set_xlabel(r'$\rho$ [Kg/m³]')
#     ax.set_title(r'denslidade da água líquida saturada ')
#     plt.plot(denslc,P,color='b',linewidth=1,linestyle='-',alpha=1)
#     plt.scatter(densl,P,color='b',alpha=0.7)

#     fig = plt.figure()
#     ax =fig.add_subplot(111)
#     ax.set_title(r'denslidade vapor de água saturada')
#     ax.set_ylabel(r'P [Pa]')
#     ax.set_xlabel(r'$\rho$ [Kg/m³]')
#     plt.plot(densvc,P,color='r', linewidth=1,linestyle='-',alpha=1)
#     plt.scatter(densv,P,color='r',alpha=0.7)


# testwater2()

# def testdiclorofluormetano():
#     import matplotlib.pyplot as plt
#     import numpy as np
#     #diclorofluormetano e metano
#     ncomp = 2
#     x = np.asarray([1,0])
#     M = np.array([ 86.47,16.043])
#     m = np.array([2.54660,1.])
#     sigma = s = np.array([ 3.10920,3.7039])
#     epsilon_k = e = np.array([185.470,150.03])
#     kbi = k_ij = np.zeros([ncomp,ncomp])
#     kbi[0,1] =   0.022
#     kbi[1,0] =  0.022

#     pyargs = {'m':m, 's':s, 'e':e,'k_ij':k_ij}
#     eos=PC_SAFT_EOS(m,sigma,epsilon_k,M,kbi)

#     P = np.linspace(2,9,15)*1e+6
#     kappa = P*1
#     for i in range(len(P)):
#         kappa[i] = eos.PC_SAFT_kappat(310,P[i],x)*1e+9
#         if kappa[i] > 100:
#             kappa[i] = np.nan


#     plt.plot(P,kappa,color='b',linewidth=1,linestyle='--',alpha=1)
# testdiclorofluormetano()

# def testdiclorofluormetano():
#     import matplotlib.pyplot as plt
#     import numpy as np
#     #diclorofluormetano e metano
#     ncomp = 2
#     x = np.asarray([1,0])
#     M = np.array([ 86.47,16.043])
#     m = np.array([1.86000,1.])
#     sigma = s = np.array([ 3.51420,3.7039])
#     epsilon_k = e = np.array([ 269.776,150.03])
#     kbi = k_ij = np.zeros([ncomp,ncomp])
#     kbi[0,1] =   0.022
#     kbi[1,0] =  0.022

#     pyargs = {'m':m, 's':s, 'e':e,'k_ij':k_ij}
#     eos=PC_SAFT_EOS(m,sigma,epsilon_k,M,kbi)

#     P = np.linspace(2.5,9,15)*1e+6
#     kappa = P*1
#     for i in range(len(P)):
#         kappa[i] = eos.PC_SAFT_kappat(320,P[i],x)*1e+9
#         if kappa[i] > 100:
#             kappa[i] = np.nan

#     plt.plot(P,kappa,color='r',linewidth=0.5,linestyle='--',alpha=1)
# testdiclorofluormetano()

# def tolueneePropanol():
#     import matplotlib.pyplot as plt
#     import numpy as np
#     #toluene e Propanol
#     ncomp = 2
#     x = np.asarray([1,0])
#     M = np.array([ 92.14,60.0952])
#     m = np.array([ 2.8149,3.4957])
#     sigma = s = np.array([ 3.7169,3.0551])
#     epsilon_k = e = np.array([ 285.69, 214.57])
#     volAB = np.asarray([0,  0.035128])
#     eAB  = np.asarray([0, 2089.8])
#     k_ij = np.asarray([[0, 0],
#                         [0, 0]])

#     eos=PC_SAFT_EOS(m,sigma=s,epsilon_k=e,kbi=k_ij,kAB_k=volAB,eAB_k=eAB,S=np.array([[0,2],[0, 2]]),M=M)

#     X = np.linspace(0,1,50)
#     kappa = X*1
#     denso=X*1
#     denss=X*1
#     for i in range(len(X)):
#         x=np.array([1-X[i], X[i]])
#         kappa[i] = eos.PC_SAFT_kappat(303.15,5e+6,x)*1e+6
#         if kappa[i] > 100 or kappa[i] < 0:
#             kappa[i] = np.nan
#     plt.plot(X,kappa,color='black',linewidth=1,linestyle='--',alpha=1)
# tolueneePropanol()

# def methylbutyrate():
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
    
#     import time
#     from datetime import timedelta
#     start_time = time.monotonic()
    
#     tab=pd.read_excel('dados.xlsx',sheet_name='methylbutyrate')
#     P = tab.P.values*1e+6
#     kappaexp = tab.kappa.values
#     ncomp = 1
#     x = np.asarray([1])
#     M = np.array([ 102.132])
#     m = np.array([  3.556])
#     sigma = s = np.array([3.478])
#     epsilon_k = e = np.array([ 244.990])
#     eos=PC_SAFT_EOS(m,sigma=s,epsilon_k=e,M=M)
    
#     testet=0
#     while testet<1:
#         testet+=1
#         kappa = P*1
#         for i in range(len(P)):
#             kappa[i] = eos.PC_SAFT_kappat(353.15,P[i],x)*1e+9
#             print(kappa[i])
#             if kappa[i] > 7 or kappa[i] < 0:
#                 kappa[i] = np.nan


#     fig = plt.figure()
#     ax =fig.add_subplot(111)
#     ax.set_xlabel(r'$P/Pa$')
#     ax.set_ylabel(r'$\kappa/GPa$')
#     ax.set_title(r'Compressibilidade isotérmica do butirato de metila a 353K')
#     ax.plot(P,kappa,color='black',linewidth=1,linestyle='-',alpha=1)
#     ax.scatter(P,kappaexp,color='black',linewidth=1,linestyle='-',alpha=1)
#     end_time = time.monotonic()
#     tempot = (timedelta(seconds=end_time - start_time))
#     print(tempot)
    
# methylbutyrate()

# def testeqlv():
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     tab=pd.read_excel('dados.xlsx',sheet_name='waterethanol')
#     P = tab.P.values*1e+3
#     T = tab.Temp.values
#     x = np.array([tab.x1.values, 1-tab.x1.values])
#     x = x.transpose()
#     y = np.array([tab.y1.values, 1-tab.y1.values])
#     y = y.transpose()
#     M = np.array([18.015, 46.069 ])
#     m = np.asarray([1.2047, 2.38267])
#     s = np.asarray([0,  3.17706])
#     e = np.asarray([353.95, 198.237])
#     volAB = np.asarray([0.0451,  0.03238])
#     eAB  = np.asarray([2425.67,  2653.39])
#     k_ij = np.asarray([[0, -0.185],
#                         [-0.185, 0]])


#     ycalc = y*1
#     Pcalc = P*1
#     for i in range(len(P)):
#         t = T[i]
#         s[0] = 2.7927 + 10.11*np.exp(-0.01775*t) - 1.417 *np.exp(-0.01146*t)
#         eos=PC_SAFT_EOS(m,sigma=s,epsilon_k=e,M=M,kbi=k_ij,kAB_k=volAB,eAB_k=eAB,S=np.array([[1, 1],[1, 1]]))
#         Pcalc[i],ycalc[i],_,_ = eos.PC_SAFT_BubblePy(T[i],x[i],P[i],y[i])
#         print(Pcalc[i],P[i])
#         print(ycalc[i],y[i])

#     fig = plt.figure()
#     ax =fig.add_subplot(111)
#     ax.set_xlabel(r'$P/Pa$')
#     ax.set_ylabel(r'$\kappa/GPa$')
#     ax.set_title(r'Compressibilidade isotérmica para butirato de metila a 353K')
#     ax.scatter(P,x[0],color='black',linewidth=1,linestyle='-',alpha=1)
#     ax.scatter(P,y[0],color='black',linewidth=1,linestyle='-',alpha=1)
#     ax.plt(Pcalc,ycalc[0],color='black',linewidth=1,linestyle='-',alpha=1)
# testeqlv()


# def testptn():
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     # Binary mixture: water-acetic acid
#     #0 = water, 1 = lysozyme
#     t = 298.15 # K
#     p = 1e+6 # Pa
#     M = np.array([18.015, 66000 ])
#     m = np.asarray([1.2047, 4115.57])
#     s = np.asarray([0, 2.600])
#     e = np.asarray([353.95, 344.58])
#     volAB = np.asarray([0.0451, 1.00])
#     eAB  = np.asarray([2425.67,  1892.58])
#     k_ij = np.asarray([[0, -0.0716],
#                 [-0.0716, 0]])

#     tab=pd.read_excel('dados.xlsx',sheet_name='lysozyme')
#     x2 = tab.molalidade.values
#     x2 = x2*1e-3/(1e+3/M[0])
#     x = np.array([1-x2, x2])
#     x = x.transpose()
#     densp = tab.densp.values
#     denspcalc = densp*1
#     s[0] = 2.7927 + 10.11*np.exp(-0.01775*t) - 1.417 *np.exp(-0.01146*t)
#     eos=PC_SAFT_EOS(m,sigma=s,epsilon_k=e,kbi=k_ij,kAB_k=volAB,eAB_k=eAB,S=np.array([[1, 176],[1, 153]]),M = M)
#     for i in range(len(densp)):
#         denspcalc[i] = eos.PC_SAFT_massdens(t,p,x[i],phase='liq')

#     fig = plt.figure()
#     ax =fig.add_subplot(111)
#     ax.set_ylabel(r'$\rho$ [kg/m³]')
#     ax.set_xlabel(r'$x_1$')
#     ax.set_title('Densidades da solução de BSA(1) dissolvida em água a 298 K e 1 bar')
#     ax.scatter(x[:,1],densp,color='black',linewidth=1,linestyle='-',alpha=0.7)
#     ax.plot(x[:,1],denspcalc,color='black',linewidth=1,linestyle='-',alpha=1)
# testptn()

# def butmet():
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     #butane and methane
#     ncomp = 2
#     x = np.asarray([0.5,0.5])
#     M = np.array([ 58.123,16.043])
#     m = np.array([2.3316,1.])
#     sigma = s = np.array([ 3.7086,3.7039])
#     epsilon_k = e = np.array([ 222.88,150.03])
#     kbi = k_ij = np.zeros([ncomp,ncomp])
#     kbi[0,1] =   0.022
#     kbi[1,0] =  0.022

#     tab=pd.read_excel('dados.xlsx',sheet_name='butmet')
#     x1 = 1-tab.x1.values
#     y1 = 1-tab.y1.values
#     T = tab.Temp.values[0]
#     P = tab.P.values
#     eos=PC_SAFT_EOS(m,sigma=s,epsilon_k=e,kbi=k_ij,M = M)
#     npontos = 30
#     vx=np.linspace(0.18,0.999, npontos)
#     vP = []
#     vy = []
#     for i in range(npontos):
#         vPi,vyi = eos.curva_BubblePy(T,np.array([vx[i],1-vx[i]]),1e+7,0.4)
#         vy.append(vyi[0])
#         vP.append(vPi[0])

#     fig = plt.figure()
#     ax =fig.add_subplot(111)
#     ax.set_ylabel('P [Pa]')
#     ax.set_xlabel(r'$x_1$')
#     ax.set_title('ELV de butano(1) + metano(2) a 233K')
#     ax.scatter(x1,P,color='blue',alpha=0.7)
#     ax.scatter(y1,P,color='red',alpha=0.7)
#     ax.plot(vx,vP,color='b',linewidth=1,linestyle='-',alpha=1)
#     ax.plot(vy,vP,color='c', linewidth=1,linestyle='-',alpha=1)


# butmet()

# def testarg():
#     import matplotlib.pyplot as plt
#     from pcsaft import pcsaft_den,pcsaft_ares
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     ncomp = 1
#     x = np.asarray([0.9285])
#     M = np.array([ 39.948])
#     m = np.array([1])
#     sigma = s = np.array([3.478])
#     epsilon_k = e = np.array([122.23])
#     kbi = k_ij = np.zeros([ncomp,ncomp])
#     # kbi[0,1] =   0.001
#     # kbi[1,0] =  0.001
#     eos=PC_SAFT_EOS(m,sigma=s,epsilon_k=e,kbi=k_ij,M=M)

#     import pandas as pd
#     tab=pd.read_excel('dados.xlsx',sheet_name='psatarg')
#     T = tab.Temp.values
#     densv = tab.densv.values
#     P = tab.P.values
#     densl = tab.densl.values

#     Tcalc = np.linspace(min(T)-10, max(T)+10,50)
#     Pguess = np.linspace(min(P)*0.9, max(P)*1.1,50)
#     Psat = Tcalc*1
#     P= P+0.0
#     Pcalc = P*1
#     densvc = P*1
#     denslc = P*1


#     for i in range(len(Pcalc)):
#         # Psat[i] = eos.PC_SAFT_Psat2(T[i],Pguess[i])[0]
#         denslc[i],densvc[i] = eos.PC_SAFT_massdens(T[i], Pcalc[i], x)
    
#     fig = plt.figure()
#     ax =fig.add_subplot(111)
#     ax.set_ylabel('P [Pa]')
#     ax.set_xlabel(r'$\rho $ [kg/m³]')
#     ax.set_title('densidade do argonio saturado sem HC')
#     ax.plot(denslc,P,color='b',linewidth=1,linestyle='-',alpha=1)
#     ax.plot(densvc,P,color='r', linewidth=1,linestyle='-',alpha=1)
#     ax.scatter(densv,P,color='r',alpha = 0.7)
#     ax.scatter(densl,P,color='b',alpha =0.7)
        
# testarg()

# def co2met():
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     #co2 and methane
#     ncomp = 2
#     x = np.asarray([0.9539,0.0461])
#     M = np.array([ 44.01,16.043])
#     m = np.array([2.6037,1.])
#     sigma = s = np.array([ 2.555,3.7039])
#     epsilon_k = e = np.array([ 151.04,150.03])
#     kbi = k_ij = np.zeros([ncomp,ncomp])
#     kbi[0,1] = kbi[1,0]=  0.0561 
#     eos=PC_SAFT_EOS(m,sigma=s,epsilon_k=e,kbi=k_ij,M=M)
#     P = np.linspace(10,35,50)*1e+6
#     T = 293.15
#     kappa = P*1
#     for i in range(len(P)):
#         kappa[i] = eos.PC_SAFT_kappat(T,P[i],x)*1e+6
#     fig = plt.figure()
#     ax =fig.add_subplot(111)
#     ax.set_ylabel('P [Pa]')
#     ax.set_xlabel(r'$x_1$')
#     ax.set_title('ELV de butano(1) + metano(2) a 293K')
#     # ax.scatter(P,kappa,color='blue',alpha=0.7)
#     # ax.scatter(y1,P,color='red',alpha=0.7)
#     ax.plot(P,kappa,color='b',linewidth=1,linestyle='-',alpha=1)
#     # ax.plot(vy,vP,color='c', linewidth=1,linestyle='-',alpha=1)


# co2met()

# def co2():
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     tab=pd.read_excel('dados.xlsx',sheet_name='CO2')
#     Cpexp = tab.Cp.values
#     Temp = tab.Temp.values
#     #co2 and methane
#     ncomp = 2
#     x = np.asarray([1,0])
#     M = np.array([ 44.01,16.043])
#     m = np.array([2.6037,1.])
#     sigma = s = np.array([ 2.555,3.7039])
#     epsilon_k = e = np.array([ 151.04,150.03])
#     kbi = k_ij = np.zeros([ncomp,ncomp])
#     kbi[0,1] = kbi[1,0]=  0.0561 
#     eos=PC_SAFT_EOS(m,sigma=s,epsilon_k=e,kbi=k_ij,M=M)
#     P =  8e+6
#     T = np.linspace(250,450,100)
#     Cp = T*1
#     for i in range(len(T)):
#         Cp[i] = eos.PC_SAFT_Cv(T[i],P,x,phase='vap')
#         if Cp[i]<30:
#             Cp[i] = np.nan
#     fig = plt.figure()
#     ax =fig.add_subplot(111)
#     ax.set_ylabel('Cp [J/mol*K]')
#     ax.set_xlabel(r'$T$ [K]')
#     ax.set_title('Capacidade térmica (Cp) de CO2 a 8 MPa')
#     # ax.scatter(Temp,Cpexp,color='blue',alpha=0.7)
#     ax.plot(T,Cp,color='b',linewidth=1,linestyle='-',alpha=1)


# co2()