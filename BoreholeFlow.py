#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Created by Joshua Thompson 
# Last updated 09/02/2022

import numpy as np
import math as math
import pandas as pd
import quadprog as qp
from scipy import optimize
from scipy import signal
from scipy.signal import lsim
import matplotlib.pyplot as plt
np.set_printoptions(precision=15)
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size


class BhFlow:
    # Lets the class initialize the object's attributes
    def __init__(self, K, R0, Rw):
        self.K = K
        self.R0 = R0
        self.Rw = Rw
        
    # Defines the number of layers within the well based on the number of hydraulic conductivities
    def NumberOfLayers(self):
        nlayers = len(self.K)
        return nlayers
    
    # Creates a zero array (fills with all zeros)  with dimensions (nlayers x 1)
    def NlayersZerosArray(self, nlayers):
        a = np.zeros((nlayers, 1))
        return a
    
    # Creates a zero array or with dimensions (nlayers x nlayers) or square matrix
    def NxNLayersZerosArray(self, nlayers):
        a = np.zeros((nlayers, nlayers))
        return a
    
    # Creates ones array (fills with all ones) with dimensions (nlayers x 1)
    def NlayersOnesArray(self, nlayers):
        a = np.ones((nlayers, 1))
        return a
    
    # discretize the screened portion of the well based on interval thickness (b) and depth to bottom of well
    def Discretize(self, zwellbot, nlayers, b):
        z = np.zeros(nlayers)
        ztop = self.NlayersZerosArray(nlayers)
        zmid = self.NlayersZerosArray(nlayers)
        zbot = self.NlayersZerosArray(nlayers)
        
        zbot[0] = zwellbot
        ztop[0] = zbot[0] + b[0]
        zmid[0] = zbot[0] + b[0] / 2

        for i in np.arange(1, nlayers):
            zbot[i] = ztop[i - 1]
            ztop[i] = zbot[i] + b[i]
            zmid[i] = zmid[i - 1] + b[i - 1]/2 + b[i]/2
            
        return ztop, zmid, zbot
    
    # Runs flow model
    def BoreholeFlowModel(self,nlayers, b, h0, zwellbot, ztop, zmid, zbot, zobs, zpump, Qpump, Qsampling):
    
        # constants
        g = 9.81 * 100; # (cm/s^2)
        nu = 1.0023e-6 * 100 * 100; # (cm^2/s)
        
        # Discretize the well. Calling the discretize method
        self.Discretize(zwellbot, nlayers, b)

        if zobs.size > 0:
            iobs = np.argwhere((zbot <= zobs) & (ztop > zobs))[0, 0]
        if zpump.size > 0:
            ipump = np.argwhere((zbot <= zpump) & (ztop > zpump))[0, 0]
        Ca = np.round(2 * np.pi,4) * np.round(self.K * b, 4) / np.round(np.log(self.R0 / self.Rw), 4)
        Cw = np.zeros((nlayers, nlayers))
        Cw[0 , 1] = np.pi * g * self.Rw ** 4 / (8 * nu * 0.5 * (b[0] + b[1]))

        for i in np.arange(1, nlayers - 1):
            Cw[i, i + 1] = np.pi * g * self.Rw ** 4 / (8 * nu * 0.5 * (b[i] + b[i + 1]))
            Cw[i, i - 1] = np.pi * g * self.Rw ** 4 / (8 * nu * 0.5 * (b[i] + b[i - 1]))
        Cw[-1, -2] = np.pi * g * self.Rw ** 4 / (8 * nu * 0.5 * (b[-1] + b[-2]))
        
        A = np.zeros((nlayers, nlayers))
        B = np.zeros((nlayers, nlayers * 2))

        for i in range(nlayers):
            iinorout = np.argwhere(Cw[i, :] != 0)
            A[i, i] = -Ca[i] - np.sum(Cw[i, iinorout])
            A[i, iinorout] = Cw[i, iinorout]
            B[i, i] = Ca[i]
        u = np.zeros((nlayers * 2, 1))
        u[0:nlayers] = h0
        if (ipump == iobs):
            B[ipump, nlayers + ipump] = 1
            u[nlayers + ipump, 0] = - Qpump - Qsampling
        else:
            B[ipump, nlayers + ipump] = 1
            B[iobs, nlayers + iobs] = 1
            u[nlayers + ipump, 0] = -Qpump
            u[nlayers + iobs, 0] = -Qsampling

        h = -np.linalg.inv(A) @ B @ u
        
        Qr = Ca * (h0 - h)
        Qb = self.NlayersZerosArray(nlayers)
        
        for i in range(nlayers - 1):
            Qb[i] = Cw[i, i+1] * (h[i] - h[i+1])

        return Qb, Qr, h, zmid, zbot, ztop
    
    def BoreholeTransportModel(self, nlayers, b, Qb, Qr, C0, ztop, zmid, zbot, zobs, zpump, Qpump, Qsampling):
        V = np.pi * self.Rw**2 *b # Volume of each well interval (cm^3)
        Qp = self.NlayersZerosArray(nlayers)
 
        if zobs.size > 0:
            iobs = np.argwhere((zbot <= zobs) & (ztop > zobs))[0, 0]
            Qp[iobs] = Qsampling
            
        if zpump.size > 0:
            ipump = np.argwhere((zbot <= zpump) & (ztop > zpump))[0, 0]
            Qp[ipump] = Qp[ipump] + Qpump
        cobs = []
        
        Qin_total = np.sum(Qr)
        Qout_total = np.sum(Qp)
        if (np.abs(Qin_total - Qout_total) > 1e-4):
            print('Pumping and borehole flows do not balance.', np.abs(Qin_total - Qout_total))
            
        Q = self.NxNLayersZerosArray(nlayers)
        A = self.NxNLayersZerosArray(nlayers)
        B = self.NxNLayersZerosArray(nlayers)
        H = self.NlayersZerosArray(nlayers)

        for i in range(nlayers):
            if(Qb[i] > 0):
                Q[i, i + 1] = Qb[i]
            elif (Qb[i] < 0):
                Q[i + 1, i] = -Qb[i]

        for i in range(nlayers):
            iOut = np.argwhere(Q[i, :] > 0)
            iIn = np.argwhere(Q[:, i] > 0)
            if iOut.size > 0:
                A[i, i] = -np.sum(Q[i, iOut])
            if iIn.size > 0:
                A[i, iIn] = Q[iIn, i]
            A[i, i] = A[i, i] + np.minimum(Qr[i], 0) - Qp[i]
            A[i, :] = A[i, :] / V[i]
            B[i, i] = np.maximum(Qr[i], 0)
            B[i, :] = B[i, :] / V[i]
       
        if((zobs < zmid[0]).any() or (zobs > zmid[-1]).any()):
            H[iobs] = 1
        else:
            if((zobs == zmid[iobs]).any()):
                H[iobs] = 1
            elif((zobs < zmid[iobs]).any()):
                H[iobs] = (zobs[i] - zmid[iobs - 1]) / (zmid[iobs] - zmid[iobs - 1])
                H[iobs - 1] = 1 - H[iobs]
            elif((zobs > zmid[iobs]).any()):
                H[iobs] = (zobs - zmid[iobs]) / (zmid[iobs + 1] - zmid[iobs])
                H[iobs + 1] = 1 - H[iobs]
        H = H.T
        c = -np.linalg.inv(A) @ B @ C0
        G = -H @ np.linalg.inv(A) @ B
        
        cobs = G @ C0
        
        if zpump.size > 0:
            cpump = c[ipump]
        else:
            cpump = []
        return c, cobs, G
    
    def TransientTransportModel(self, t, nlayers, b, Qb, Qr, C0, Cinitial, ztop, zmid, zbot, zobs, zpump, Qpump, Qsampling):
        V = np.pi * self.Rw**2 *b # Volume of each well interval (cm^3)
        Qp = self.NlayersZerosArray(nlayers)
 
        if zobs.size > 0:
            iobs = np.argwhere((zbot <= zobs) & (ztop > zobs))[0, 0]
            Qp[iobs] = Qsampling
            
        if zpump.size > 0:
            ipump = np.argwhere((zbot <= zpump) & (ztop > zpump))[0, 0]
            Qp[ipump] = Qp[ipump] + Qpump
        cobs = []
        
        Qin_total = np.sum(Qr)
        Qout_total = np.sum(Qp)
        if (np.abs(Qin_total - Qout_total) > 1e-4):
            print('Pumping and borehole flows do not balance.', np.abs(Qin_total - Qout_total))
            
        Q = self.NxNLayersZerosArray(nlayers)
        A = self.NxNLayersZerosArray(nlayers)
        B = self.NxNLayersZerosArray(nlayers)
        H = self.NlayersZerosArray(nlayers)

        for i in range(nlayers):
            if(Qb[i] > 0):
                Q[i, i + 1] = Qb[i]
            elif (Qb[i] < 0):
                Q[i + 1, i] = -Qb[i]

        for i in range(nlayers):
            iOut = np.argwhere(Q[i, :] > 0)
            iIn = np.argwhere(Q[:, i] > 0)
            if iOut.size > 0:
                A[i, i] = -np.sum(Q[i, iOut])
            if iIn.size > 0:
                A[i, iIn] = Q[iIn, i]
            A[i, i] = A[i, i] + np.minimum(Qr[i], 0) - Qp[i]
            A[i, :] = A[i, :] / V[i]
            B[i, i] = np.maximum(Qr[i], 0)
            B[i, :] = B[i, :] / V[i]
       
        if((zobs < zmid[0]).any() or (zobs > zmid[-1]).any()):
            H[iobs] = 1
        else:
            if((zobs == zmid[iobs]).any()):
                H[iobs] = 1
            elif((zobs < zmid[iobs]).any()):
                H[iobs] = (zobs[i] - zmid[iobs - 1]) / (zmid[iobs] - zmid[iobs - 1])
                H[iobs - 1] = 1 - H[iobs]
            elif((zobs > zmid[iobs]).any()):
                H[iobs] = (zobs - zmid[iobs]) / (zmid[iobs + 1] - zmid[iobs])
                H[iobs + 1] = 1 - H[iobs]
        H = H.T
        css = -np.linalg.inv(A) @ B @ C0
        
        zero_matrix = np.zeros((1, nlayers))
        C0 = C0.reshape(nlayers, 1)
        C0rep = np.repeat(C0, len(t), axis = 1)
        C0rep = C0rep.T
        bhsys = signal.StateSpace(A, B, H, zero_matrix)
        t = np.asarray(t)
        tout, cobs, c = lsim(bhsys, U=C0rep, T=t, X0=Cinitial)
        
        return c, css, cobs, tout
    
    def MakeDataLists(self):
        ctrueAll = []
        cobsAll = []
        gAll = []
        zobsAll = []
        zpumpAll = []
        QpumpAll = []
        QobsAll = []
        QbAll = []
        QrAll = []
        hAll = []
        return ctrueAll, cobsAll, gAll, zobsAll, zpumpAll, QpumpAll, QobsAll, QbAll, QrAll, hAll
    
    def MakeDataListsTransient(self):
        ctrueAll = []
        cobsAll = []
        zobsAll = []
        cssAll = []
        tAll = []
        QbAll = []
        QrAll = []
        return ctrueAll, cobsAll, zobsAll, cssAll, tAll, QbAll, QrAll
    
    def AppendData(self, c, cobs, G, zobs, zpump, Qpump, Qsampling,
                   Qb, Qr, h, ctrueAll, cobsAll, gAll, zobsAll, zpumpAll, QpumpAll,
                   QobsAll, QbAll, QrAll, hAll):
        ctrueAll.append(c)
        cobsAll.append(cobs)
        gAll.append(G)
        zobsAll.append(zobs)
        zpumpAll.append(zpump)
        QpumpAll.append(Qpump)
        QobsAll.append(Qsampling)
        QbAll.append(Qb)
        QrAll.append(Qr)
        hAll.append(h)
        return ctrueAll, cobsAll, gAll, zobsAll, zpumpAll, QpumpAll, QobsAll, QbAll, QrAll, hAll
    
    def AppendDataTransient(self, c, cobs, zobs, css, t, Qb, Qr, ctrueAll, cobsAll, zobsAll, cssAll, tAll, QbAll, QrAll):
        ctrueAll.append(c)
        cobsAll.append(cobs)
        zobsAll.append(zobs)
        cssAll.append(css)
        tAll.append(t)
        QbAll.append(Qb)
        QrAll.append(Qr)
        return ctrueAll, cobsAll, zobsAll, cssAll, tAll, QbAll, QrAll

    def Convert2Arrays(self, zobs, nlayers, ctrueAll, cobsAll, gAll, zobsAll, zpumpAll, QpumpAll, QobsAll, QbAll, QrAll, hAll):
        ctrueAll = np.array(ctrueAll).reshape(325, 1)
        cobsAll = np.array(cobsAll).reshape(len(zobs), 1)
        gAll = np.array(gAll).reshape(len(zobs), nlayers)
        zobsAll = np.array(zobsAll).reshape(len(zobs), 1)
        zpumpAll = np.array(zpumpAll)
        QpumpAll = np.array(QpumpAll).reshape(len(zobs), 1)
        QobsAll = np.array(QobsAll).reshape(len(zobs), 1)
        QbAll = np.array(QbAll).reshape(len(zobs), nlayers)
        QrAll = np.array(QrAll).reshape(len(zobs), nlayers)
        hAll = np.array(hAll).reshape(len(zobs), nlayers)
        return ctrueAll, cobsAll, gAll, zobsAll, zpumpAll, QpumpAll, QobsAll, QbAll, QrAll, hAll
    
    def Convert2ArraysTransient(self, nlayers, zobs, t, ctrueAll, cobsAll, zobsAll, cssAll, tAll, QbAll, QrAll):
        ctrueAll = np.array(ctrueAll).reshape(len(t) * len(zobs), len(zobs))
        cobsAll = np.array(cobsAll).reshape(len(zobs), len(t))
        zobsAll = np.array(zobsAll).reshape(len(zobs), 1)
        cssAll = np.array(cssAll).reshape(len(zobs) * len(zobs), 1)
        tAll = np.array(tAll).reshape(len(zobs), len(t))
        QbAll = np.array(QbAll).reshape(len(zobs), nlayers)
        QrAll = np.array(QrAll).reshape(len(zobs), nlayers)
        return ctrueAll, cobsAll, zobsAll, cssAll, tAll, QbAll, QrAll
    
    def CombineFandTOutput(self, ctrue1, ctrue2, cobs1, cobs2, g1, g2, zobs1, zobs2, zpump1, zpump2, Qpump1, Qpump2, 
                           Qobs1, Qobs2, Qb1, Qb2, Qr1, Qr2, h1, h2):
        ctrueCombine = np.concatenate((ctrue1, ctrue2), axis=0)
        cobsCombine = np.concatenate((cobs1, cobs2), axis=0)
        gCombine = np.concatenate((g1, g2), axis=0)
        zobsCombine = np.concatenate((zobs1, zobs2), axis=0)
        zpumpCombine = np.concatenate((zpump1, zpump2), axis=0)
        QpumpCombine = np.concatenate((Qpump1, Qpump2), axis=0)
        QobsCombine = np.concatenate((Qobs1, Qobs2), axis=0)
        QbCombine = np.concatenate((Qb1, Qb2), axis=0)
        QrCombine = np.concatenate((Qr1, Qr2), axis=0)
        hCombine = np.concatenate((h1, h2), axis=0)
        return ctrueCombine, cobsCombine, gCombine, zobsCombine, zpumpCombine, QpumpCombine, QobsCombine, QbCombine, QrCombine, hCombine

class invertBHconc:
    def __init__(self, nlayers, G, X, cobs, concerr, alphmax, alph):
        self.nlayers = nlayers
        self.G = G
        self.X = X
        self.cobs = cobs
        self.concerr = concerr
        self.alphmax = alphmax
        self.alph = alph        
        
    def chi2misfit(self, alph):
        c0est = np.linalg.inv(self.G.T @ self.G + alph * np.identity(self.nlayers) - alph * self.X @ np.linalg.inv(self.X.T @ self.X) @             self.X.T) @ self.G.T @ self.cobs
        rms = (np.mean((self.G @ c0est - self.cobs)**2))**0.5
        return (rms - self.concerr)**2
    
    def chi2misfit4quad(self, alph):
            Gquad = self.G.T @ self.G + alph * np.identity(self.nlayers) - alph * self.X @ np.linalg.inv(self.X.T @ self.X) @ self.X.T
            a = self.G.T @ self.cobs
            a = a.ravel()
            Cquad = np.identity(self.nlayers)
            Cquad = np.array(Cquad, dtype = np.double)
            bquad = np.zeros((self.nlayers, 1))
            bquad = bquad.ravel()
            c0est = qp.solve_qp(Gquad, a, Cquad, bquad)
            rms = (np.mean((self.G @ c0est[0] - self.cobs.ravel())**2))**0.5
            return (rms - concerr)**2

    def InvertData(self):
        alphEst = optimize.fminbound(self.chi2misfit, 0, self.alphmax)
        c0est = np.linalg.inv(self.G.T @ self.G + alphEst * np.identity(self.nlayers) - alphEst * self.X @ np.linalg.inv(self.X.T @ self.X)         @ self.X.T) @ self.G.T @ self.cobs
        rms = (np.mean((self.G @ c0est - self.cobs) **2))**0.5
        R = np.linalg.inv(self.G.T @ self.G + alphEst * np.identity(self.nlayers) - alphEst * self.X @ np.linalg.inv(self.X.T @ self.X)         @ self.X.T) @ self.G.T @ self.G
        if(np.min(c0est) < 0):
            print('negative concentrationm estimates produced; switching to quadratic programming', np.min(c0est))
            alphEst = optimize.fminbound(self.chi2misfit, 0, self.alphmax)
            Gquad = self.G.T @ self.G + alphEst * np.identity(self.nlayers) - alphEst * self.X @ np.linalg.inv(self.X.T @ self.X) @ self.X.T
            a = self.G.T @ self.cobs
            a = a.ravel()
            Cquad = np.identity(self.nlayers)
            Cquad = np.array(Cquad, dtype = np.double)
            bquad = np.zeros((self.nlayers, 1))
            bquad = bquad.ravel()
            c0est = qp.solve_qp(Gquad, a, Cquad, bquad)
            rms = (np.mean((self.G @ c0est[0] - self.cobs.ravel())**2))**0.5
            R = np.linalg.inv(self.G.T @ self.G + alphEst * np.identity(self.nlayers) - alphEst * self.X @ np.linalg.inv(self.X.T @ self.X)         @ self.X.T) @ self.G.T @ self.G
        return c0est, rms, R
    
class PlotFigures:

    def PlotFigure3(self, C0, K, QbAmbient, zmid, QrAmbient, cobsLF1, zobsLF1):
        fig, (ax1, ax2, ax3)  = plt.subplots(1, 3, figsize = (9, 7), sharey=False)
        logK = np.log10(K)
        logK = np.flip(logK)
        yvals = np.linspace(0, -50, 11)
        ztopplot = np.linspace(-5000 / 100, 0 / 100, 26)
        ax1.vlines(x = 0, ymin = -50, ymax = 0, linestyle = (0, (5, 5)), linewidth = 1.5, color = '#918f8f' )
        ax1.plot(QbAmbient * 60, zmid / 100, color = '#14b4f3', linewidth = 2.5)
        ax1.imshow(logK, cmap = 'gray', extent=[-1000, 300, -50, 0], aspect = 'auto')
        ax1.set_xlabel('Flow (ml / min)', color = '#4c4d4f', fontsize = 12, fontweight = 'bold')
        ax1.set_ylabel('Depth (m)', color = '#4c4d4f', fontsize = 12, fontweight = 'bold')
        ax1.set_xlim(-1000, 300)
        ax1.set_xticks([-1000, 0]) 
        ax1.set_ylim(-50, 0)
        ax1.set_yticks(yvals);

        ax2.vlines(x = 0, ymin = -50, ymax = 0, linestyle = (0, (5, 5)), linewidth = 1.5, color = '#918f8f' )
        ax2.plot(QrAmbient * 60, zmid / 100, label = 'estimated', color = '#14b4f3', linewidth = 2.5)
        ax2.imshow(logK, cmap = 'gray', extent=[-200, 200, -50, 0], aspect = 'auto')
        ax2.set_xlabel('Flow (ml / min)', color = '#4c4d4f', fontsize = 12, fontweight = 'bold')
        ax2.set_ylabel('Depth (m)', color = '#4c4d4f', fontsize = 12, fontweight = 'bold')
        ax2.set_xlim(-200, 200)
        ax2.set_xticks([-200, 0, 200]) 
        ax2.set_ylim(-50, 0)
        ax2.set_yticks(yvals);

        ax3.scatter(cobsLF1, zobsLF1 / 100, color = 'red', label = 'sampled')
        ax3.stairs(C0, ztopplot, orientation='horizontal', color = '#14b4f3', linewidth = 2.5, baseline = None, label = 'true')
        im = ax3.imshow(logK, cmap = 'gray', extent=[1, 550, -50, 0], aspect = 'auto')
        ax3.set_xlabel('Concentration (ppm)', color = '#4c4d4f', fontsize = 12, fontweight = 'bold')
        ax3.set_ylabel('Depth (m)', color = '#4c4d4f', fontsize = 12, fontweight = 'bold')
        ax3.set_xlim(0, 550)
        ax3.set_xticks([0, 500]) 
        ax3.set_ylim(-50, 0)
        ax3.set_yticks(yvals)
        ax3.legend()

        cax = fig.add_axes([ax3.get_position().x1 + 0.03, ax3.get_position().y0, 0.02, ax3.get_position().height])
        cbar = plt.colorbar(im, cax = cax)
        cbar.set_label('$ \mathbf{log_10[K (cm/s)]}$', fontsize = 12, color = '#4c4d4f', weight = 'bold')
        cbar.ax.tick_params(direction="in")

        plt.subplots_adjust(wspace=0.4)
        return
    def PlotFigure4(self, C0, K, zobs, cobsLF1, tLF1T, imin):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (7, 9))
        logK = np.log10(K)
        logK = np.flip(logK)
        ztopplot = np.linspace(-5000 / 100, 0 / 100, 26)

        ax1.stairs(C0.ravel(), ztopplot, orientation='horizontal', color = '#14b4f3', linewidth = 2.5, baseline = None, label = 'true')
        ax1.set_ylabel('Depth (m)', color = '#4c4d4f', fontsize = 12, fontweight = 'bold')
        ax1.set_xlabel('Concentration (ppm)', color = '#4c4d4f', fontsize = 12, fontweight = 'bold')
        ax1.set_xlim(0, 550)
        ax1.set_xticks([0, 500])
        im = ax1.imshow(logK, cmap = 'gray', extent=[1, 550, -50, 0], aspect = 'auto')
        cax = fig.add_axes([ax1.get_position().x1 - 0.075, ax1.get_position().y0, 0.02, ax1.get_position().height])
        cbar = plt.colorbar(im, cax = cax)
        cbar.set_label('$ \mathbf{log_10[K (cm/s)]}$', fontsize = 12, color = '#4c4d4f', weight = 'bold')
        cbar.ax.tick_params(direction="in")

        ax2.plot(tLF1T[imin] / 60 / 60, zobs / 100, color = 'k')
        ax2.set_ylabel('Depth (m)', color = '#4c4d4f', fontsize = 12, fontweight = 'bold')
        ax2.set_xlabel('Time (hours)', color = '#4c4d4f', fontsize = 12, fontweight = 'bold')
        ax2.set_xlim(-10, 200)
        ax2.set_xticks([0, 100, 200])
        im2 = ax2.imshow(cobsLF1, aspect = 'auto', origin = 'lower', extent=[-10, 200, -50, 0], cmap = 'turbo')
        cax2 = fig.add_axes([ax2.get_position().x1 + 0.015, ax2.get_position().y0, 0.02, ax2.get_position().height])
        cbar2 = plt.colorbar(im2, cax = cax2)
        cbar2.set_label('$ \mathbf{log_{10}[K (cm/s)]}$', fontsize = 12, color = '#4c4d4f', weight = 'bold')
        cbar2.ax.tick_params(direction="in")

        plt.subplots_adjust(wspace=1)
        return
    
    def PlotFigure5(self, C0, cobsLF1, zobsLF1, c0estLF1, cobsLF2, zobsLF2, c0estLF2, c0estCombine, c0estComX, zmid, RLF1, RLF2, RCombine, RComX):
        fig, axs = plt.subplots(2, 4, figsize = (10, 7))
        yvals = np.linspace(0, -50, 11)
        ztopplot = np.linspace(-5000 / 100, 0 / 100, 26)

        axs[0, 0].stairs(C0.ravel(), ztopplot, orientation='horizontal', color = '#14b4f3', linewidth = 2.5, baseline = None, label =               'true')
        axs[0, 0].scatter(cobsLF1, zobsLF1 / 100, color = 'red', label = 'sampled')
        axs[0, 0].plot(c0estLF1, zmid / 100, color = 'k', linewidth = 1.8, label = 'estimates')
        # ax1.set_xlabel('Concentration (mg/l)', color = '#4c4d4f', fontsize = 12, fontweight = 'bold')
        axs[0, 0].set_ylabel('Depth (m)', color = '#4c4d4f', fontsize = 12, fontweight = 'bold')
        axs[0, 0].set_xlim(0, 550)
        axs[0, 0].set_xticks([0, 500]) 
        # axs[0, 0].set_ylim(-50, 0)
        axs[0, 0].set_yticks(yvals)
        axs[0, 0].legend(loc='lower center', bbox_to_anchor=(0.5, -0.573))

        axs[0, 1].stairs(C0.ravel(), ztopplot, orientation='horizontal', color = '#14b4f3', linewidth = 2.5, baseline = None, label =               'true')
        axs[0, 1].scatter(cobsLF2, zobsLF2 / 100, color = 'red', label = 'sampled' )
        axs[0, 1].plot(c0estLF2, zmid / 100, color = 'k', linewidth = 1.8, label = 'estimates')
        # ax2.set_xlabel('Concentration (mg/l)', color = '#4c4d4f', fontsize = 12, fontweight = 'bold')
        axs[0, 1].set_xlim(0, 550)
        axs[0, 1].set_xticks([0, 500]) 
        # axs[0, 1].set_ylim(-50, 0)
        axs[0, 1].set_yticks(yvals)
        axs[0, 1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.573))
        # axs[0, 1].get_shared_y_axes().join(axs[0, 0], axs[0, 1])
        axs[0, 1].set_yticklabels([])

        axs[0, 2].stairs(C0.ravel(), ztopplot, orientation='horizontal', color = '#14b4f3', linewidth = 2.5, baseline = None, label =               'true')
        axs[0, 2].scatter(cobsLF1, zobsLF1 / 100, color = 'red', label = 'L-F #1')
        axs[0, 2].scatter(cobsLF2, zobsLF2 / 100, color = 'None', edgecolor = 'red', label = 'L-F #2')
        axs[0, 2].plot(c0estCombine, zmid / 100, color = 'k', linewidth = 1.8, label = 'estimates')
        # ax3.set_xlabel('Concentration (mg/l)', color = '#4c4d4f', fontsize = 12, fontweight = 'bold')
        axs[0, 2].set_xlim(0, 550)
        axs[0, 2].set_xticks([0, 500]) 
        # axs[0, 2].set_ylim(-50, 0)
        axs[0, 2].set_yticks(yvals)
        axs[0, 2].set_yticklabels([])
        axs[0, 2].legend(loc='lower center', bbox_to_anchor=(0.5, -0.65))

        axs[0, 3].stairs(C0.ravel(), ztopplot, orientation='horizontal', color = '#14b4f3', linewidth = 2.5, baseline = None, label =               'true')
        axs[0, 3].scatter(cobsLF1, zobsLF1 / 100, color = 'red', label = 'L-F #1')
        axs[0, 3].scatter(cobsLF2, zobsLF2 / 100, color = 'None', edgecolor = 'red', label = 'L-F #2')
        axs[0, 3].plot(c0estComX, zmid / 100, color = 'k', linewidth = 1.8, label = 'estimates')
        # ax4.set_xlabel('Concentration (mg/l)', color = '#4c4d4f', fontsize = 12, fontweight = 'bold')
        axs[0, 3].set_xlim(0, 550)
        axs[0, 3].set_xticks([0, 500]) 
        # axs[0, 3].set_ylim(-50, 0)
        axs[0, 3].set_yticks(yvals)
        axs[0, 3].set_yticklabels([])

        axs[0, 3].legend(loc='lower center', bbox_to_anchor=(0.5, -0.65))

        fig.text(0.5, 0.475, 'Concentration (mg/l)', ha='center', va='center', color = '#4c4d4f', fontsize = 12, fontweight = 'bold')

        fig.text(0.5, 0.225, 'Resolution Matrices', ha='center', va='center', color = '#4c4d4f', fontsize = 16, fontweight = 'bold')


        im1 = axs[1, 0].imshow(RLF1, extent=[40, 0, 40, 0], origin='lower', cmap = 'turbo')
        im1.set_clim(-.25,1)
        axs[1, 0].set_ylabel('Depth (m)', color = '#4c4d4f', fontsize = 12, fontweight = 'bold')
        axs[1, 0].set_yticks([0, 20, 40])
        axs[1, 0].set_xticks([0, 20, 40])

        im2 = axs[1, 1].imshow(RLF2, extent=[40, 0, 40, 0], origin='lower', cmap = 'turbo')
        im2.set_clim(-.25,1)
        axs[1, 1].set_yticklabels([])
        axs[1, 1].set_xticks([0, 20, 40])
        

        im3 = axs[1, 2].imshow(RCombine, extent=[40, 0, 40, 0], origin='lower', cmap = 'turbo')
        im3.set_clim(-.25,1)
        axs[1, 2].set_yticklabels([])
        axs[1, 2].set_xticks([0, 20, 40])


        im4 = axs[1, 3].imshow(RComX, extent=[40, 0, 40, 0], origin='lower', cmap = 'turbo')
        im4.set_clim(-.25,1)
        axs[1, 3].set_yticklabels([])
        axs[1, 3].set_xticks([0, 20, 40])


        cax = fig.add_axes([axs[1, 3].get_position().x1 + 0.01, axs[1, 3].get_position().y0 - 0.225, 0.01, axs[1, 3].get_position().height +         0.01])
        cbar = plt.colorbar(im1, cax, cmap = 'turbo', ticks = [-0.2, 0, 0.5, 0.99])
        cbar.set_ticklabels([-0.2, 0, 0.5, 1])

        fig.text(0.5, -0.1, 'Depth (m)', ha='center', va='center', color = '#4c4d4f', fontsize = 12, fontweight = 'bold')

        plt.subplots_adjust(left=0.1, bottom=-.1, right=0.9, top=0.9, wspace=0.2, hspace=0.8)
        return
    def PlotFigure6(self, Qb, zobs, zmid, ztop, gAll):
        yvals = np.linspace(400, 240, 9)

        fig = plt.figure(figsize = (18, 6))
        ax1 = plt.subplot2grid((1, 25), (0, 1), colspan=2)
        ax1.plot(Qb, -1*ztop)
        ax1.set_yticks(yvals)
        ax1.set_xticks([-1, -0.5, 0])
        ax1.set_ylabel('Depth (m)', color = '#4c4d4f', fontsize = 12, fontweight = 'bold')
        ax1.set_xlabel('Vertical Flow (l/min)', color = '#4c4d4f', fontsize = 12, fontweight = 'bold')
        ax1.invert_yaxis()

        yvals4Heat = np.linspace(250, 400, 4)
        ax2 = plt.subplot2grid((1, 25), (0, 5), colspan=10)
        im = ax2.imshow(gAll, origin='lower', cmap = 'turbo', aspect = 'auto', extent = [-400, -250, 400, 250])
        ax2.set_ylabel('Sample Depth (m)', color = '#4c4d4f', fontsize = 12, fontweight = 'bold')
        ax2.set_xlabel('Depth in Aquifer (m)', color = '#4c4d4f', fontsize = 12, fontweight = 'bold')
        ax2.set_yticks(yvals4Heat)
        ax2.set_xticks(-1*yvals4Heat)

        cax = fig.add_axes([ax2.get_position().x1 + 0.01, ax2.get_position().y0, 0.01, ax2.get_position().height])
        cbar = plt.colorbar(im, cax, cmap = 'turbo')
        # cbar.set_ticklabels([0, 0.5, 1])
        cbar.set_label('Sensitivity (-)', fontsize = 12, color = '#4c4d4f', weight = 'bold')

        ax3 = plt.subplot2grid((1, 25), (0, 18), colspan=2)
        ax3.plot(gAll[1,:], -1*ztop, label = zobs[1], color = 'red')
        ax3.plot(gAll[5,:], -1*ztop, label = zobs[5], color = 'green')
        ax3.plot(gAll[9,:], -1*ztop, label = zobs[9], color = 'k')
        ax3.plot(gAll[14,:], -1*ztop, label = zobs[14], color = 'blue')
        ax3.set_yticks(yvals)
        ax3.invert_yaxis()
        ax3.set_xlabel('Sensitivity (-)', color = '#4c4d4f', fontsize = 12, fontweight = 'bold')
        ax3.set_ylabel('Sample Depth (m)', color = '#4c4d4f', fontsize = 12, fontweight = 'bold')
        ax3.legend(bbox_to_anchor = (1.05, 0.8), loc='center')

        diagG = np.diagonal(gAll.T @ gAll)
        ax4 = plt.subplot2grid((1, 25), (0, 22), colspan=2)
        ax4.plot(diagG, -1*zmid, color = 'k')
        ax4.set_yticks(yvals)
        ax4.invert_yaxis()
        ax4.set_xticks([0, 0.5, 1, 1.5])
        ax4.set_ylabel('Sample Depth (m)', color = '#4c4d4f', fontsize = 12, fontweight = 'bold')
        ax4.set_xlabel('Cumulative Sensitivity (-)', color = '#4c4d4f', fontsize = 12, fontweight = 'bold');
        return