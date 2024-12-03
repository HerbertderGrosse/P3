
import uncertainties.unumpy as unp
import uncertainties as u
from uncertainties.umath import log


from uncertainties import ufloat
import numpy as np
import pandas as pd
import scipy as sp
import kafe2
import matplotlib.pyplot as plt
import os
import glob
import struct
import sympy
from scipy.optimize import curve_fit
from pathlib import Path


def dat_extract(file):
    nevent1 = 0
    xList1 = []
    tList1 = []
    tDAQ1  = []
    event1 = []
    f1 = open(file, "rb")
    notEOF = True
    while notEOF and nevent1 < 100000000:
        # read header: record, boardID, channel, pattern, event, time_ns
        s = f1.read(24)
        if len(s) != 24:
            notEOF = False
            break
        record,boardID,channel,pattern,evt,time_ns = struct.unpack("<LLLLLL", s)
        #print record,boardID,channel,pattern,evt,time_ns
        # read data: record = Byte length of the waveform event (16 bit per sample) + 24 (6*4Byte) header length 
        s = f1.read(record-24)
        if len(s) != (record-24):
            notEOF = False
            break
        data = struct.unpack("<"+str(int((record-24)/2))+"H", s)
        # determine the baseline signal (average of first 10 samples)
        n_average = 20 
        x0=0
        for i in range(n_average):
            x0 = x0 + data[i]
        x0 = x0 / n_average

        # find the maximum peak in the waveform event (gliding average of 10 samples)
        imax = 0
        xmax = 0.
        xsum = 0.
        for i in range(n_average,int((record-24)/2)):
            xsum = xsum + data[i-n_average] - data[i]
            #xsum =  x0 - data[i]
            if xmax < xsum:
                xmax = xsum
                imax = i
        xList1.append(xmax/n_average)
        tList1.append(imax * 8)
        tDAQ1.append(time_ns * 8)
        event1.append(evt)
        nevent1 = nevent1 + 1
    f1.close()
    print("done")
    return xList1,tList1,nevent1

def plot_hist(xList1,nevent1,nBins=1000,binSize=2,x_left=None,x_right=None):
    
    
    n1, bins, patches1 = plt.hist(xList1, nBins, range=(0,nBins*binSize))
    plt.xlabel('energy (chn)')
    plt.ylabel('no. of events') 
    plt.grid(True)
    print ("Number of events: ",nevent1)
    if not(x_left is None):
        plt.xlim(left=x_left)
    if not(x_right is None):
        plt.xlim(right=x_right)
    return n1, bins, patches1
def getfromtxt(file):
    dat = np.genfromtxt(file, skip_header = 1)
    x,y = dat.T
    return x,y   
def integrate(x,y,mu, sig,const_offset=0,dat_name=None,func=None,label=True):

    sig=np.abs(sig)    
    von = int((mu-1.5*sig)/2)
    bis = int((mu+1.5*sig)/2)

    
    if dat_name:
        dat = np.genfromtxt("./"+dat_name+ '.txt', skip_header = 1)
        x,y = dat.T    
    if func is None:
        if label:
            plt.fill_between(x[von:bis], y[von:bis], y2=const_offset, color = "green", alpha = 0.5, label="integrated area")
        else:
            plt.fill_between(x[von:bis], y[von:bis], y2=const_offset, color = "green", alpha = 0.5)
    else:
        if label:
            plt.fill_between(func(x[von:bis]), y[von:bis], y2=const_offset, color = "green", alpha = 0.5, label="integrated area")
        else:
            plt.fill_between(func(x[von:bis]), y[von:bis], y2=const_offset, color = "green", alpha = 0.5)

    plt.xlabel('energy (chn)')
    plt.ylabel('no. of events')
    plt.grid()

    
    return np.sum(np.array(y[von:bis]-const_offset))


def gauss_fit(x,y,beg_ch,end_ch,offset=0,func=None,label=True):
    def gauss (x,A,sigma,µ):
        return A/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-µ)**2/(2*sigma**2))

    def gauss_const (x,A,sigma,µ,c):
        return A/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-µ)**2/(2*sigma**2))+c
    if offset>0:
        p_0 = [500, 30, (end_ch+beg_ch)/2,offset]
        par, cov= curve_fit(xdata = x[int(beg_ch/2):int(end_ch/2)], ydata = y[int(beg_ch/2):int(end_ch/2)], f =gauss_const, p0= p_0)
        if func is None:
            if label:
                plt.plot(x, gauss_const(x, par[0], par[1], par[2], par[3]),label="gauss fit with offset",color="orange")
            else:
                plt.plot(x, gauss_const(x, par[0], par[1], par[2], par[3]),color="orange")

        else:
            if label:
                plt.plot(func(x), gauss_const(x, par[0], par[1], par[2], par[3]),label="gauss fit with offset",color="orange")
            else:
                plt.plot(x, gauss_const(x, par[0], par[1], par[2], par[3]),color="orange")

        print("offset")
        return par[2], par[1],par[3]
    
    else:
        p_0 = [500, 30, (end_ch+beg_ch)/2]
        par, cov= curve_fit(xdata = x[int(beg_ch/2):int(end_ch/2)], ydata = y[int(beg_ch/2):int(end_ch/2)], f =gauss, p0= p_0)
        if func is None:
            if label:
                plt.plot(x, gauss(x, par[0], par[1], par[2]),label="gauss fit",color="r")
            else:
                plt.plot(x, gauss(x, par[0], par[1], par[2]),color="r")

        else:
            if label:
                plt.plot(func(x), gauss(x, par[0], par[1], par[2]),label="gauss fit",color="r")
            else:
                plt.plot(func(x), gauss(x, par[0], par[1], par[2]),color="r")


        return par[2], par[1],0


