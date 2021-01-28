import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit

try:
    import uncertainties.unumpy as unp
    import uncertainties as unc
except:
    import pip
    pip.main(['install', 'uncertainties'])
    import uncertainties.unumpy as unp
    import uncertainties as unc

sigma = np.genfromtxt("sigma_table.dat", skip_header=16, usecols=(1,2,3)) 

sigma_all = np.genfromtxt("sigma_table.dat", skip_header=2, usecols=(1,2,3), max_rows=11)

def sqrt_func(channel, a):
    #Constant term is minimum bin width squared, cannot rebin to smaller bin widths
    return a*np.sqrt(channel + 400)

channel_array = np.linspace(0,15000,15000)

P, cov = curve_fit(sqrt_func, sigma[:,0], sigma[:,1]*2.355, sigma=np.sqrt(sigma[:,2]), absolute_sigma=False)
P_err = unc.correlated_values(P, cov)


FWHM_err = sqrt_func(channel_array, P_err[0])
nom = unp.nominal_values(FWHM_err)
std = unp.std_devs(FWHM_err)

plt.plot(sigma_all[:,0], sigma_all[:,1]*2.355, "g.", label="All peaks")
plt.plot(sigma[:,0], sigma[:,1]*2.355, "r.", label="Sharpest peaks")
plt.plot(channel_array, sqrt_func(channel_array, P[0]))
plt.plot(channel_array, nom - 1 * std, c='orange', label='{}*68% Confidence Region Linear'.format(1))
plt.plot(channel_array, nom + 1 * std, c='orange')
plt.xlabel("Energy of peaks [keV]", fontsize=14)
plt.ylabel("FWHM of peak [keV]", fontsize=14)
plt.legend(fontsize=14)
plt.grid()
plt.show()

#bin_number_array = np.linspace(0,10,11)
#print(sqrt_func(100000, P[0]))

bin_edge = 0.0
edge_list = []

while bin_edge < 15000:
	edge_list.append(bin_edge)
	bin_width = sqrt_func(bin_edge, P[0])
	bin_edge += bin_width

#print(np.around(edge_list,3))

np.savetxt("edge_list.dat", np.around(edge_list,3))





