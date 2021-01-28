from mama_to_python import read_mama_1D, read_mama_2D
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from tabulate import tabulate
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

#Min and max considered gamma energies
E_g_min = 100.0 #keV
E_g_max = 10000.0 #keV

#Efficiency at 1.33 MeV, if use native efficiency scaling in MAMA
eps_LaBr = 0.2714
eps_LaBr_unc = 0.0004

#To check that statistical uncertainties behaves correctly with less data
test_scale_factor=1.0

#Number of fissions
F = float(133132/test_scale_factor) #252Cf
F_unc = np.sqrt(F) #Uncertainty in nr of fissions

#Only filling matrix in MAMA, unfolding in OMpy
M, C, y, x = read_mama_2D("28jan2021/252Cf_energy_labr_fission_all_unf_28jan2021.m")
M_delayed1, C_delayed1, y_delayed1, x_delayed1 = read_mama_2D("28jan2021/252Cf_energy_labr_fission_all_delayed1_unf_28jan2021.m")
M_delayed2, C_delayed2, y_delayed2, x_delayed2 = read_mama_2D("28jan2021/252Cf_energy_labr_fission_all_delayed2_unf_28jan2021.m")
M_delayed3, C_delayed3, y_delayed3, x_delayed3 = read_mama_2D("28jan2021/252Cf_energy_labr_fission_all_delayed3_unf_28jan2021.m")
M_delayed4, C_delayed4, y_delayed4, x_delayed4 = read_mama_2D("28jan2021/252Cf_energy_labr_fission_all_delayed4_unf_28jan2021.m")


M_unc, C_unc, y_unc, x_unc = read_mama_2D("28jan2021/252Cf_energy_labr_fission_mama_fnrn_ompy_unf_std_11jun2020.m")

#Modifying because OMpy gives it two excitaton energy bins
M = M[0]
M_unc = M_unc[0]

##########################################
###  Plot data before any corrections  ###
##########################################


##########################################
### Calculate Spectral Characteristics ###
##########################################

#Create arrays to store multiplicities and total energies
M_g = 0.0
E_tot = 0.0

M_g_unc = 0.0
E_tot_unc = 0.0

#Calculating spectral char
for i in range(len(x)-1):
	if x[i] >= E_g_min and x[i] < E_g_max:

		M_g += M[i]/(eps_LaBr*test_scale_factor*F)
		E_tot += M[i]*x[i]/(eps_LaBr*test_scale_factor*F)

		M_g_unc += (M_unc[i]/(F*eps_LaBr))**2 + (M[i]*F_unc/(F**2*eps_LaBr))**2 + ((M[i]*eps_LaBr_unc)/(F*eps_LaBr**2))**2
		E_tot_unc += (M_unc[i]*x[i]/(F*eps_LaBr))**2 + (M[i]*F_unc*x[i]/(F**2*eps_LaBr))**2 + ((M[i]*eps_LaBr_unc*x[i])/(F*eps_LaBr**2))**2

M_g_unc = np.sqrt(M_g_unc)
E_tot_unc = np.sqrt(E_tot_unc)

E_g = E_tot/M_g
E_g_unc = np.sqrt( (E_tot_unc/M_g)**2 + (E_tot*M_g_unc/(M_g**2))**2 )

################################################
### Rebin Ex and Egamma, variable Eg-binning ###
################################################

edge_list = np.genfromtxt("Variable_binwidth/edge_list_252Cf.dat")
bin_middle_list = []

M_variable_bins = np.zeros(len(edge_list)-1)
M_variable_bins_unc = np.zeros(len(edge_list)-1)

#Rebin
for i in range(len(edge_list)-1):
	bin_sum = 0.0
	bin_sum_unc = 0.0

	for j in range(len(x)-1):
		energy = x[j]

		if energy >= edge_list[i] and energy < edge_list[i+1]:
			bin_sum += M[j]
			bin_sum_unc += M_unc[j]**2 

	#Scale to counts/[MeV fission]
	bin_width = edge_list[i+1] - edge_list[i]
	bin_middle = edge_list[i] + bin_width/2.0
	bin_middle_list.append(bin_middle)
	MeV_scale_factor = 1000.0/bin_width

	M_variable_bins[i] += bin_sum*MeV_scale_factor/(float(F)*eps_LaBr)
	#Remember: right now is bin_sum_unc the uncertainty squared! Thus np.sqrt(bin_sum_unc) in line below
	M_variable_bins_unc[i] += ((MeV_scale_factor*np.sqrt(bin_sum_unc))/(F*eps_LaBr))**2 + ((MeV_scale_factor*bin_sum*F_unc)/(F**2*eps_LaBr))**2 + ((MeV_scale_factor*bin_sum*eps_LaBr_unc)/(F*eps_LaBr**2))**2

M_variable_bins_unc = np.sqrt(M_variable_bins_unc)

#Should not be any negative counts left now: remove them if that is the case
for i in range(len(edge_list)-1):
	if M_variable_bins[i] < 0:
		M_variable_bins[i] = 0
		M_variable_bins_unc[i] = 0

#Chech difference no negative bins has for PFG char

#Create arrays to store multiplicities and total energies
M_g_varbin = 0.0
E_tot_varbin = 0.0
E_g_varbin = 0.0

M_g_varbin_unc = 0.0
E_tot_varbin_unc = 0.0
E_g_varbin_unc = 0.0

#Calculating spectral char with variable bins that have been put to 0 if negative
for i in range(len(edge_list)-1):
	#if edge_list[i] >= E_g_min and edge_list[i] < E_g_max:

	bin_width = edge_list[i+1] - edge_list[i]
	MeV_scale_factor = 1000.0/bin_width

	M_g_varbin += M_variable_bins[i]/MeV_scale_factor
	E_tot_varbin += M_variable_bins[i]*(edge_list[i]+bin_width/float(2.0))/MeV_scale_factor

	M_g_varbin_unc += (M_variable_bins_unc[i]/MeV_scale_factor)**2
	E_tot_varbin_unc += (M_variable_bins_unc[i]*(edge_list[i]+bin_width/float(2.0))/(MeV_scale_factor))**2


M_g_varbin_unc = np.sqrt(M_g_varbin_unc)
E_tot_varbin_unc = np.sqrt(E_tot_varbin_unc)

E_g_varbin = E_tot_varbin/M_g_varbin
E_g_varbin_unc = np.sqrt( (E_tot_varbin_unc/M_g_varbin)**2 + (E_tot_varbin*M_g_varbin_unc/(M_g_varbin**2))**2 )

###############################
### Import previous results ###
###############################

Verbinski = np.genfromtxt("Verbinski.dat", usecols=(0,1))
Billnert = np.genfromtxt("Billnert.dat", usecols=(0,1))
Oberstedt_LaCl = np.genfromtxt("Oberstedt_Cf252/Oberstedt2015_LaCl3.dat", usecols=(0,1,2))
Oberstedt_LaBr = np.genfromtxt("Oberstedt_Cf252/Oberstedt2015_LaBr3.dat", usecols=(0,1,2))
Oberstedt_LaBr_2 = np.genfromtxt("Oberstedt_Cf252/Oberstedt2015_LaBr3_2.dat", usecols=(0,1,2))
Oberstedt_LaBr_3 = np.genfromtxt("Oberstedt_Cf252/Oberstedt2015_LaBr3_3.dat", usecols=(0,1,2))

#####################################
###     Bin-by-bin corr factor    ###
#####################################

#Billnert correction
Billnert_correction = np.zeros(len(M_variable_bins))

for j in range(len(M_variable_bins)):

	counter = 0
	sum_correction = 0

	for i in range(len(Billnert[:,1])):
		E = Billnert[:,0][i]

		if E >= edge_list[j]/1000.0 and E < edge_list[j+1]/1000.0 and Billnert[:,1][i] > 0 and M_variable_bins[j] > 0:
			sum_correction += Billnert[:,1][i]/float(M_variable_bins[j])
			counter += 1

		if sum_correction > 0:
			Billnert_correction[j] = sum_correction/float(counter)
		else:
			Billnert_correction[j] = 1.0


#Oberstedt_LaBr correction
Oberstedt_LaBr_correction = np.zeros(len(M_variable_bins))

for j in range(len(M_variable_bins)):

	counter = 0
	sum_correction = 0

	for i in range(len(Oberstedt_LaBr[:,1])):
		E = Oberstedt_LaBr[:,0][i]

		if E >= edge_list[j]/1000.0 and E < edge_list[j+1]/1000.0 and Oberstedt_LaBr[:,1][i] > 0 and M_variable_bins[j] > 0:
			sum_correction += Oberstedt_LaBr[:,1][i]/float(M_variable_bins[j])
			counter += 1

		if sum_correction > 0:
			Oberstedt_LaBr_correction[j] = sum_correction/float(counter)
		else:
			Oberstedt_LaBr_correction[j] = 1.0


#Oberstedt_LaBr_2 correction
Oberstedt_LaBr_2_correction = np.zeros(len(M_variable_bins))

for j in range(len(M_variable_bins)):

	counter = 0
	sum_correction = 0

	for i in range(len(Oberstedt_LaBr_2[:,1])):
		E = Oberstedt_LaBr_2[:,0][i]

		if E >= edge_list[j]/1000.0 and E < edge_list[j+1]/1000.0 and Oberstedt_LaBr_2[:,1][i] > 0 and M_variable_bins[j] > 0:
			sum_correction += Oberstedt_LaBr_2[:,1][i]/float(M_variable_bins[j])
			counter += 1

		if sum_correction > 0:
			Oberstedt_LaBr_2_correction[j] = sum_correction/float(counter)
		else:
			Oberstedt_LaBr_2_correction[j] = 1.0

#Oberstedt_LaBr_2 correction
Oberstedt_LaBr_3_correction = np.zeros(len(M_variable_bins))

for j in range(len(M_variable_bins)):

	counter = 0
	sum_correction = 0

	for i in range(len(Oberstedt_LaBr_3[:,1])):
		E = Oberstedt_LaBr_3[:,0][i]

		if E >= edge_list[j]/1000.0 and E < edge_list[j+1]/1000.0 and Oberstedt_LaBr_3[:,1][i] > 0 and M_variable_bins[j] > 0:
			sum_correction += Oberstedt_LaBr_3[:,1][i]/float(M_variable_bins[j])
			counter += 1

		if sum_correction > 0:
			Oberstedt_LaBr_3_correction[j] = sum_correction/float(counter)
		else:
			Oberstedt_LaBr_3_correction[j] = 1.0

#Take average of all correction factors
average_correction = (Billnert_correction + Oberstedt_LaBr_correction + Oberstedt_LaBr_2_correction + Oberstedt_LaBr_3_correction)/float(4)

#np.savetxt("252Cf_correction/252Cf_PFGS_correction.dat", [np.array(edge_list[0:-1]), average_correction])


#Correction range
E_corr_min = 100.0
E_corr_max = 700.0

M_variable_bins_corrected = np.zeros(len(M_variable_bins))
M_variable_bins_corrected_unc = np.zeros(len(M_variable_bins))

for i in range(len(M_variable_bins)):
	E = bin_middle_list[i]
	if E > E_corr_min and E < E_corr_max:
		M_variable_bins_corrected[i] = M_variable_bins[i]*average_correction[i]
		M_variable_bins_corrected_unc[i] = M_variable_bins_unc[i]*average_correction[i]
	else:
		M_variable_bins_corrected[i] = M_variable_bins[i]
		M_variable_bins_corrected_unc[i] = M_variable_bins_unc[i]

#New PFG characteristics based on corrected spectrum

#Create arrays to store multiplicities and total energies
M_g_varbin_corrected = 0.0
E_tot_varbin_corrected = 0.0
E_g_varbin_corrected = 0.0

M_g_varbin_corrected_unc = 0.0
E_tot_varbin_corrected_unc = 0.0
E_g_varbin_corrected_unc = 0.0

#Calculating spectral char with variable bins that have been put to 0 if negative
for i in range(len(edge_list)-1):
	#if edge_list[i] >= E_g_min and edge_list[i] < E_g_max:

	bin_width = edge_list[i+1] - edge_list[i]
	MeV_scale_factor = 1000.0/bin_width

	M_g_varbin_corrected += M_variable_bins_corrected[i]/MeV_scale_factor
	E_tot_varbin_corrected += M_variable_bins_corrected[i]*(edge_list[i]+bin_width/float(2.0))/MeV_scale_factor

	M_g_varbin_corrected_unc += (M_variable_bins_corrected_unc[i]/MeV_scale_factor)**2
	E_tot_varbin_corrected_unc += (M_variable_bins_corrected_unc[i]*(edge_list[i]+bin_width/float(2.0))/(MeV_scale_factor))**2

M_g_varbin_corrected_unc = np.sqrt(M_g_varbin_corrected_unc)
E_tot_varbin_corrected_unc = np.sqrt(E_tot_varbin_corrected_unc)

E_g_varbin_corrected = E_tot_varbin_corrected/M_g_varbin_corrected
E_g_varbin_corrected_unc = np.sqrt( (E_tot_varbin_unc/M_g_varbin)**2 + (E_tot_varbin*M_g_varbin_unc/(M_g_varbin**2))**2 )


#####################################
###  Plot PFG spectrum and char   ###
#####################################

#Uncertainty in OMpy spectrum
#M_err = np.sqrt( ((OMpy_MeV_scale_factor*OMpy[:,2])/(F*eps_LaBr))**2 + ((OMpy_MeV_scale_factor*OMpy[:,1]*F_unc)/(F**2*eps_LaBr))**2 + ((OMpy_MeV_scale_factor*OMpy[:,1]*eps_LaBr_unc)/(F*eps_LaBr**2))**2 )

xticks_photonspectrum = ["0"," ","1"," ","2", " ","3", " ","4", " ","5", " ","6", " ","7", " ","8", " "]
xticks_arr = np.arange(0,9000,500)
yticks_arr = np.array([10**(-4), 10**(-3), 10**(-2), 10**(-1), 1, 10])

#Spectrum
fig, ax = plt.subplots()
ax.errorbar(bin_middle_list, M_variable_bins, yerr=M_variable_bins_unc, label="This work", color="r")
#plt.errorbar(bin_middle_list, M_variable_bins_corrected, yerr=M_variable_bins_corrected_unc, label="This work, corrected")
plt.plot(Verbinski[:,0]*1000, Verbinski[:,1], label="Verbinski et al. 1973", color="b")
ax.plot(Billnert[:,0]*1000, Billnert[:,1], label="Billnert et al. 2013", color="darkorange")
#plt.errorbar(Oberstedt_LaCl[:,0]*1000, Oberstedt_LaCl[:,1], yerr=Oberstedt_LaCl[:,2], label="LaCl3:Ce (SEB 347), Oberstedt et al. 2015")
ax.errorbar(Oberstedt_LaBr[:,0]*1000, Oberstedt_LaBr[:,1], yerr=Oberstedt_LaBr[:,2], label="Oberstedt et al. 2015", color="g")
#plt.errorbar(Oberstedt_LaBr_2[:,0]*1000, Oberstedt_LaBr_2[:,1], yerr=Oberstedt_LaBr_2[:,2], label="LaBr3:Ce (Q491), Oberstedt et al. 2015")
#plt.errorbar(Oberstedt_LaBr_3[:,0]*1000, Oberstedt_LaBr_3[:,1], yerr=Oberstedt_LaBr_3[:,2], label="LaBr3:Ce (2987), Oberstedt et al. 2015")
ax.set_yscale('log', nonposy='clip')
ax.set_ylabel("Photons/(Fission MeV)", fontsize=13, fontweight='bold')
ax.set_xlabel("E$_{\gamma}$ [MeV]", fontsize=13, fontweight='bold')
ax.set_xlim(0,8000)
ax.set_ylim(10**(-4),15)
#ax.set_xlim(100,700)
#ax.set_ylim(10**(0),15)
ax.tick_params(length=7, width=1, which="major")
ax.tick_params(length=4, width=1, which="minor")
#ax.axis([0,8,10**(-4),100 ])
plt.legend(bbox_to_anchor = [0.65, 0.6], fontsize=13, frameon=False)
#plt.xticks(xticks_arr, xticks_photonspectrum, fontsize=13)
#plt.yticks(yticks_arr, fontsize=13)
#plt.show()

#fig.savefig("252Cf_spec.pdf", bbox_inches='tight')

bin_middle_list = np.array(bin_middle_list)


# fig, (ax1,ax2) = plt.subplots(2)
# ax1.plot(Verbinski[:,0], Verbinski[:,1], label="Verbinski et al. 1973")
# ax1.plot(Billnert[:,0], Billnert[:,1], label="Billnert et al. 2013")
# ax1.errorbar(Oberstedt_LaBr[:,0], Oberstedt_LaBr[:,1], yerr=Oberstedt_LaBr[:,2], label="Oberstedt et al. 2015")
# ax1.errorbar(bin_middle_list/1000.0, M_variable_bins, yerr=M_variable_bins_unc, label="This work")
# ax1.set_yscale('log', nonposy='clip')
# ax1.grid(True)
# #ax1.set_ylabel("Photons/(Fission MeV)", fontsize=13, fontweight='bold')
# ax1.legend(loc="lower left", fontsize=18, frameon=False)
# ax1.tick_params(axis="x", which="major", labelsize=18)
# ax1.tick_params(axis="y", which="major", labelsize=18)
# ax1.set_xlim(0,8)
# ax1.set_ylim(5*10**(-4),20)

# ax2.plot(Verbinski[:,0], Verbinski[:,1], label="Verbinski et al. 1973")
# ax2.plot(Billnert[:,0], Billnert[:,1], label="Billnert et al. 2013")
# ax2.errorbar(Oberstedt_LaBr[:,0], Oberstedt_LaBr[:,1], yerr=Oberstedt_LaBr[:,2], label="Oberstedt et al. 2015")
# ax2.errorbar(bin_middle_list/1000.0, M_variable_bins, yerr=M_variable_bins_unc, label="This work")
# ax2.set_yscale('log', nonposy='clip')
# ax2.tick_params(axis="x", which="major", labelsize=18)
# ax2.tick_params(axis="y", which="major", labelsize=18)
# ax2.set_xlim(0.1,0.7)
# ax2.set_ylim(1,15)
# ax2.set_xlabel("E$_{\gamma}$ [MeV]", fontsize=18)
# ax2.grid(True)

# fig.text(0.04, 0.5, "Photons/(Fission MeV)", fontsize=18, va='center', rotation='vertical')
# plt.subplots_adjust(hspace=0.12)
# plt.show()


#Table of results compared to other results
l = [ \
["This work, linbin", round(E_tot/1000.0,2), round(E_tot_unc/1000,3), round(M_g,2), round(M_g_unc,3) , round(E_g/1000.0,2), round(E_g_unc/1000.0,2), "%.2f - %.2f" % (E_g_min/1000.0, E_g_max/1000.0)], \
["This work, varbin, no neg counts", round(E_tot_varbin/1000.0,2), round(E_tot_varbin_unc/1000,3), round(M_g_varbin,2), round(M_g_varbin_unc,3) , round(E_g_varbin/1000.0,2), round(E_g_varbin_unc/1000.0,2), "%.2f - %.2f" % (E_g_min/1000.0, E_g_max/1000.0)], \
#["This work, varbin, 252Cf corrected", round(E_tot_varbin_corrected/1000.0,2), round(E_tot_varbin_corrected_unc/1000,3), round(M_g_varbin_corrected,2), round(M_g_varbin_corrected_unc,3) , round(E_g_varbin_corrected/1000.0,2), round(E_g_varbin_corrected_unc/1000.0,2), "%.2f - %.2f" % (E_g_min/1000.0, E_g_max/1000.0)], \
["Billnert_2013", "6.64", "0.08","8.30", "0.08", "0.80", "0.01", "0.1-7.2"], \
["Verbinski_1973", "6.84", "0.30", "7.80", "0.30", "0.88", "0.04", "0.14-10"], \
#["Oberstedt_2015 LaCl3:Ce (SEB 347)", "6.81", "0.14", "8.28", "0.44", "0.82", "0.05", "0.1-6.0"],\
["Oberstedt_2015 LaBr3:Ce (Q489)", "6.74", "0.09", "8.29", "0.07", "0.81", "0.01", "0.1-6.4"], \
["Oberstedt_2015 LaBr3:Ce (Q491)", "6.76", "0.09", "8.28", "0.08", "0.82", " 0.01", "0.1-6.5"], \
["Oberstedt_2015 LaBr3:Ce (2987)", "6.51", "0.07", "8.28", "0.07", "0.79", "0.01", "0.1-7.7"], \
#["Oberstedt_2018", "6.51", "0.76", "8.28", "0.51", "0.79", "0.10", " "]
]
table = tabulate(l, headers=['Results', 'E_tot', "unc", 'M_g', "unc", "E_g ", "unc" , "E Range [MeV]"], tablefmt='orgtbl')
#print(table)

# #Calculation of correction factors on E_tot and M_g
# l_2 = [ \
# ["Billnert_2013", 6.64/(E_tot/float(1000)), 8.30/(M_g)], \
# ["Oberstedt_2015 LaBr3:Ce (Q489)", 6.74/(E_tot/float(1000)), 8.29/(M_g)], \
# ["Oberstedt_2015 LaBr3:Ce (Q491)", 6.76/(E_tot/float(1000)), 8.28/(M_g)], \
# ["Oberstedt_2015 LaBr3:Ce (2987)", 6.51/(E_tot/float(1000)), 8.28/(M_g)], \
# ]
# table_2 = tabulate(l_2, headers=['Correction factor', 'E_tot', 'M_g'], tablefmt='orgtbl')
# #print(table_2)

#Calculation of weighted correction factors on E_tot and M_g

def energy_flat(channel, Px, Py):
    return Px*channel + Py


E_tot_prev_exp_weighted = np.average(np.array([6.64, 6.74, 6.76, 6.51]), weights=1.0/(np.array([0.08, 0.09, 0.09, 0.07]))**2)
M_g_prev_exp_weighted = np.average(np.array([8.30, 8.29, 8.28, 8.28]), weights=1.0/(np.array([0.08, 0.07, 0.08, 0.07]))**2)

#print("E_tot_prev_exp_weighted: %.4f" % E_tot_prev_exp_weighted)
#print("M_g_prev_exp_weighted: %.4f" % M_g_prev_exp_weighted)









