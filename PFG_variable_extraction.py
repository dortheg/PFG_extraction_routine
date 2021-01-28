from mama_to_python import read_mama_2D, read_mama_1D
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from tabulate import tabulate
import pandas as pd
import scipy.interpolate as inp
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter

try:
    import uncertainties.unumpy as unp
    import uncertainties as unc
except:
    import pip
    pip.main(['install', 'uncertainties'])
    import uncertainties.unumpy as unp
    import uncertainties as unc

#This script has variable binning for gamma energies and constant for Ex

# #Only filling matrix in MAMA, unfolding in OMpy
# M, C, y, x = read_mama_2D("29sep2020_241Pu_final/Pu_exgam_ppac_mama_fnrn_ompy_unf_29sep2020.m")
# M_unc, C_unc, y_unc, x_unc = read_mama_2D("29sep2020_241Pu_final/Pu_exgam_ppac_mama_fnrn_ompy_unf_std_29sep2020.m")
# #Extract number of fissions data
# Ex_F = np.genfromtxt("29sep2020_241Pu_final/Pu_number_of_fissions_29sep2020.dat", skip_header=1, usecols=0)
# F = np.genfromtxt("29sep2020_241Pu_final/Pu_number_of_fissions_29sep2020.dat", skip_header=1, usecols=1)
# F_all = np.genfromtxt("29sep2020_241Pu_final/Pu_number_of_fissions_all_29sep2020.dat", skip_header=1, usecols=1)
# F_bg = np.genfromtxt("29sep2020_241Pu_final/Pu_number_of_fissions_bg_29sep2020.dat", skip_header=1, usecols=1)

#Only filling matrix in MAMA, unfolding in OMpy
M, C, y, x = read_mama_2D("14des2020_241Pu_final/Pu_exgam_ppac_mama_fnrn_ompy_unf_14des2020.m")
M_unc, C_unc, y_unc, x_unc = read_mama_2D("14des2020_241Pu_final/Pu_exgam_ppac_mama_fnrn_ompy_unf_std_14des2020.m")
#Extract number of fissions data
Ex_F = np.genfromtxt("14des2020_241Pu_final/Pu_number_of_fissions_14des2020.dat", skip_header=1, usecols=0)
F = np.genfromtxt("14des2020_241Pu_final/Pu_number_of_fissions_14des2020.dat", skip_header=1, usecols=1)
F_all = np.genfromtxt("14des2020_241Pu_final/Pu_number_of_fissions_all_14des2020.dat", skip_header=1, usecols=1)
F_bg = np.genfromtxt("14des2020_241Pu_final/Pu_number_of_fissions_bg_14des2020.dat", skip_header=1, usecols=1)

eps_LaBr = 0.2714
eps_LaBr_unc = 0.0004
FillNeg = False

E_g_min = 100.0 #keV
E_g_max = 10000.0 #keV

#To check that statistical uncertainties behaves correctly with less data
test_scale_factor=1.0

#Rebin number of fissions to match Ex of exgam-matrix, should still be linear
binwidth_ex = y[2]-y[1]
F_rebinned = np.zeros(len(y))
F_all_rebinned = np.zeros(len(y))
F_bg_rebinned = np.zeros(len(y))

for i in range(len(Ex_F)):
	ex_bin = int(Ex_F[i]//binwidth_ex)
	F_rebinned[ex_bin] += F[i]
	F_all_rebinned[ex_bin] += F_all[i]
	F_bg_rebinned[ex_bin] += F_bg[i]

F = F_rebinned
F_all = F_all_rebinned
F_bg = F_bg_rebinned


################################################
### Rebin Ex and Egamma, variable Eg-binning ###
################################################

Ex_bin_width = 500.0 #keV
Ex_array = np.arange(0,15000, Ex_bin_width) #Creates an# array of evenly spaced values with difference Ex_bin_width, from 0 to 15000
Ex_bin_middle = Ex_array + Ex_bin_width/2.0
Ex_nbins = len(Ex_array)

#Get edges of the G-array from variable bin width calculation
G_array = np.genfromtxt("Variable_binwidth/edge_list_241Pu.dat")
G_nbins = len(G_array)-1

G_array_middle = []

for i in range(G_nbins):
	bin_width = G_array[i+1] - G_array[i]
	bin_middle = G_array[i] + bin_width/2.0
	G_array_middle.append(bin_middle)

#The rebinned matrices/arrays
M_rebinned = np.zeros((Ex_nbins,G_nbins))
M_unc_rebinned = np.zeros((Ex_nbins,G_nbins))

F_rebinned = np.zeros(Ex_nbins)
F_all_rebinned = np.zeros(Ex_nbins)
F_bg_rebinned = np.zeros(Ex_nbins)

for k in range(len(y)):
	ex_bin = int(y[k]//Ex_bin_width)
	F_rebinned[ex_bin] += F[k]
	F_all_rebinned[ex_bin] += F_all[k]
	F_bg_rebinned[ex_bin] += F_bg[k]

	for i in range(G_nbins):
		bin_sum = 0
		bin_sum_unc = 0

		for j in range(len(x)):
			energy = x[j]

			if energy >= G_array[i] and energy < G_array[i+1]:
				bin_sum += M[k][j]
				bin_sum_unc += M_unc[k][j]**2

		#Must have += here, because more than one ex-bin, else overwrites stuff
		M_rebinned[ex_bin][i] += bin_sum
		M_unc_rebinned[ex_bin][i] += bin_sum_unc

M = M_rebinned
M_unc = np.sqrt(M_unc_rebinned)
F = F_rebinned
F_all = F_all_rebinned
F_bg = F_bg_rebinned

##########################################
### Counting Negatives After Rebinning ###
##########################################
neg_counts = np.zeros((Ex_nbins, G_nbins))

for i in range(len(Ex_array)):
	for j in range(G_nbins):
		if M[i][j] < 0:
			neg_counts[i][j] += abs(M[i][j])
			#If removing the neg counts

			if FillNeg == True:
				M[i][j] = 0

# plt.figure(3)
# for i in range(11,17):
# 	plt.plot(G_array_middle, neg_counts[i], label="Neg counts 241Pu E$_{x}$ = %.0f - %.0f keV" % (Ex_array[i],Ex_array[i+1]))
# plt.grid()
# plt.ylabel("Negative counts", fontsize=16)
# plt.xlabel("E$_{\gamma}$ [keV]", fontsize=16)
# plt.legend()
# plt.show()

##########################################
### Calculate Spectral Characteristics ###
##########################################

#Create arrays to store multiplicities and total energies
M_g = np.zeros(len(Ex_array))
E_tot = np.zeros(len(Ex_array))

M_g_unc = np.zeros(len(Ex_array))
E_tot_unc = np.zeros(len(Ex_array))

#Extract spectral characteristics
for i in range(len(Ex_array)):
	F_unc = np.sqrt(F_all[i] + F_bg[i])

	if F[i] == 0:
		M_g[i] = 0
		E_tot[i] = 0
	else:
		for j in range(G_nbins):
			if G_array[j] >= E_g_min and G_array[j] < E_g_max:

				M_g[i] += M[i][j]/(eps_LaBr*test_scale_factor*F[i])
				E_tot[i] += M[i][j]*G_array_middle[j]/(eps_LaBr*test_scale_factor*F[i])
		
				M_g_unc[i] += (M_unc[i][j]/(F[i]*eps_LaBr))**2 + (M[i][j]*F_unc/(F[i]**2*eps_LaBr))**2 + ((M[i][j]*eps_LaBr_unc)/(F[i]*eps_LaBr**2))**2
				E_tot_unc[i] += (M_unc[i][j]*G_array_middle[j]/(F[i]*eps_LaBr))**2 + (M[i][j]*F_unc*G_array_middle[j]/(F[i]**2*eps_LaBr))**2 + ((M[i][j]*eps_LaBr_unc*G_array_middle[j])/(F[i]*eps_LaBr**2))**2

M_g_unc = np.sqrt(M_g_unc)
E_tot_unc = np.sqrt(E_tot_unc)

#Calculate E_g
E_g = np.zeros(len(Ex_array))
E_g_unc = np.zeros(len(Ex_array))


for i in range(len(Ex_array)):
	F_unc = np.sqrt(F_all[i] + F_bg[i])

	if M_g[i] == 0:
		E_g[i] = 0
	else:
		E_g[i] = E_tot[i]/M_g[i]
		E_g_unc[i] = np.sqrt( (E_tot_unc[i]/M_g[i])**2 + ( E_tot[i]*M_g_unc[i]/(M_g[i]**2))**2)

###########################################
### Rescale M to [counts/(MeV fission)] ###
###########################################

M_rescaled = np.zeros((Ex_nbins,G_nbins))
M_unc_rescaled = np.zeros((Ex_nbins,G_nbins))

for i in range(len(Ex_array)):
	F_unc = np.sqrt(F_all[i] + F_bg[i])
	for j in range(G_nbins):
		bin_width = G_array[j+1] - G_array[j]
		MeV_scale_factor_variable_bin = 1000.0/bin_width
		if F[i] == 0:
			M_rescaled[i][j] = 0
			M_unc_rescaled[i][j] = 0
		else:			
			M_rescaled[i][j] = M[i][j]*MeV_scale_factor_variable_bin/(float(F[i])*eps_LaBr)
			M_unc_rescaled[i][j] = np.sqrt( ((MeV_scale_factor_variable_bin*M_unc[i][j])/(F[i]*eps_LaBr))**2 + ((MeV_scale_factor_variable_bin*M[i][j]*F_unc)/(F[i]**2*eps_LaBr))**2 + ((MeV_scale_factor_variable_bin*M[i][j]*eps_LaBr_unc)/(F[i]*eps_LaBr**2))**2 )

M = M_rescaled
M_unc = M_unc_rescaled

##############################################
### 		Use 252Cf PFGS to correct      ###
##############################################

PFGS_252Cf_corr = np.loadtxt("252Cf_correction/252Cf_PFGS_correction.dat")

M_rescaled_corrected = np.zeros((Ex_nbins,G_nbins))
M_unc_rescaled_corrected = np.zeros((Ex_nbins,G_nbins))

for j in range(len(PFGS_252Cf_corr[0])-1):

	for i in range(len(G_array)-1):
		E = G_array[i]
		if E >= PFGS_252Cf_corr[0][j] and E < PFGS_252Cf_corr[0][j+1]:

			for k in range(len(Ex_array)-1):
				M_rescaled_corrected[k][i] = M_rescaled[k][i]*PFGS_252Cf_corr[1][j]
				M_unc_rescaled_corrected[k][i] = M_unc_rescaled[k][i]*PFGS_252Cf_corr[1][j]

		else:
			M_rescaled_corrected[k][i] = M_rescaled[k][i]
			M_unc_rescaled_corrected[k][i] = M_unc_rescaled[k][i]


#Create arrays to store multiplicities and total energies
M_g_PFGS_corr = np.zeros(len(Ex_array))
E_tot_PFGS_corr = np.zeros(len(Ex_array))
E_g_PFGS_corr = np.zeros(len(Ex_array))

M_g_PFGS_corr_unc = np.zeros(len(Ex_array))
E_tot_PFGS_corr_unc = np.zeros(len(Ex_array))
E_g_PFGS_corr_unc = np.zeros(len(Ex_array))

for k in range(len(Ex_array)):
	for i in range(len(G_array)-1):

		bin_width = G_array[i+1] - G_array[i]
		MeV_scale_factor = 1000.0/bin_width

		M_g_PFGS_corr[k] += M_rescaled_corrected[k][i]/MeV_scale_factor
		E_tot_PFGS_corr[k] += M_rescaled_corrected[k][i]*(G_array[i]+bin_width/float(2.0))/MeV_scale_factor

		M_g_PFGS_corr_unc[k] += (M_unc_rescaled_corrected[k][i]/MeV_scale_factor)**2
		E_tot_PFGS_corr_unc[k] += (M_unc_rescaled_corrected[k][i]*(G_array[i]+bin_width/float(2.0))/(MeV_scale_factor))**2

	M_g_PFGS_corr_unc[k] = np.sqrt(M_g_PFGS_corr_unc[k])
	E_tot_PFGS_corr_unc[k] = np.sqrt(E_tot_PFGS_corr_unc[k])

	if M_g_PFGS_corr[k] != 0:
		E_g_PFGS_corr[k] = E_tot_PFGS_corr[k]/M_g_PFGS_corr[k]
		E_g_PFGS_corr_unc[k] = np.sqrt( (E_tot_PFGS_corr_unc[k]/M_g_PFGS_corr[k])**2 + (E_tot_PFGS_corr[k]*M_g_PFGS_corr_unc[k]/(M_g_PFGS_corr[k]**2))**2 )

	else:
		E_g_PFGS_corr[k] = 0
		E_g_PFGS_corr_unc[k] = 0


#####################################
### Linear regression on PFG_char ###
#####################################

def energy_lin(channel, Px, Py):
    return Px*channel + Py

P_M_g, cov_M_g = curve_fit(energy_lin, Ex_bin_middle[11:17]/1000.0, M_g[11:17], sigma=M_g_unc[11:17], absolute_sigma=True) #Might be that it should be True
P_M_g_err = unc.correlated_values(P_M_g, cov_M_g)

P_E_tot, cov_E_tot = curve_fit(energy_lin, Ex_bin_middle[11:17]/1000.0, E_tot[11:17]/1000.0, sigma=E_tot_unc[11:17]/1000.0, absolute_sigma=True) #Might be that it should be True
P_E_tot_err = unc.correlated_values(P_E_tot, cov_E_tot)

P_E_g, cov_E_g = curve_fit(energy_lin, Ex_bin_middle[11:17]/1000.0, E_g[11:17]/1000.0, sigma=E_g_unc[11:17]/1000.0, absolute_sigma=True) #Might be that it should be True
P_E_g_err = unc.correlated_values(P_E_g, cov_E_g)

# print("\n")
# print("M_g linear fit slope: %.6e pm %.6e" % (unp.nominal_values(P_M_g_err[0]), unp.std_devs(P_M_g_err[0]) ))
# print("E_tot linear fit slope: %.6e pm %.6e" % (unp.nominal_values(P_E_tot_err[0]), unp.std_devs(P_E_tot_err[0]) ))
# print("E_g linear fit slope: %.6e pm %.6e" % (unp.nominal_values(P_E_g_err[0]), unp.std_devs(P_E_g_err[0]) ))

##########################################
###    Use 252Cf PFG char to correct   ###
##########################################

#Cf252_Etot_Mg_corr = np.genfromtxt("29sep2020_241Pu_final/252Cf_Etot_Mg_correction_factors.dat", skip_header=1, usecols=(1,2))
E_tot_corr_factor = 6.6410/6.18
M_g_corr_factor = 8.2872/6.37

#Create arrays to store the 252Cf-factor corrected PFG char
M_g_corr_by_factor = np.zeros(len(Ex_array))
E_tot_corr_by_factor = np.zeros(len(Ex_array))
E_g_corr_by_factor = np.zeros(len(Ex_array))

M_g_unc_corr_by_factor = np.zeros(len(Ex_array))
E_tot_unc_corr_by_factor = np.zeros(len(Ex_array))
E_g_unc_corr_by_factor = np.zeros(len(Ex_array))

#Correct by constant factor
E_tot_corr_by_factor = E_tot*E_tot_corr_factor
M_g_corr_by_factor = M_g*M_g_corr_factor

E_tot_unc_corr_by_factor = E_tot_unc*E_tot_corr_factor
M_g_unc_corr_by_factor = M_g_unc*M_g_corr_factor

for i in range(len(Ex_array)):
	if M_g_corr_by_factor[i] != 0:
		E_g_corr_by_factor[i] = E_tot_corr_by_factor[i]/float(M_g_corr_by_factor[i])
		E_g_unc_corr_by_factor[i] = np.sqrt( (E_tot_unc_corr_by_factor[i]/M_g_corr_by_factor[i])**2 + ( E_tot_corr_by_factor[i]*M_g_unc_corr_by_factor[i]/(M_g[i]**2))**2)
	else:
		E_g_corr_by_factor[i] = 0
		E_g_unc_corr_by_factor[i] = 0		

###############################################
### Linear regression on corrected PFG_char ###
###############################################

P_M_g_corr_by_factor, cov_M_g_corr_by_factor = curve_fit(energy_lin, Ex_bin_middle[11:17]/1000.0, M_g_corr_by_factor[11:17], sigma=M_g_unc_corr_by_factor[11:17], absolute_sigma=True) #Might be that it should be True
P_M_g_err_corr_by_factor = unc.correlated_values(P_M_g_corr_by_factor, cov_M_g_corr_by_factor)

P_E_tot_corr_by_factor, cov_E_tot_corr_by_factor = curve_fit(energy_lin, Ex_bin_middle[11:17]/1000.0, E_tot_corr_by_factor[11:17]/1000.0, sigma=E_tot_unc_corr_by_factor[11:17]/1000.0, absolute_sigma=True) #Might be that it should be True
P_E_tot_err_corr_by_factor = unc.correlated_values(P_E_tot_corr_by_factor, cov_E_tot_corr_by_factor)

P_E_g_corr_by_factor, cov_E_g_corr_by_factor = curve_fit(energy_lin, Ex_bin_middle[11:17]/1000.0, E_g_corr_by_factor[11:17]/1000.0, sigma=E_g_unc_corr_by_factor[11:17]/1000.0, absolute_sigma=True) #Might be that it should be True
P_E_g_err_corr_by_factor = unc.correlated_values(P_E_g_corr_by_factor, cov_E_g_corr_by_factor)

# print("\n")
# print("M_g_corr_by_factor linear fit slope: %.6e pm %.6e" % (unp.nominal_values(P_M_g_err_corr_by_factor[0]), unp.std_devs(P_M_g_err_corr_by_factor[0]) ))
# print("E_tot_corr_by_factor linear fit slope: %.6e pm %.6e" % (unp.nominal_values(P_E_tot_err_corr_by_factor[0]), unp.std_devs(P_E_tot_err_corr_by_factor[0]) ))
# print("E_g_corr_by_factor linear fit slope: %.6e pm %.6e" % (unp.nominal_values(P_E_g_err_corr_by_factor[0]), unp.std_devs(P_E_g_err_corr_by_factor[0]) ))
# print("\n")

#####################################
###    Import FREYA simulations   ###
#####################################

data_freya_100 = np.loadtxt('FREYA_gmin=100_tmax=3ns_31aug2020/data_as_func_of_excitation_energy.dat.unchanged', skiprows=0, usecols=(0,1,2,3,4,5,6,7,8,9,10))
photonspectrum_freya_6_75_100 = np.genfromtxt("FREYA_gmin=100_tmax=3ns_31aug2020/photon_spectrum_Ex_6_75MeV.dat", skip_header=1, usecols=(0,1))
N_fissions_FREYA = 1000000.0
photonspectrum_freya_6_75_100_unc = np.sqrt(np.genfromtxt("FREYA_gmin=100_tmax=3ns_31aug2020/photon_spectrum_Ex_6_75MeV.dat", skip_header=1, usecols=(1))*N_fissions_FREYA)/N_fissions_FREYA

Ex_freya_100 = data_freya_100[:,4]

avg_ph_mult_freya_100 = data_freya_100[:,5]
avg_ph_energy_freya_100 = data_freya_100[:,7]
total_ph_E_freya_100 = data_freya_100[:,9]

FREYA_MeV_scale_factor_100 = 1000/(photonspectrum_freya_6_75_100[101][0]*1000-photonspectrum_freya_6_75_100[100][0]*1000)

# data_freya_500 = np.loadtxt('FREYA_gmin=500_tmax=3ns_16sep2020/data_as_func_of_excitation_energy.dat.unchanged', skiprows=0, usecols=(0,1,2,3,4,5,6,7,8,9,10))
# photonspectrum_freya_6_75_500 = np.genfromtxt("FREYA_gmin=500_tmax=3ns_16sep2020/photon_spectrum_Ex_6_75MeV.dat", skip_header=1, usecols=(0,1))

# Ex_freya_500 = data_freya_500[:,4]

# avg_ph_mult_freya_500 = data_freya_500[:,5]
# avg_ph_energy_freya_500 = data_freya_500[:,7]
# total_ph_E_freya_500 = data_freya_500[:,9]

# FREYA_MeV_scale_factor_500 = 1000/(photonspectrum_freya_6_75_500[101][0]*1000-photonspectrum_freya_6_75_500[100][0]*1000)


#####################################
###     Write PFG char to file    ###
#####################################

infile = open("241Pu_PFG_uncorrected_char.dat", "w")
infile.write("Ex 			M_g 		M_g_unc 	E_tot 		E_tot_unc 	E_g 		E_g_unc \n")
for i in range(len(Ex_bin_middle)):
	if Ex_bin_middle[i] > 5500 and Ex_bin_middle[i] < 8500:
		infile.write("%.4f		%.4f		%.4f		%.4f		%.4f		%.4f		%.4f \n" % (Ex_bin_middle[i]/1000.0, M_g[i], M_g_unc[i], E_tot[i]/1000.0, E_tot_unc[i]/1000.0, E_g[i]/1000.0, E_g_unc[i]/1000.0))
infile.close()

infile_2 = open("241Pu_PFG_corrected_char.dat", "w")
infile_2.write("Ex 			M_g 		M_g_unc 	E_tot 		E_tot_unc 	E_g 		E_g_unc \n")
for i in range(len(Ex_bin_middle)):
	if Ex_bin_middle[i] > 5500 and Ex_bin_middle[i] < 8500:
		infile_2.write("%.4f		%.4f		%.4f		%.4f		%.4f		%.4f		%.4f \n" % (Ex_bin_middle[i]/1000.0, M_g_corr_by_factor[i], M_g_unc_corr_by_factor[i], E_tot_corr_by_factor[i]/1000.0, E_tot_unc_corr_by_factor[i]/1000.0, E_g_corr_by_factor[i]/1000.0, E_g_unc_corr_by_factor[i]/1000.0))
infile_2.close()

#####################################
### Plot Spectral Characteristics ###
#####################################

# plt.figure(0)
# plt.errorbar(Ex_bin_middle, M_g, yerr=M_g_unc, fmt="bx-", label="OMpy unf, MAMA fn")
# plt.plot(Ex_bin_middle, M_g_thr, "gx-", label=" %d keV thr,NoNegFill" % E_g_min)
# plt.title("Total photon multiplicity")
# plt.xlabel("E$_{x}$ [keV]", fontsize=16)
# plt.ylabel("M$_{g}$ [photons/fission]", fontsize=16)
# plt.axis([5500,8500, 4, 8])
# plt.grid()
# plt.legend(fontsize=15)
# #plt.show()

# plt.figure(1)
# plt.errorbar(Ex_bin_middle, E_tot, yerr=E_tot_unc, fmt="bx-", label="OMpy unf, MAMA fn")
# plt.plot(Ex_bin_middle, E_tot_thr, "gx-", label=" %d keV thr,NoNegFill" % E_g_min)
# plt.title("Total photon energy")
# plt.xlabel("E$_{x}$ [keV]", fontsize=16)
# plt.ylabel("E$_{tot}$ [MeV/fission]", fontsize=16)
# plt.axis([5500,8500, 4000, 10000])
# plt.grid()
# plt.legend(fontsize=15)
# #plt.show()

xticks = np.arange(0,8.5,0.25)
xticks_text = [" ", "6.0", " ", "7.0", " ", "8.0", " "]
#xticks_text = [" ", " ", "6.0", " ", " ", "7.0", " "," ", "8.0", " ", " "]
ax1_yticks = np.arange(3.5,7,0.5)
ax2_yticks = np.arange(4,8,0.5)
ax3_yticks = np.arange(0.8,1.3,0.1)

csfont = {'fontname':'Times New Roman'}

Ex_plotting_array = np.linspace(5500,8500,100000)


#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
fig1, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
ax1.plot(Ex_freya_100, avg_ph_mult_freya_100, "go", label="FREYA")
ax1.plot(Ex_plotting_array/1000.0, 0.05633373352659756*(Ex_plotting_array/1000.0) + 6.941487741293519, "g-", label="FREYA linear fit")
#ax1.plot(Ex_freya_500, avg_ph_mult_freya_500, "mo--", label="FREYA, from 0.5 MeV")
ax1.plot(Ex_plotting_array/1000.0, energy_lin(Ex_plotting_array/1000.0, P_M_g[0], P_M_g[1]), "r-" ,label="Exp, weighted lin fit")
ax1.errorbar(Ex_bin_middle/1000.0, M_g, yerr=M_g_unc, capsize=5, fmt="rv", label="Exp")
ax1.errorbar(Ex_bin_middle/1000.0, M_g_corr_by_factor, yerr=M_g_unc_corr_by_factor, capsize=5, fmt="b^", label="Exp, correct_by_factor")
ax1.plot(Ex_plotting_array/1000.0, energy_lin(Ex_plotting_array/1000.0, P_M_g_corr_by_factor[0], P_M_g_corr_by_factor[1]), "b-" ,label="Exp corr_by_factor, weighted lin fit")
#ax1.errorbar(Ex_bin_middle/1000.0, M_g_PFGS_corr, yerr=M_g_PFGS_corr_unc, capsize=5, fmt="ko", label="Exp, PFGS corrected")
#ax1.grid(True)
ax1.set_ylim(5, 8)
ax1.set_ylabel("M$_{g}$ [photons/fission]", fontsize=13, fontweight='bold')
#ax1.set_yticklabels(ax1_yticks, fontsize=13)
#ax1.legend(frameon=False, fontsize=16, loc='upper right', bbox_to_anchor=(1.02, 0.88), ncol=2)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax1.tick_params(axis='y', which='major', labelsize=13)
ax1.tick_params(length=7, width=1, which="major")
ax1.tick_params(length=4, width=1, which="minor")

ax2.errorbar(Ex_bin_middle/1000.0, E_tot/1000.0, yerr=E_tot_unc/1000.0, fmt="rv", capsize=5)
ax2.plot(Ex_plotting_array/1000.0, energy_lin(Ex_plotting_array/1000.0, P_E_tot[0], P_E_tot[1]), "r-", label="Weighted linear fit")
ax2.errorbar(Ex_bin_middle/1000.0, E_tot_corr_by_factor/(1000.0), yerr=E_tot_unc_corr_by_factor/1000.0, capsize=5, fmt="b^", label="Exp, weighted by const. %")
ax2.plot(Ex_plotting_array/1000.0, energy_lin(Ex_plotting_array/1000.0, P_E_tot_corr_by_factor[0], P_E_tot_corr_by_factor[1]), "b-", label="Exp corr_by_factor, Weighted linear fit")
#ax2.errorbar(Ex_bin_middle/1000.0, E_tot_PFGS_corr/1000.0, yerr=E_tot_PFGS_corr_unc/1000.0, capsize=5, fmt="ko", label="Exp, PFGS corrected")
ax2.set_ylabel("E$_{tot}$ [MeV/fission]", fontsize=13, fontweight='bold')
ax2.plot(Ex_freya_100, total_ph_E_freya_100,"go", label="FREYA")
ax2.plot(Ex_plotting_array/1000.0, 0.07169885411604877*(Ex_plotting_array/1000.0) + 6.3870274717905335, "g-")
#ax2.plot(Ex_freya_500, total_ph_E_freya_500,"mo--", label="FREYA")
#ax2.set_yticklabels(ax2_yticks, rotation=0, fontsize=13)
ax2.set_ylim(6, 7.2)
ax2.tick_params(axis='y', which='major', labelsize=13)
#ax2.grid(True)
ax2.tick_params(length=7, width=1, which="major")
ax2.tick_params(length=4, width=1, which="minor")

ax3.errorbar(Ex_bin_middle/1000.0, E_g/1000.0, yerr=E_g_unc/1000.0, fmt="rv", capsize=5)
ax3.plot(Ex_plotting_array/1000.0, energy_lin(Ex_plotting_array/1000.0, P_E_g[0], P_E_g[1]), "r-", label="Weighted linear fit")
ax3.errorbar(Ex_bin_middle/1000.0, E_g_corr_by_factor/(1000.0), yerr=E_g_unc_corr_by_factor/(1000.0), capsize=5, fmt="b^", label="Exp, weighted by const. %")
ax3.plot(Ex_plotting_array/1000.0, energy_lin(Ex_plotting_array/1000.0, P_E_g_corr_by_factor[0], P_E_g_corr_by_factor[1]), "b-", label="Exp corr_by_factor, Weighted linear fit")
#ax3.errorbar(Ex_bin_middle/1000.0, E_g_PFGS_corr/1000.0, yerr=E_g_PFGS_corr_unc/1000.0, capsize=5, fmt="ko", label="Exp, PFGS corrected")
ax3.plot(Ex_freya_100, avg_ph_energy_freya_100, "go", label="FREYA")
ax3.plot(Ex_plotting_array/1000.0, 0.002534623161945984*(Ex_plotting_array/1000.0) + 0.9212988050713691, "g-")
#ax3.plot(Ex_freya_500, avg_ph_energy_freya_500, "mo--", label="FREYA")
ax3.set_xlabel("E$_{x}$ [MeV]", fontsize=13, fontweight='bold')
ax3.set_ylabel("E$_{\gamma}$ [MeV/photon]", fontsize=13, fontweight='bold')
#ax3.set_xticks(xticks, minor=True)
ax3.set_xlim(5.5, 8.5)
#ax3.set_xticklabels(xticks_text, fontsize=13)
#ax3.set_yticklabels(ax3_yticks, rotation=0, fontsize=13)
ax3.tick_params(axis='x', which='major', labelsize=13)
ax3.tick_params(axis='y', which='major', labelsize=13)
ax3.tick_params(length=7, width=1, which="major")
ax3.tick_params(length=4, width=1, which="minor")
#ax3.set_ylim(0.5, 1.6)
ax3.set_ylim(0.85, 1.4)
#ax3.grid(True)
plt.subplots_adjust(hspace=0.1)
#plt.show()

#fig1.savefig("241Pu_char_raw.pdf", bbox_inches='tight')


xticks_photonspectrum = ["0"," ","1"," ","2", " ","3", " ","4", " ","5", " ","6", " ","7", " ","8", " ", "9"]
xticks_arr = np.arange(0,9000,500)
yticks_arr = np.array([10**(-4), 10**(-3), 10**(-2), 10**(-1), 1, 10, 100])

G_array_middle = np.array(G_array_middle)

fig, ax = plt.subplots()
#ax.errorbar(G_array_middle/1000.0, M[11], fmt="x", yerr=M_unc[11], label="E$_{x}$ = %.1f - %.1f MeV" % (Ex_array[11]/1000.0,Ex_array[12]/1000.0))
#for i in range(12,17): #(11-17)
# 	#plt.errorbar(G_array_middle, M[i], fmt="x", yerr=M_unc[i], label="MAMA fn rn / OMpy unf E$_{x}$ = %.0f - %.0f keV" % (Ex_array[i],Ex_array[i+1]))
	#ax.plot(G_array_middle/1000.0, M[i], label="E$_{x}$ = %.1f - %.1f MeV" % (Ex_array[i]/1000.0,Ex_array[i+1]/1000.0))
	#ax.plot(G_array_middle/1000.0, M_rescaled_corrected[i], label="E$_{x}$ = %.1f - %.1f MeV" % (Ex_array[i]/1000.0,Ex_array[i+1]/1000.0))
ax.errorbar(photonspectrum_freya_6_75_100[:,0], photonspectrum_freya_6_75_100[:,1]*FREYA_MeV_scale_factor_100, yerr=photonspectrum_freya_6_75_100_unc*FREYA_MeV_scale_factor_100, label="FREYA 100 E$_{x}$=6.75 MeV", fmt="k-")
ax.plot(G_array_middle/1000.0, M[11], "orange" , label="E$_{x}$ = %.1f - %.1f MeV" % (Ex_array[11]/1000.0,Ex_array[11+1]/1000.0))
ax.plot(G_array_middle/1000.0, M[12], "limegreen" , label="E$_{x}$ = %.1f - %.1f MeV" % (Ex_array[12]/1000.0,Ex_array[12+1]/1000.0))
ax.errorbar(G_array_middle/1000.0, M[13],yerr=M_unc[13], color="royalblue", fmt=".", markersize=10, label="E$_{x}$ = %.1f - %.1f MeV" % (Ex_array[13]/1000.0,Ex_array[13+1]/1000.0))
ax.plot(G_array_middle/1000.0, M[14], "gold" , label="E$_{x}$ = %.1f - %.1f MeV" % (Ex_array[14]/1000.0,Ex_array[14+1]/1000.0))
ax.plot(G_array_middle/1000.0, M[15], "orchid" , label="E$_{x}$ = %.1f - %.1f MeV" % (Ex_array[15]/1000.0,Ex_array[15+1]/1000.0))
ax.plot(G_array_middle/1000.0, M[16], "aqua" , label="E$_{x}$ = %.1f - %.1f MeV" % (Ex_array[16]/1000.0,Ex_array[16+1]/1000.0))
#ax.errorbar(G_array_middle/1000.0, M[11], fmt="x", yerr=M_unc[11], label="E$_{x}$ = %.1f - %.1f MeV" % (Ex_array[11]/1000.0,Ex_array[12]/1000.0))
#ax.errorbar(G_array_middle/1000.0, M_rescaled_corrected[11], fmt="x", yerr=M_unc_rescaled_corrected[11], label="E$_{x}$ = %.1f - %.1f MeV" % (Ex_array[11]/1000.0,Ex_array[12]/1000.0))
#ax.plot(photonspectrum_freya_6_75_500[:,0], photonspectrum_freya_6_75_500[:,1]*FREYA_MeV_scale_factor_500, label="FREYA 500 E$_{x}$=6.75 MeV")
#ax.yscale('log', nonposy='clip')
ax.set_yscale('log', nonposy='clip')
#ax.grid(True)
ax.set_ylabel("Photons/(Fission MeV)", fontsize=13, fontweight='bold')
ax.set_xlabel("E$_{\gamma}$ [MeV]", fontsize=13, fontweight='bold')
ax.set_xlim(0,8)
ax.set_ylim(10**(-3),15)
ax.tick_params(length=7, width=1, which="major")
ax.tick_params(length=4, width=1, which="minor")
#ax.set_xlim(0.05,1)
#ax.set_ylim(9*10**(-1),12)

#handles,labels = ax.get_legend_handles_labels()

#handles = [handles[6], handles[0], handles[1], handles[2], handles[3], handles[4], handles[5]]
#labels = [labels[6], labels[0], labels[1], labels[2], labels[3], labels[4], labels[5]]
#ax.legend()
#ax.legend(handles, labels, loc="lower left", fontsize=20, frameon=False)
#ax.tick_params(axis="x", which="major", labelsize=18)
#ax.tick_params(axis="y", which="major", labelsize=18)
plt.show()

#fig.savefig("241Pu_PFGS_raw.pdf", bbox_inches='tight')


# plt.figure(2)
# plt.plot(photonspectrum_freya_6_75[:,0]*1000, photonspectrum_freya_6_75[:,1]*FREYA_MeV_scale_factor, label="FREYA E$_{x}$=6750 keV")
# for i in range(12,17): #(11-17)
# 	#plt.errorbar(G_array_middle, M[i], fmt="x", yerr=M_unc[i], label="MAMA fn rn / OMpy unf E$_{x}$ = %.0f - %.0f keV" % (Ex_array[i],Ex_array[i+1]))
# 	plt.plot(G_array_middle, M[i], label="E$_{x}$ = %.0f - %.0f keV" % (Ex_array[i],Ex_array[i+1]))
# plt.errorbar(G_array_middle, M[11], fmt="x", yerr=M_unc[11], label="E$_{x}$ = %.0f - %.0f keV" % (Ex_array[11],Ex_array[12]))
# plt.yscale('log', nonposy='clip')
# plt.grid()
# plt.ylabel("Photons/(Fission MeV)", fontsize=13, fontweight='bold')
# plt.xlabel("E$_{\gamma}$ [MeV]", fontsize=13, fontweight='bold')
# plt.axis([0,9000,10**(-4),100 ])
# plt.legend(bbox_to_anchor = [0.65, 0.6], fontsize=13, frameon=False)
# plt.xticks(xticks_arr, xticks_photonspectrum, fontsize=13)
# plt.yticks(yticks_arr, fontsize=13)
# plt.show()

#####################################
### Deviation FREYA vs experiment ###
#####################################

func_interpolate_FREYA_Mg = inp.interp1d(Ex_freya_100,avg_ph_mult_freya_100)
func_interpolate_FREYA_Etot = inp.interp1d(Ex_freya_100,total_ph_E_freya_100)
func_interpolate_FREYA_Eg = inp.interp1d(Ex_freya_100,avg_ph_energy_freya_100)

#Calculate initial fragment 1
FREYA_interpolated_Mg = func_interpolate_FREYA_Mg(Ex_bin_middle[11:17]/1000.0)
FREYA_interpolated_Etot = func_interpolate_FREYA_Etot(Ex_bin_middle[11:17]/1000.0)
FREYA_interpolated_Eg = func_interpolate_FREYA_Eg(Ex_bin_middle[11:17]/1000.0)

#Percent deviation
Mg_dev = abs((M_g_corr_by_factor[11:17] - FREYA_interpolated_Mg)*100.0/M_g_corr_by_factor[11:17])
Etot_dev = abs((E_tot_corr_by_factor[11:17]/1000.0 - FREYA_interpolated_Etot)*100/(E_tot_corr_by_factor[11:17]/1000.0))
Eg_dev = abs((E_g_corr_by_factor[11:17]/1000.0-FREYA_interpolated_Eg)*100/(E_g_corr_by_factor[11:17]/1000.0))

# print(Mg_dev)
# print("\n")
# print(Etot_dev)
# print("\n")
# print(Eg_dev)





