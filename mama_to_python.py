import numpy as np

def read_mama_2D(filename):
     # Reads a MAMA matrix file and returns the matrix as a numpy array,
     # as well as a list containing the four calibration coefficients
     # (ordered as [bx, ax, by, ay] where Ei = ai*channel_i + bi)
     # and 1-D arrays of calibrated x and y values for plotting and similar.
     matrix = np.genfromtxt(filename, skip_header=10, skip_footer=1)
     cal = {}
     with open(filename, 'r') as datafile:
         calibration_line = datafile.readlines()[6].split(",")
         # a = [float(calibration_line[2][:-1]), float(calibration_line[3][:-1]), float(calibration_line[5][:-1]),float(calibration_line[6][:-1])]
         # JEM update 20180723: Changing to dict, including second-order term for generality:
         # print("calibration_line =", calibration_line, flush=True)
         cal = {"a0x":float(calibration_line[1]), "a1x":float(calibration_line[2]), "a2x":float(calibration_line[3]), "a0y":float(calibration_line[4]),"a1y":float(calibration_line[5]), "a2y":float(calibration_line[6])}
     # TODO: INSERT CORRECTION FROM CENTER-BIN TO LOWER EDGE CALIBRATION HERE.
     # MAKE SURE TO CHECK rebin_and_shift() WHICH MIGHT NOT LIKE NEGATIVE SHIFT COEFF.
     # (alternatively consider using center-bin throughout, but then need to correct when plotting.)
     Ny, Nx = matrix.shape
     y_array = np.linspace(0, Ny-1, Ny)
     y_array = cal["a0y"] + cal["a1y"]*y_array + cal["a2y"]*y_array**2
     x_array = np.linspace(0, Nx-1, Nx)
     x_array = cal["a0x"] + cal["a1x"]*x_array + cal["a2x"]*x_array**2
     # x_array = np.linspace(cal["a0x"], cal["a0x"]+cal["a1x"]*Nx, Nx) #BIG TODO: This is probably center-bin calibration,
     # x_array = np.linspace(a[2], a[2]+a[3]*(Ny), Ny) # and should be shifted down by half a bin?
                                                     # Update 20171024: Started changing everything to lower bin edge,
                                                     # but started to hesitate. For now I'm inclined to keep it as
                                                     # center-bin everywhere.
     return matrix, cal, y_array, x_array # Returning y (Ex) first as this is axis 0 in matrix language

def read_mama_1D(filename):
    # Reads a MAMA spectrum file and returns the spectrum as a numpy array, 
    # as well as a list containing the calibration coefficients
    # and 1-D arrays of calibrated x values for plotting and similar.
    with open(filename) as file:
        lines = file.readlines()
        a0 = float(lines[6].split(",")[1]) # calibration
        a1 = float(lines[6].split(",")[2]) # coefficients [keV]
        a2 = float(lines[6].split(",")[3]) # coefficients [keV]
        N = int(lines[8][15:]) +1 # 0 is first index
        cal = {"a0x":a0, "a1x":a1, "a2x":a2}
    x_array = np.linspace(0, N-1, N)
    x_array = cal["a0x"] + cal["a1x"]*x_array + cal["a2x"]*x_array**2
    # Read the rest:
    array = np.genfromtxt(filename, comments="!")
    return array, cal, x_array