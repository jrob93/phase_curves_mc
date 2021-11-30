import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.modeling.fitting import LevMarLSQFitter
from sbpy.photometry import HG, HG1G2, HG12, HG12_Pen16

def resample_data_uniform(df_data1):
    """
    resample df_data1, drawing new data points from a uniform distribution of the error on each point
    """
    df_data2=df_data1.copy()
    lower = df_data1["reduced_mag"] - df_data1["merr"]
    upper = df_data1["reduced_mag"] + df_data1["merr"]
    df_data2["reduced_mag"] = np.random.uniform(lower,upper)
    return df_data2

def fit_phase_curve(phase_angle,reduced_mag,mag_err):
    """
    Fit a single phase curve function to the reduced magnitude vs phase angle.
    Weighted by magnitude error.
    Requires correct astropy units
    """

    fitter = LevMarLSQFitter() # select fitter
    model = HG() # select sbpy model, could choose others here
    model = fitter(model, phase_angle, reduced_mag, weights=1.0/np.array(mag_err))

    return model

name = "Hidalgo" # object name
N_iter = int(1e1) # number of iterations
np.random.seed(0) # set random seed for reproducibility

# load the observational data. Single file containing both filters, might need to rename some columns to match this script
# required columns: phase_angle, reduced_mag, merr, filter
df_all_data = pd.read_csv("data/{}_ATLAS_forced_phot.csv".format(name),index_col=0)
print(df_all_data)

# Set up empty lists to hold all values of H and G
i_list = []
filt_list = []
H_list = []
G_list = []

for filt in ["o","c"]:

    # select only data in the chosen filter
    df_data = df_all_data[df_all_data["filter"]==filt]
    print(df_data)

    # initiate plot for the data
    fig = plt.figure()
    gs = gridspec.GridSpec(1,1)
    ax1 = plt.subplot(gs[0,0])

    # plot the starting observations
    ax1.errorbar(df_data["phase_angle"],df_data["reduced_mag"],df_data['merr'], fmt='ko', markersize="2")
    ax1.invert_yaxis()
    ax1.set_xlabel("phase_angle")
    ax1.set_ylabel("reduced_mag")

    # loop over number of iterations
    for i in range(N_iter):

        # resample the data
        _df_data = resample_data_uniform(df_data)
        ax1.scatter(_df_data["phase_angle"],_df_data["reduced_mag"], facecolor="k", edgecolor = "none",alpha=0.3, s=25, marker="_")

        # do a single phase curve fit. sbpy likes arrays of data with astropy units
        phase_angle = np.array(_df_data["phase_angle"])*u.deg
        reduced_mag = np.array(_df_data["reduced_mag"])*u.mag
        merr = np.array(_df_data['merr'])*u.mag
        model = fit_phase_curve(phase_angle,reduced_mag,merr)
        print(model)
        print(model.parameters)
        i_list.append(i)
        filt_list.append(filt)
        H_list.append(model.parameters[0])
        G_list.append(model.parameters[1])

        # plot the model
        alpha = np.linspace(np.amin(_df_data["phase_angle"]),np.amax(_df_data["phase_angle"]))
        ax1.plot(alpha,model(alpha*u.deg),c = "r", alpha = 0.1)

    plt.show()
    # plt.close()

    break # comment this out to do both filters

# store all the results in a dataframe
df_results = pd.DataFrame(np.array([i_list,filt_list,H_list,G_list]).T, columns=["i","filter","H","G"])
df_results = df_results.astype(dtype = {"i":"int","filter":"str","H":"float64","G":"float64"})
print(df_results)
print(df_results.dtypes)

# scatter plot H and G values
fig = plt.figure()
gs = gridspec.GridSpec(1,1)
ax1 = plt.subplot(gs[0,0])

_df_results = df_results[df_results["filter"]=="o"]
ax1.scatter(_df_results["H"], _df_results["G"])
ax1.set_xlabel("H")
ax1.set_xlabel("G")

plt.show()

# plot the histogram distributions in H and G
fig = plt.figure()
gs = gridspec.GridSpec(2,1)
ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[1,0])

_df_results = df_results[df_results["filter"]=="o"]
ax1.hist(_df_results["H"], bins = "auto", histtype="step")
ax1.set_xlabel("H")
ax2.hist(_df_results["G"], bins = "auto", histtype="step")
ax2.set_xlabel("G")

plt.tight_layout()
plt.show()
