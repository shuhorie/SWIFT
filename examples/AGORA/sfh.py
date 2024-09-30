import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
import numpy as np
import glob
import os
import sys
import re


t_min, t_max = 000, 1000
sfr_min = 0
sfr_max = 8

ms = 10
fs = 24
fs_txt = 18
fs_l = 14
lw = 2
lwt = lw * 2
ls = 16
hs = 0.4
ws = 0.45
Alpha = 0.1
padinches = 0.05
dpi = 300

gamma = 1.66666666666667 # 5/3
HYDROGEN_MASS_FRAC = 0.76
PROTONMASS = 1.6726e-24
BOLTZMANN_CONST = 1.38066e-16
MyrInS = 3.155e13
GRAVITY_CGS = 6.672e-8

UNIT_MASS_IN_CGS = 1.989e33
UNIT_LENGTH_IN_CGS = 3.085678e21
UNIT_VELOCITY_IN_CGS = 1.e5
UNIT_TIME_IN_CGS = UNIT_LENGTH_IN_CGS / UNIT_VELOCITY_IN_CGS
UNIT_DENSITY_IN_CGS = UNIT_MASS_IN_CGS / (UNIT_LENGTH_IN_CGS*UNIT_LENGTH_IN_CGS*UNIT_LENGTH_IN_CGS)
UNIT_DENSITY_IN_NHCGS = UNIT_DENSITY_IN_CGS / PROTONMASS
UNIT_TIME_IN_MYR = UNIT_TIME_IN_CGS / MyrInS
GRAVITY = (GRAVITY_CGS / (UNIT_LENGTH_IN_CGS*UNIT_LENGTH_IN_CGS*UNIT_LENGTH_IN_CGS) * UNIT_MASS_IN_CGS * (UNIT_TIME_IN_CGS*UNIT_TIME_IN_CGS))


matplotlib.rc('font', family='serif')

def get_max_num_snapshot(snapshot_dir):
    files = os.listdir(snapshot_dir)
    pattern = re.compile(r'snapshot_(\d{4})\.hdf5')
    numbers = [int(pattern.search(f).group(1)) for f in files if pattern.search(f)]
    print(max(numbers))
    return max(numbers)


def calc_sfh(snapshot_dir):
    snapshot_num = get_max_num_snapshot(snapshot_dir=snapshot_dir)

    t = np.empty(0, dtype=np.float64)
    sfr1 = np.zeros(snapshot_num+1, dtype=np.float64)
    sfr10 = np.empty(0, dtype=np.float64)
    sfr100 = np.empty(0, dtype=np.float64)
    total = np.empty(0, dtype=np.float64)

    file = f"{snapshot_dir}/snapshot_{snapshot_num:04d}.hdf5"
    ds = h5py.File(file, 'r')
    boxsize = ds['Header'].attrs['BoxSize']
    center = np.full(3, 0.5*boxsize)
    Time = ds['Header'].attrs['Time']
    Time *= UNIT_TIME_IN_MYR
    NumPart = ds['Header'].attrs['NumPart_ThisFile']
    Ntotal = NumPart[4]
    if Ntotal <= 0:
        print("No stars in %s" % snapshot_dir)
        sys.exit()

    sftime = np.array(ds['PartType4/BirthTimes'])
    mass = np.array(ds['PartType4/Masses'])

    sftime *= UNIT_TIME_IN_MYR

    cond_data = (sftime >= 0)
    for i in range(snapshot_num+1):
        cond_data_i = (np.ceil((sftime-t_min)/1.0).astype(int)) == i
        sfr1[i] += mass[cond_data & cond_data_i].sum()
        t = np.append(t, i*1.0)

    sfr1 *= 1e10    # mass unit is 10^10 Msun
    sfr1 *= 1e-6    # per yr

    for loop in range(snapshot_num+1):
        l = min([9, loop])
        sfr10 = np.append(sfr10, np.average(sfr1[loop-l:loop+1]))

        l = min([99, loop])
        sfr100 = np.append(sfr100, np.average(sfr1[loop-l:loop+1]))

    for loop in range(1, snapshot_num+1):
        total = np.append(total, sfr1[0:loop].cumsum()*1e6)

    return t, sfr1, sfr10, sfr100, total


def sfh_fig():
    # EAGLE_MFM
    snapshot_dir0 = f"{SNAPSHOT_DIR}/eagle_mfm/snapshot"
    t_0, sfr1_0, sfr10_0, sfr100_0, total_0 = calc_sfh(snapshot_dir=snapshot_dir0)

    # EAGLE_MFM
    snapshot_dir1 = f"{SNAPSHOT_DIR}/eagle/snapshot"
    t_1, sfr1_1, sfr10_1, sfr100_1, total_1 = calc_sfh(snapshot_dir=snapshot_dir1)

    # EAGLE_MFM
    snapshot_dir2 = f"{SNAPSHOT_DIR}/agora/snapshot"
    t_2, sfr1_2, sfr10_2, sfr100_2, total_2 = calc_sfh(snapshot_dir=snapshot_dir2)

    fig = plt.figure(figsize=(8.09, 5))
    plt.rcParams["mathtext.fontset"]= "cm"
    # plt.subplots_adjust(hspace=0, wspace=0)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    ax = fig.add_subplot(111)
    ax.set_xlabel(r"$t \ {\rm [Myr]}$", fontsize=fs)
    ax.set_ylabel(r"${\rm SFR \ [M_{\odot} \ yr^{-1}]}$", fontsize=fs)
    ax.set_xlim(t_min, t_max)
    # ax.set_ylim(sfr_min, sfr_max)
    plt.tick_params(labelsize=ls)
    plt.grid(linewidth=1)
    ax.plot(t_0, sfr1_0, color="blue", linestyle="-", linewidth=lw, label="EAGLE_MFM")
    ax.plot(t_1, sfr1_1, color="green", linestyle="--", linewidth=lw, label="EAGLE")
    ax.plot(t_2, sfr1_2, color="red", linestyle=":", linewidth=lw, label="AGORA")
    plt.legend(loc='upper right', fontsize=fs_l)

    plt.savefig(f"{FIG_DIR}/sfh.png", bbox_inches="tight", pad_inches=padinches, dpi=dpi)
    plt.close()


if __name__ == "__main__":

    SNAPSHOT_DIR = "/data3/EVERGREEN/sims/iso"
    FIG_DIR = './fig/star-formation'

    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)

    sfh_fig()

