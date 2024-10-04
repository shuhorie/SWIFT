import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors
import h5py
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import os
import sys
import re
import cv2

txt_dense = 4
txt_temp = 5
txt_press = 2

fs = 20
fs_txt = 16
padinches = 0.05
dpi = 300


gamma = (5.0/3.0) # 5/3
MASSFRAC = 0.76
Msun_in_cgs = 1.989e33
PROTONMASS = 1.6726e-24
BOLTZMANN_CONST = 1.38066e-16
Myr_in_s = 3.155e13

matplotlib.rc('font',family='serif')


def rho_P_u2T_unit_converter(unit_mass_in_cgs, unit_length_in_cgs, unit_time_in_cgs):
    unit_velocity_in_cgs = unit_length_in_cgs / unit_time_in_cgs

    factor_density = MASSFRAC * unit_mass_in_cgs / (unit_length_in_cgs**3.0) / PROTONMASS

    UNIT_ENERGY = unit_mass_in_cgs * (unit_velocity_in_cgs**2)
    factor_press = UNIT_ENERGY / (unit_length_in_cgs**3) / BOLTZMANN_CONST

    u2T = (PROTONMASS/BOLTZMANN_CONST) * (UNIT_ENERGY/unit_mass_in_cgs)

    return factor_density, factor_press, u2T

def phase_diagram():
    file = f"{SNAPSHOT_DIR}/snapshot_{FIRST_SNAPSHOT:04d}.hdf5"
    ds = h5py.File(file, 'r')

    unit_mass_in_cgs = ds['Units'].attrs['Unit mass in cgs (U_M)']
    unit_length_in_cgs = ds['Units'].attrs['Unit length in cgs (U_L)']
    unit_time_in_cgs = ds['Units'].attrs['Unit time in cgs (U_t)']
    factor_density, factor_press, u2T = rho_P_u2T_unit_converter(unit_mass_in_cgs, unit_length_in_cgs, unit_time_in_cgs)

    x_min, x_max, Nx = -3.0, 6.0, 181
    xbin = np.linspace(x_min, x_max, Nx)

    y0_min, y0_max, Ny0 = 0.0, 7.0, 141
    y0bin = np.linspace(y0_min, y0_max, Ny0)
    axis_ratio0 = (x_max - x_min) / (y0_max - y0_min)

    y1_min, y1_max, Ny1 = 0.0, 10.0, 201
    y1bin = np.linspace(y1_min, y1_max, Ny1)
    axis_ratio1 = (x_max - x_min) / (y1_max - y1_min)

    norm = mcolors.Normalize(vmin=1e4, vmax=1e8)

    molecular_weight_above_10000K = 4.0 / (8.0 - 5.0*(1.0-MASSFRAC))
    molecular_weight_above_1000K = 4.0 / (1.0 + 3.0*MASSFRAC)
    molecular_weight_below_1000K = 1.0 / (0.5*MASSFRAC + 0.25*(1.0-MASSFRAC) * 1.0/28.0)
    U_AT_10000K = 10000.0 / ((gamma-1.0) * u2T * molecular_weight_above_10000K)
    U_AT_1000K = 1000.0 / ((gamma-1.0) * u2T * molecular_weight_above_1000K)

    for i_snapshot in range(FIRST_SNAPSHOT, LAST_SNAPSHOT+1):
        file = f"{SNAPSHOT_DIR}/snapshot_{i_snapshot:04d}.hdf5"
        ds = h5py.File(file, 'r')

        NumPart = ds['Header'].attrs['NumPart_ThisFile']
        n = NumPart[0]
        if n <= 0:
            continue

        mass = np.array(ds['PartType0/Masses'])
        mass *= unit_mass_in_cgs / Msun_in_cgs
        density = np.array(ds['PartType0/Densities'])
        intU = np.array(ds['PartType0/InternalEnergies'])

        pressure = np.array(ds['PartType0/Pressures'])
        # pressure = (gamma-1.0) * density * intU
        pressure = np.log10(pressure * factor_press)

        # temperature = np.log10(temperature)
        density = np.log10(density * factor_density)

        # temperature = []
        # for i in range(len(intU)):
        #     t = intU[i] * (gamma-1.0) * u2T
        #     if intU[i] >= U_AT_10000K:
        #         t *=molecular_weight_above_10000K
        #     elif intU[i] >= U_AT_1000K:
        #         t *= molecular_weight_above_1000K
        #     else:
        #         t *= molecular_weight_below_1000K
        #     temperature.append(t)
        # temperature = np.where(intU >= U_AT_10000K,
        #                        intU * (gamma - 1.0) * u2T * molecular_weight_above_10000K,
        #                        np.where(intU >= U_AT_1000K,
        #                                 intU * (gamma - 1.0) * u2T * molecular_weight_above_1000K,
        #                                 intU * (gamma - 1.0) * u2T * molecular_weight_below_1000K))
        temperature = intU * (gamma - 1.0) * u2T * molecular_weight_above_1000K
        temperature = np.log10(temperature)


        fig = plt.figure(figsize=(16.18, 10))
        plt.rcParams["mathtext.fontset"]= "cm"
        plt.subplots_adjust(hspace=0.4, wspace=0.25)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        ax = fig.add_subplot(1,2,1)
        ax.set_xlabel(r"$\log \, n_{\rm H} \ {\rm [cm^{-3}]}$", fontsize=fs)
        ax.set_ylabel(r"$\log \, T \ {\rm [K]}$", fontsize=fs)
        # ax.set_aspect(aspect=axis_ratio0, adjustable='box')
        ax.set_box_aspect(1)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y0_min, y0_max)
        # ax.text(txt_dense, txt_temp, r"$t = %3.0f \, {\rm Myr}$" % myr, fontsize=fs_txt, fontname="serif", color="black")
        _, _, _, im = plt.hist2d(density, temperature, bins=[xbin, y0bin], weights=mass, norm=LogNorm(), cmap='jet')
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0)
        # cbar = plt.colorbar(cax=cax)
        im.set_clim(1e5, 1e9)
        cbar = plt.colorbar(im, pad=0, shrink=0.616)
        # plt.colorbar(im, label="Counts")
        cbar.set_label(r"${\rm Cell \ Mass \ [M_\odot]}$", fontsize=20)

        ax = fig.add_subplot(1,2,2)
        ax.set_xlabel(r"$\log \, n_{\rm H} \ {\rm [cm^{-3}]}$", fontsize=fs)
        ax.set_ylabel(r"$\log \, (P/k_{\rm B}) \ {\rm [K \, cm^{-3}]}$", fontsize=fs)
        # ax.set_aspect(aspect=axis_ratio1, adjustable='box')
        ax.set_box_aspect(1)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y1_min, y1_max)
        # ax.text(txt_dense, txt_press, r"$t = %3.0f \, {\rm Myr}$" % myr, fontsize=fs_txt, fontname="serif", color="black")
        _, _, _, im = plt.hist2d(density, pressure, bins=[xbin, y1bin], weights=mass, norm=LogNorm(), cmap='jet')
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0)
        # cbar = plt.colorbar(cax=cax)
        im.set_clim(1e5, 1e9)
        cbar = plt.colorbar(im, pad=0, shrink=0.616)
        cbar.set_label(r"${\rm Cell \ Mass \ [M_\odot]}$", fontsize=20)

        plt.savefig(f"{FIG_DIR}/phase_diagram_{i_snapshot:04d}.png", bbox_inches="tight", pad_inches=padinches, dpi=dpi)
        plt.close()

def png2mp4():
    fps = 30

    image_files = sorted(glob.glob(f"{FIG_DIR}/phase_diagram_????.png"))
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4用のコーデック
    video = cv2.VideoWriter(f"{FIG_DIR}/phase_diagram.mp4", fourcc, fps, (width, height))  # {fps} FPS (フレーム/秒)

    for image_file in image_files:
        img = cv2.imread(image_file)
        video.write(img)

    # 動画作成完了後にリソースを解放
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = sys.argv
    FIRST_SNAPSHOT = int(args[1])
    LAST_SNAPSHOT = int(args[2])

    SNAPSHOT_DIR = "./snapshot"
    FIG_DIR = './fig/phase'

    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)

    phase_diagram()
    png2mp4()



