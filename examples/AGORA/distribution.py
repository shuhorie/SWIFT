import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
import numpy as np
import glob
import os
import sys
import cv2

import warnings
warnings.simplefilter('ignore')

ms = 10
fs = 18
fs_l = 14
lw = 2
ls = 12

gamma = 1.66666666666667 # 5/3
MASSFRAC = 0.76
UNITMASS = 1.989e42
UNITLENGTH = 3.085678e21
UNITVELOCITY = 1.e5
UNITTIME = UNITLENGTH / UNITVELOCITY
PROTONMASS = 1.6726e-24
BOLTZMANN_CONST = 1.38066e-16
GRAVITY_CONST = 6.672e-8
Gint = GRAVITY_CONST / UNITLENGTH**3.0 * UNITTIME**2.0 * UNITMASS
factor_density = MASSFRAC*UNITMASS/(UNITLENGTH*UNITLENGTH*UNITLENGTH)/PROTONMASS
factor_press = UNITMASS * UNITVELOCITY * UNITVELOCITY / (UNITLENGTH * UNITLENGTH * UNITLENGTH) / BOLTZMANN_CONST
MyrInS = 3.155e13
MYR = MyrInS / UNITTIME

bbox_text = {
    'facecolor': 'white',
    'edgecolor': 'gray',
    'alpha': 1,
    'boxstyle': 'Round'
}


matplotlib.rc('font',family='serif')


def projection(x, y, z):
    proj_xy = np.vstack((x, y)).T
    proj_xz = np.vstack((x, z)).T
    return proj_xy, proj_xz

def particle_distribution():
    for i_snapshot in range(FIRST_SNAPSHOT, LAST_SNAPSHOT+1):
        file = '%s/snapshot_%04d.hdf5' % (SNAPSHOT_DIR, i_snapshot)
        ds = h5py.File(file, 'r')
        boxsize = ds['Header'].attrs['BoxSize']
        center = np.full(3, 0.5*boxsize)
        Time = ds['Header'].attrs['Time']
        tmyr = Time / MYR
        NumPart = ds['Header'].attrs['NumPart_ThisFile']

        pos_gas = np.array(ds['PartType0/Coordinates'])
        mass_gas = np.array(ds['PartType0/Masses'])
        pos_dm = np.array(ds['PartType1/Coordinates'])
        mass_dm = np.array(ds['PartType1/Masses'])
        pos_star = np.array(ds['PartType4/Coordinates'])
        mass_star = np.array(ds['PartType4/Masses'])


        plt.subplots_adjust(wspace=0.4)

        fig = plt.figure(figsize=(12, 20))
        plt.rcParams["mathtext.fontset"] = "cm"

        # gas
        sigma_max, sigma_min = 3e2, 3e-1
        x_min, x_max, Nx = -15.0, 15.0, 256
        y_min, y_max, Ny = -15.0, 15.0, 256
        xbin = np.linspace(x_min, x_max, Nx)
        ybin = np.linspace(y_min, y_max, Ny)
        dx = (x_max - x_min) / Nx
        dy = (y_max - y_min) / Ny
        area = dx * dy

        proj_xy, proj_xz = projection(pos_gas[:,0]-center[0], pos_gas[:,1]-center[1], pos_gas[:,2]-center[2])

        ax = fig.add_subplot(321)
        ax.set_xlabel(r"$x \ [{\rm kpc}]$", fontsize=fs)
        ax.set_ylabel(r"$y \ [{\rm kpc}]$", fontsize=fs)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        # ax.hist2d(proj_xy[:,0], proj_xy[:,1], bins=[xbin, ybin], weights=m[cond]/area, cmin=sigma_min, cmax=sigma_max, cmap='inferno', norm=LogNorm())
        hist, xedges, yedges = np.histogram2d(proj_xy[:, 0], proj_xy[:, 1], bins=[xbin, ybin], weights=mass_gas/area)
        ax.imshow(np.log10(hist.T), origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='inferno', aspect='auto')
        ax.set_aspect('equal')

        ax = fig.add_subplot(322)
        ax.set_xlabel(r"$x \ [{\rm kpc}]$", fontsize=fs)
        ax.set_ylabel(r"$z \ [{\rm kpc}]$", fontsize=fs)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        # ax.hist2d(proj_xy[:,0], proj_xy[:,1], bins=[xbin, ybin], weights=m[cond]/area, cmin=sigma_min, cmax=sigma_max, cmap='inferno', norm=LogNorm())
        hist, xedges, yedges = np.histogram2d(proj_xz[:, 0], proj_xz[:, 1], bins=[xbin, ybin], weights=mass_gas/area)
        ax.imshow(np.log10(hist.T), origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='inferno', aspect='auto')
        ax.set_aspect('equal')
        ax.text(3.0, 12.0, r"$t=%.3f \ {\rm Gyr}$" % (tmyr*0.001), fontsize=fs, bbox=bbox_text)

        # dm
        sigma_max, sigma_min = 3e2, 3e-1
        x_min, x_max, Nx = -100.0, 100.0, 256
        y_min, y_max, Ny = -100.0, 100.0, 256
        xbin = np.linspace(x_min, x_max, Nx)
        ybin = np.linspace(y_min, y_max, Ny)
        dx = (x_max - x_min) / Nx
        dy = (y_max - y_min) / Ny
        area = dx * dy

        proj_xy, proj_xz = projection(pos_dm[:,0]-center[0], pos_dm[:,1]-center[1], pos_dm[:,2]-center[2])

        ax = fig.add_subplot(323)
        ax.set_xlabel(r"$x \ [{\rm kpc}]$", fontsize=fs)
        ax.set_ylabel(r"$y \ [{\rm kpc}]$", fontsize=fs)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        # ax.hist2d(proj_xy[:,0], proj_xy[:,1], bins=[xbin, ybin], weights=m[cond]/area, cmin=sigma_min, cmax=sigma_max, cmap='inferno', norm=LogNorm())
        hist, xedges, yedges = np.histogram2d(proj_xy[:, 0], proj_xy[:, 1], bins=[xbin, ybin], weights=mass_dm/area)
        ax.imshow(np.log10(hist.T), origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='inferno', aspect='auto')
        ax.set_aspect('equal')

        ax = fig.add_subplot(324)
        ax.set_xlabel(r"$x \ [{\rm kpc}]$", fontsize=fs)
        ax.set_ylabel(r"$z \ [{\rm kpc}]$", fontsize=fs)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        # ax.hist2d(proj_xy[:,0], proj_xy[:,1], bins=[xbin, ybin], weights=m[cond]/area, cmin=sigma_min, cmax=sigma_max, cmap='inferno', norm=LogNorm())
        hist, xedges, yedges = np.histogram2d(proj_xz[:, 0], proj_xz[:, 1], bins=[xbin, ybin], weights=mass_dm/area)
        ax.imshow(np.log10(hist.T), origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='inferno', aspect='auto')
        ax.set_aspect('equal')


        # star
        sigma_max, sigma_min = 3e2, 3e-1
        x_min, x_max, Nx = -15.0, 15.0, 256
        y_min, y_max, Ny = -15.0, 15.0, 256
        xbin = np.linspace(x_min, x_max, Nx)
        ybin = np.linspace(y_min, y_max, Ny)
        dx = (x_max - x_min) / Nx
        dy = (y_max - y_min) / Ny
        area = dx * dy

        proj_xy, proj_xz = projection(pos_star[:,0]-center[0], pos_star[:,1]-center[1], pos_star[:,2]-center[2])

        ax = fig.add_subplot(325)
        ax.set_xlabel(r"$x \ [{\rm kpc}]$", fontsize=fs)
        ax.set_ylabel(r"$y \ [{\rm kpc}]$", fontsize=fs)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        # ax.hist2d(proj_xy[:,0], proj_xy[:,1], bins=[xbin, ybin], weights=m[cond]/area, cmin=sigma_min, cmax=sigma_max, cmap='inferno', norm=LogNorm())
        hist, xedges, yedges = np.histogram2d(proj_xy[:, 0], proj_xy[:, 1], bins=[xbin, ybin], weights=mass_star/area)
        ax.imshow(np.log10(hist.T), origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='inferno', aspect='auto')
        ax.set_aspect('equal')

        ax = fig.add_subplot(326)
        ax.set_xlabel(r"$x \ [{\rm kpc}]$", fontsize=fs)
        ax.set_ylabel(r"$z \ [{\rm kpc}]$", fontsize=fs)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        # ax.hist2d(proj_xy[:,0], proj_xy[:,1], bins=[xbin, ybin], weights=m[cond]/area, cmin=sigma_min, cmax=sigma_max, cmap='inferno', norm=LogNorm())
        hist, xedges, yedges = np.histogram2d(proj_xz[:, 0], proj_xz[:, 1], bins=[xbin, ybin], weights=mass_star/area)
        ax.imshow(np.log10(hist.T), origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='inferno', aspect='auto')
        ax.set_aspect('equal')

        plt.savefig(f"{FIG_DIR}/projection_{i_snapshot:04d}.png", bbox_inches="tight", dpi=300)
        plt.close()

def png2mp4():
    fps = 30

    image_files = sorted(glob.glob(f"{FIG_DIR}/projection_????.png"))
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4用のコーデック
    video = cv2.VideoWriter(f"{FIG_DIR}/projection.mp4", fourcc, fps, (width, height))  # {fps} FPS (フレーム/秒)

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

    FIG_DIR = './fig/distribution'
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)

    particle_distribution()
    png2mp4()

