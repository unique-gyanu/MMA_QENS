import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

__version__ = "0.4.1"

def plot_msd(*data, percentage=1, legend=None,legend_loc = 'best', fit=None, subdiffusion = False, timeunit='ps'):
    """
    Plot the mean square displacement

    :param data: time series
    :type data: np.ndarray
    :param percentage: percentage of the data that will be plotted,
                       value between 0 and 1.
    :type percentage: float
    :param legend: data legend
    :type legend: list of string
    :param legend_loc: location of legend
    :type legend: int or string
    :param fit: fitting coefficients
    :type fit: list of 2 floats
    :param subdiffusion: if subdiffusion, the subplot is different
    :type subdiffusion: bool
    :param timeunit: unit for the time axis
    :type timeunit: str
    """
    fig, ax = plt.subplots(figsize=(10,7))

    # plot data
    if len(data[0].shape) == 1:
        for msd in data:
            msd = msd[:int(len(msd)*percentage)]
            ax.plot(np.arange(len(msd)), msd)
            ax.set_xlabel('t', size=16)
            ax.set_title('MSD', size=16)
    if len(data[0].shape) == 2:
        for msd in data:
            ax.plot(msd[0, :int(len(msd[0])*percentage)],
                    msd[1, :int(len(msd[1])*percentage)])
            ax.set_xlabel(f'time [{timeunit}]', size=16)
            ax.set_ylabel(r'$MSD [nm^2]$', size=16)
            ax.set_xlim([0, msd[0, int(len(msd[0])*percentage)-1]])
            ax.set_ylim([0, msd[1, int(len(msd[1])*percentage)-1]])
        if fit:
            if subdiffusion:
                ax_sub = fig.add_axes([0.18, 0.65, 0.2, 0.2],
                             xlim=(0, msd[0, int(len(msd[0])*percentage)-1]*0.1),
                             ylim=(0, msd[1, int(len(msd[1])*percentage)-1]*0.45))
                # Remove spines and ticks
                ax_sub.spines['right'].set_color('none')
                ax_sub.spines['top'].set_color('none');
                for msd in data:
                    ax_sub.plot(msd[0, :int(len(msd[0])*percentage)],
                                msd[1, :int(len(msd[1])*percentage)])
                # Adding text
                x = msd[0, int(len(msd[0])*percentage)] / 2
                ax.text(x = x, y = 0.4, s = r"Fit: 2 $D_{\alpha} t^{\alpha}$",
                        fontsize = 18, color='#FF8000', alpha = 1)
                ax.text(x = x, y = 0.3,
                        s = r"$D_{\alpha}$ =" + f"{fit[0]:.4f} " + r"$nm^2/$" + f"{timeunit}" + r"$^{\alpha}$",
                        fontsize = 16, color='#FF8000', alpha = 1)
                ax.text(x = x, y = 0.2, s = r"$\alpha$ =" + f"{fit[1]:.4f}",
                        fontsize = 16, color='#FF8000', alpha = 1)
            else:
                ax_sub = fig.add_axes([0.20, 0.65, 0.2, 0.2],
                             xlim=(0, msd[0, int(len(msd[0])*percentage)-1]*0.1),
                             ylim=(0, msd[1, int(len(msd[1])*percentage)-1]*0.15))
                # Remove spines and ticks
                ax_sub.spines['right'].set_color('none')
                ax_sub.spines['top'].set_color('none');
                for msd in data:
                    ax_sub.plot(msd[0, :int(len(msd[0])*percentage)],
                                msd[1, :int(len(msd[1])*percentage)])
                # Adding text
                ax.text(x = 5, y = 0.08, s = "Fit: 6 D t + a",
                        fontsize = 18, color='#FF8000', alpha = 1)
                ax.text(x = 5.6, y = 0.06, s = f"D = {fit[0]/6:.4f}",
                        fontsize = 16, color='#FF8000', alpha = 1)
                ax.text(x = 5.6, y = 0.04, s = f"a = {fit[1]:.4f}",
                        fontsize = 16, color='#FF8000', alpha = 1)


    # labels
    ax.tick_params(labelsize=16)

    # Legend
    if legend:
        ax.legend(legend,
                  loc=legend_loc,
                  frameon=True,
                  shadow=True,
                  facecolor='#FFFFFF',
                  framealpha=0.9,
                  fontsize=14)

    # Remove spines and ticks
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none');


def plot_coordinates(coord_pbc, coord_nopbc):
    """
    Plot coord_pbc and coord_nopbc

    :param coord_pbc: coordinates in one dimension (x, y or z along time) of
                      one atom with periodic boundary conditions
    :type coord_pbc: np.ndarray
    :param coord_nopbc: coordinates in one dimension (x, y or z along time) of
                        one atom without the periodic boundary conditions
    :type coord_nopbc: np.ndarray
    """
    fig, ax = plt.subplots(figsize=(10,7))

    # plot data
    ax.plot(coord_pbc, label = 'with PBC')
    ax.plot(coord_nopbc, label = 'without PBC')

    # labels
    ax.set_xlabel('time step', size=16)
    ax.set_ylabel('coordinate', size=16)
    ax.tick_params(labelsize=16)

    # Legend
    ax.legend(['with PBC', 'without PBC'],
              loc='best',
              frameon=True,
              shadow=True,
              facecolor='#FFFFFF',
              framealpha=0.9,
              fontsize=14)

    # Remove spines and ticks
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none');


def plot_velocities(*data, legend=None):
    """
    Plot the velocities

    :param data: time series
    :type data: np.ndarray
    :param legend: data legend
    :type legend: list of string
    """
    fig, ax = plt.subplots(figsize=(10,7))

    # plot data
    if len(data[0].shape) == 1:
        for velocity in data:
            velocity = velocity[:int(len(velocity))]
            ax.plot(np.arange(len(velocity)), velocity)
            ax.set_xlabel('t', size=16)
            ax.set_title('velocity', size=16)
    if len(data[0].shape) == 2:
        alpha = 1.3
        for velocity in data:
            alpha -= 0.4
            ax.plot(velocity[0, :int(len(velocity[0]))],
                    velocity[1, :int(len(velocity[1]))],
                    alpha=alpha, zorder=0.1)
    # 0 graduation starts at x=0
    ax.set_xlim([0, int(velocity[0, -1])+2])
    # change position of xlabel and ylabel
    #ax.set_xlabel('time [ps]', size=16)
    #ax.set_ylabel('v [nm/ps]', size=16)
    ax.text(x = velocity[0, -1] +5, y = -0.55, s = "time [ps]",
            fontsize = 18, alpha = 1)
    ax.text(x = -1, y = velocity[1].max()+0.7, s = "v [nm/ps]",
            fontsize = 18, alpha = 1)

    # ticks, labels
    ax.tick_params(labelsize=16,
                   direction='inout', length=5, top='off', right='off')
    ## remove the 0 value manually
    ax.set_xticklabels([''] + list(range(20, int(velocity[0, -1])+1, 20)))

    # Legend
    if legend:
        ax.legend(legend,
                  loc='best',
                  frameon=True,
                  shadow=True,
                  facecolor='#FFFFFF',
                  framealpha=0.9,
                  fontsize=14)

    # Remove spines and ticks
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # bottom spine -> position y=0
    ax.spines['bottom'].set_position(('data',0))
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)


def plot_vacf(*data, percentage=1, legend=None, color=None):
    """
    Plot the velocity auto-correlation function

    :param data: time series
    :type data: np.ndarray
    :param percentage: percentage of the data that will be plotted,
                       value between 0 and 1.
    :type percentage: float
    :param legend: data legend
    :type legend: list of string
    """
    fig, ax = plt.subplots(figsize=(10,7))

    print
    # plot data
    if len(data[0].shape) == 1:
        for vacf in data:
            ax.plot(vacf[0, :int(len(vacf[0])*percentage)],
                       vacf[1, :int(len(vacf[1])*percentage)])
            ax.set_xlabel('time [ps]', size=16)
            ax.set_ylabel('VACF [$nm^2/ps^2$]', size=16)
            ax.set_xlim([0, vacf[0, int(len(vacf[0])*percentage)-1]])
            ax.set_ylim([0, vacf[1, int(len(vacf[1])*percentage)-1]])

    if len(data[0].shape) == 2:
        for vacf in data:
            ax.plot(vacf[0, :int(len(vacf[0])*percentage)],
                       vacf[1, :int(len(vacf[1])*percentage)],
                       color = color)
            ax.set_xlabel('time [ps]', size=16)
            ax.set_ylabel('VACF [$nm^2/ps^2$]', size=16)
            ax.set_xlim([0, vacf[0, int(len(vacf[0])*percentage)-1]])
            ymin = np.around(vacf[1].min()-0.1, decimals=1)
            ymax = np.around(vacf[1].max()+0.1, decimals=1)
            ax.set_ylim([ymin, ymax])

    # labels
    ax.tick_params(labelsize=16)
    ## remove the 0 value manually for the x label
    ax.set_xticklabels([''] + list(np.linspace(0,
                       int(((data[0][0])*percentage)[-1]), 6)[1:]))

    # Legend
    if legend:
        ax.legend(legend,
                  loc='best',
                  frameon=True,
                  shadow=True,
                  facecolor='#FFFFFF',
                  framealpha=0.9,
                  fontsize=14)

    # Remove spines and ticks
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data',0));


def plot_dos(*data, percentage=1, legend=None):
    """
    Plot the density of states

    :param data: series
    :type data: np.ndarray
    :param percentage: percentage of the data that will be plotted,
                       value between 0 and 1.
    :type percentage: float
    :param legend: data legend
    :type legend: list of string
    """
    fig, ax = plt.subplots(figsize=(10,7))

    # plot data
    if len(data[0].shape) == 1:
        for dos in data:
            ax.plot(dos[0, :int(len(dos[0])*percentage)],
                       dos[1, :int(len(dos[1])*percentage)])
            ax.set_xlabel('$\omega$ [THz]', size=16)
            ax.set_ylabel('$g(\omega)$', size=16)
            ax.set_xlim([0, dos[0, int(len(dos[0])*percentage)-1]])
            ax.set_ylim([0, dos[1, int(len(dos[1])*percentage)-1]])

    if len(data[0].shape) == 2:
        for dos in data:
            ax.plot(dos[0, :int(len(dos[0])*percentage)],
                       dos[1, :int(len(dos[1])*percentage)],
                       )
            ax.set_xlabel('$\omega$ [THz]', size=16)
            ax.set_ylabel('$g(\omega)$', size=16)
            ax.set_xlim([0, dos[0, int(len(dos[0])*percentage)-1]])
            #ymin = np.around(dos[1].min()-0.1, decimals=1)
            #ymax = np.around(dos[1].max()+0.1, decimals=1)
            #ax.set_ylim([ymin, ymax])

    # labels
    ax.tick_params(labelsize=16)

    # Legend
    if legend:
        ax.legend(legend,
                  loc='best',
                  frameon=True,
                  shadow=True,
                  facecolor='#FFFFFF',
                  framealpha=0.9,
                  fontsize=14)

    # Remove spines and ticks
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data',0));


def plot_fcoh(data, ticks=None, time=None, percentage=1, legend=None):
    """
    Plot F-coh

    :param data: series
    :type data: np.ndarray
    :param ticks: series
    :type ticks: np.ndarray
    :param time: series
    :type time: np.ndarray
    :param percentage: percentage of the data that will be plotted,
                       value between 0 and 1.
    :type percentage: float
    :param legend: data legend
    :type legend: boolean
    """
    fig, ax = plt.subplots(figsize=(10,7))

    # plot data
    if ticks is not None and time is None:
        ax.plot(ticks[:int(len(ticks)*percentage)],
                   data[:int(len(data)*percentage)],
                   linewidth=3)
        ax.set_xlabel('$q [nm^{-1}]$', size=16)
        ax.set_ylabel('$F(q, 0)$', size=16)

    if ticks is not None and time is not None:
        for i in range(0, len(ticks), 5):
            ax.plot(time[:int(len(time)*percentage)],
                       data[i, :int(len(data[i])*percentage)],
                       linewidth=2,
                       label="q = {:.4f}".format(ticks[i]))
            ax.set_xlabel('$time [ps]$', size=16)
            ax.set_ylabel('$F(q, t)$', size=16)

    # labels
    ax.tick_params(labelsize=16)

    # Legend
    if legend:
        ax.legend(#legend,
                  loc='best',
                  frameon=True,
                  shadow=True,
                  facecolor='#FFFFFF',
                  framealpha=0.9,
                  fontsize=14)

    # Remove spines and ticks
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data',0));


def plot_benchmark(processes, data):
    fig, ax = plt.subplots(figsize=(10,7))

    textcolor='#3B4241'
    fontsize=16

    # convert data in percentage
    data_max = max(data)
    percentage = [d/data_max*100 for d in data]

    # Title
    ax.set_title("Execution time in function of the number of processes", size=fontsize+2)
    # plot data
    for i, (process, benchmark, percent) in enumerate(zip(processes, data, percentage)):
        ax.barh(i, percent, alpha=0.5, color="red")
        if process==1:
            ax.text(1, i, f"{process} process - {benchmark:.2f} seconds", fontsize=fontsize, fontdict={'color': textcolor})
        else:
            ax.text(1, i, f"{process} processes - {benchmark:.2f} seconds", fontsize=fontsize, fontdict={'color': textcolor})

    # x ticks
    ax.tick_params(labelsize=16)
    ax.set_xticks([0, 25, 50, 75, 100])
    # Add % to the x-ticks
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    xticks = mpl.ticker.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)


    # Remove spines and ticks
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data',-0.5));
    ax.set_yticks([]);
