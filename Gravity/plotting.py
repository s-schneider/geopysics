import matplotlib.pyplot as plt


def plot_gravity(Vxx, Vxy, Vxz, Vyy, Vyz, Vzz, P, Vz, title=None):

    plt.ion()
    fig = plt.figure()
    ax = range(6)
    ax[0] = plt.subplot2grid((8, 5), (0, 0), rowspan=2, colspan=2)
    ax[1] = plt.subplot2grid((8, 5), (0, 3), rowspan=2, colspan=2)
    ax[2] = plt.subplot2grid((8, 5), (3, 0), rowspan=2, colspan=2)
    ax[3] = plt.subplot2grid((8, 5), (3, 3), rowspan=2, colspan=2)
    ax[4] = plt.subplot2grid((8, 5), (6, 0), rowspan=2, colspan=2)
    ax[5] = plt.subplot2grid((8, 5), (6, 3), rowspan=2, colspan=2)

    fig.suptitle(title)
    # subplot 321
    cax = ax[0].imshow(P, aspect='auto', cmap='viridis')
    ax[0].set_title(r'Potential $[m^2/s^2]$')
    fig.colorbar(cax, ax=ax[0])

    # # subplot 322
    cax = ax[1].imshow(Vz, aspect='auto', cmap='viridis')
    ax[1].set_title(r'Gravity anomaly $[m/s^2]$')
    fig.colorbar(cax, ax=ax[1])

    # # subplot 323
    cax = ax[2].imshow(Vxx, aspect='auto', cmap='viridis')
    ax[2].set_title(r'Gravity gradient Vxx $[1/s^2]$')
    fig.colorbar(cax, ax=ax[2])

    # # subplot 324
    cax = ax[3].imshow(Vyy, aspect='auto', cmap='viridis')
    ax[3].set_title(r'Gravity gradient Vyy $[1/s^2]$')
    fig.colorbar(cax, ax=ax[3])

    # # subplot 325
    cax = ax[4].imshow(Vzz, aspect='auto', cmap='viridis')
    ax[4].set_title(r'Gravity gradient Vzz $[1/s^2]$')
    fig.colorbar(cax, ax=ax[4])

    # # subplot 326
    cax = ax[5].imshow(Vxz, aspect='auto', cmap='viridis')
    ax[5].set_title(r'Gravity gradient Vxz $[1/s^2]$')
    fig.colorbar(cax, ax=ax[5])

    plt.show()
    plt.ioff()
    return


def plot_hollow_sphere(a, b, r, g):
    rmax = 2*b
    c = g.max()
    fig, ax = plt.subplots()

    ax.plot(r, g, 'black', linewidth=0.85)
    ax.annotate(s='a', xy=(a, 0))
    ax.annotate(s='b', xy=(b, 0))
    ax.annotate(s='I', xy=(a/2., c/3.))
    ax.annotate(s='II', xy=((a+b)/2., c/3.,))
    ax.annotate(s='III', xy=(b+(rmax-b)/2., c/3.))
    ax.vlines(a, 0, c/2., 'r', linewidth=0.75,
              linestyle='--')
    ax.vlines(b, 0, c/2., 'r', linewidth=0.75,
              linestyle='--')

    plt.show()

    return
