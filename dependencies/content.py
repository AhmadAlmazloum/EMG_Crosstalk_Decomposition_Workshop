import numpy as np
import matplotlib.pyplot as plt


# Definig the font style to the required
plt.rcParams["font.size"] = 14

# Initilaize global variable
color_scale = np.zeros(2)


def generate_sources(s1,s2):
    global color_scale

    color_scale = s1

    print("Mean s1 = ", round(s1.mean(),3), "\t\tStd s1 = ", round(s1.std(ddof=1),3)),
    print("Mean s2 = ", round(s2.mean(),3), "\t\tStd s2 = ", round(s2.std(ddof=1),3),'\n')

def plot_sources_time_domain(y1, y2, sampling_frequency = 2048):
  plt.figure()
  plt.plot(np.arange(y1.__len__())/sampling_frequency,y1,color=(0.9412, 0.3922, 0.3922, 0.8))
  plt.xlabel("Time (s)")
  plt.ylabel("$s_1(t)$")
  plt.show()

  plt.figure()
  plt.plot(np.arange(y2.__len__())/sampling_frequency,y2,color=(0.3922, 2 * 0.3922, 1.0, 0.8))
  plt.xlabel("Time (s)")
  plt.ylabel("$s_2(t)$")
  plt.show()

def plot_sources_and_observations(x, y, var="sources", eigen=False, H=np.array([])):

    # Plotting the Joint Distribuition of the sources
    fig = plt.figure()
    grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)

    plt.subplots_adjust(0.18, 0.16, 0.95, 0.95)

    main_ax = fig.add_subplot(grid[1:, :-1])
    y_hist = fig.add_subplot(grid[1:, -1], yticklabels=[])
    x_hist = fig.add_subplot(grid[0, :-1], xticklabels=[])

    x_left, x_right = x.min(), x.max()
    y_bottom, y_top = y.min(), y.max()

    main_ax.plot(np.array([x_left, x_right]), np.zeros(2), "k")
    main_ax.plot(np.zeros(2), np.array([y_bottom, y_top]), "k")

    main_ax.scatter(x, y, s=0.5, c=color_scale, cmap="rainbow")
    y_hist.hist(
        y,
        100,
        histtype="bar",
        rwidth=0.8,
        density=True,
        orientation="horizontal",
        color=(0.3922, 2 * 0.3922, 1.0, 0.8),
    )
    x_hist.hist(
        x,
        100,
        histtype="bar",
        rwidth=0.8,
        density=True,
        orientation="vertical",
        color=(0.9412, 0.3922, 0.3922, 0.8),
    )

    if var == "observations" or len(var) > 12:

        if var == "observations":
            main_ax.set_ylabel("$x_2$")
            main_ax.set_xlabel("$x_1$")
        else:
            tag = var[12:].replace(" ", "")
            tag = tag.replace("-", "\_")
            tag = tag.replace("_", "\_")
            label = "$x_2^{" + tag + "}$"
            main_ax.set_ylabel(label)
            label = "$x_1^{" + tag + "}$"
            main_ax.set_xlabel(label)

        if eigen or H.shape[0] > 0:

            if H.shape[0] > 0:
                U = H
                quiver_label1 = "$\\bf{h}_{1}$"
                quiver_label2 = "$\\bf{h}_{2}$"

            else:
                Cxx = np.cov(np.array([x - np.mean(x), y - np.mean(y)]))
                d, U = np.linalg.eig(Cxx)
                quiver_label1 = "$\\bf{u}_{1}$"
                quiver_label2 = "$\\bf{u}_{2}$"

            origin = np.array([0, 0])

            scale = max(abs(x_left), abs(x_right))
            scale = max(scale, abs(y_bottom))
            scale = max(scale, abs(y_top)) / 3
            scale = max(abs(U[:, 0])) / scale

            u1 = main_ax.quiver(*origin, *(U[:, 0]), color="r", scale_units="xy", scale=scale)
            u2 = main_ax.quiver(*origin, *(U[:, 1]), color="b", scale_units="xy", scale=scale)

            main_ax.quiverkey(
                u1,
                U[0, 0] / scale,
                U[1, 0] / scale,
                1,
                quiver_label1,
                labelpos="E",
                coordinates="data",
                visible=False,
            )
            main_ax.quiverkey(
                u2,
                U[0, 1] / scale,
                U[1, 1] / scale,
                1,
                quiver_label2,
                labelpos="E",
                coordinates="data",
                visible=False,
            )

    else:
        main_ax.set_ylabel("$s_2$")
        main_ax.set_xlabel("$s_1$")

    y_hist.set_ylim(main_ax.get_ylim())
    x_labels = y_hist.get_xticks()
    x_labels = x_labels[:-1]
    y_hist.set_xticks(x_labels)
    y_hist.set_xticklabels(x_labels, rotation=30)
    x_hist.set_xlim(main_ax.get_xlim())

    plt.show()

class Skewness:
  """Class that defines the Skweness function, G(w) = (w^4)/4, and its first and second derivatives.
  Then it can be use as Cost Function in the Fixed Point Algorithm (fastICA).
  """

  @staticmethod
  def g(w: np.ndarray) -> np.ndarray:
    """First derivative of Skewness function
    G(w) = (w^4)/4 -> dG(w)/dw = g(w) = w^3
    Parameters
    ----------
        w ([float]): The i-th separation vector.
    Returns
    ----------
        ([float]): The i-th separation vector with all elments being the squared of the original.
    """
    return np.power(w, 3)

  @staticmethod
  def dg_dw(w: np.ndarray) -> np.ndarray:
    """Second derivative of Skewness function
    G(w) = (w^4)/4 -> d^2G(w)/dw^2 = dg(w)/dw = 3*w^2
    Parameters
    ----------
        w ([float]): The i-th separation vector.
    Returns
    ----------
        ([float]): The i-th separation vector with all elments being the twice the original.
    """

    return 3 * np.square(w)


class LogCosh:
  """Class that defines the Log Cosh function, G(w) = log(cosh(w)), and its first and second derivatives.
  Then it can be use as Cost Function in the Fixed Point Algorithm (fastICA). Here is used the natural
  logarithm, so the cost function becomes G(w) = ln(cosh(w)).
  """

  @staticmethod
  def g(w: np.ndarray) -> np.ndarray:
    """First derivative of Log Cosh function
    G(w) = ln(cosh(w)) -> dG(w)/dw = g(w) = tanh(w)
    Parameters
    ----------
        w ([float]): The i-th separation vector.
    Returns
    ----------
        ([float]): The i-th separation vector with all elments being the hyperbolic tangent of the original elements.
    """
    # If is desired use logarithms in another base, by example, in base 10, then the return must be:
    # return np.tanh(w) / np.log(10) # equals to np.tanh(w) / 2.302585092994046
    return np.tanh(w) / 2.302585092994046
    # return np.tanh(w)

  @staticmethod
  def dg_dw(w: np.ndarray) -> np.ndarray:
    """Second derivative of Log Cosh function
    G(w) = ln(cosh(w)) -> d^2G(w)/dw^2 = dg(w)/dw = sech^2(w) = 1 - tanh^2(w)
    Parameters
    ----------
        w ([float]): The i-th separation vector.
    Returns
    ----------
        ([float]): The i-th separation vector with all elments being the hyperbolic squared secant of the original elements.
    """
    # If is desired use logarithms in another base, by example, in base 10, then the return must be:
    # return (1 - np.square(np.tanh(w))) / np.log(10)
    return (1 - np.square(np.tanh(w))) / 2.302585092994046
    # return (1 - np.square(np.tanh(w)))


class ExpSquared:
  """Class that defines the Exponetial of w squared function, G(w) = exp(-(w^2)/2), and its first and
  second derivatives. Then it can be use as Cost Function in the Fixed Point Algorithm (fastICA).
  """

  @staticmethod
  def g(w: np.ndarray) -> np.ndarray:
    """First derivative of Exponetial of w squared function
    G(w) = exp(-(w^2)/2) -> dG(w)/dw = g(w) = -exp(-(w^2)/2)*w
    Parameters
    ----------
        w ([float]): The i-th separation vector.
    Returns
    ----------
        ([float]): The i-th separation vector with all elments being the G(w) function
        times the negative of the original elements.
    """

    return (-w) * np.exp((-1 / 2) * w * w)

  @staticmethod
  def dg_dw(w: np.ndarray) -> np.ndarray:
    """Second derivative of Exponetial of w squared function
    G(w) = exp(-(w^2)/2) -> d^2G(w)/dw^2 = dg(w)/dw = (exp(-(w^2)/2))*(w^2 - 1)
    Parameters
    ----------
        w ([float]): The i-th separation vector.
    Returns
    ----------
        ([float]): The i-th separation vector with all elments being the G(w) function
        times the squared original elements minus 1.
    """

    return ((w * w) - 1) * np.exp((-1 / 2) * w * w)


def fastICA(z: np.ndarray, M: int = 120, max_iter: int = 50, Tolx: float = 0.0001, cost: int = 1):
    """
    FastICA algorithm proposed by (Hyvärinen, Oja, 1997) and (Hyvärinen, 1999) to estimate the
    projecction vector w. This algorithm is a fixed point algorithm with orthogonalization and
    normalization steps, for a better estimation.

    Parameters
    ----------
        z ([[float]]): Whitened extended observation matrix.
        M (int): Number of iterations of the whole algorithm (that is, the maximum number of possibly estimated sources).
        max_iter (int): Maximum number of iterations that fastICA will run to find an estimated
            separation vector on the i-th iteration of main FOR loop.
        Tolx (float): The toleration or convergence criteria that sepration vectors from fastICA must
            satisfy.

    Returns
    ----------
        ([float]): Array correspondig to current estimation of the projection vector.

    """

    B: np.ndarray = np.zeros((z.shape[0], M), dtype=float)
    BB: np.ndarray = 0 * np.identity(z.shape[0])

    if cost == 1:
      cf = Skewness
    elif cost == 3:
      cf = ExpSquared
    else:
      cf = LogCosh

    for i in range(M):

        """
        1. Initialize the vector w_i(0) and w_i(-1) with unit norm
        """
        w_new = np.random.rand(z.shape[0])
        vec_norm = np.linalg.norm(w_new)
        if vec_norm > 0:
            w_new /= vec_norm

        """
                2. While |w_i(n)^{T}w_i(n - 1) - 1| > (0.0001 = Tolx)
            """
        n = 0
        while True and n < max_iter:

            w_old = np.copy(w_new)

            """
                    a. Fixed point algorithm
                        w_i(n) = E{zg[w_i(n - 1)^{T}z]} - Aw_i(n - 1)
                        with A = E{g'[w_i(n - 1)^{T}z]}
                """
            s = np.dot(w_old, z)
            w_new = (z * cf.g(s)).mean(axis=1) - cf.dg_dw(s).mean() * w_old

            """
                    b. Orthogonalization
                        w_i(n) = w_i(n) - BB^{T}w_i(n)
                """
            w_new -= np.dot(BB, w_new)

            """
                    c. Normalization
                        w_i(n) = w_i(n)/||w_i(n)||
                """
            vec_norm = np.linalg.norm(w_new)
            if vec_norm > 0:
                w_new /= vec_norm

            # Recalculate convergece criterion
            tolx = np.absolute(np.dot(w_new, w_old) - 1)

            if tolx <= Tolx:
              break

            """
                    d. Set n = n + 1
                """
            n += 1

        B[:, i] = w_new
        BB += np.dot(w_new.reshape(-1, 1), w_new.reshape(1, -1))

    return B


def fastICA2(z: np.ndarray, M: int = 120, max_iter: int = 100, Tolx: float = 0.0001, cost: int = 1):
    """
    FastICA algorithm proposed by (Hyvärinen, Oja, 1997) and (Hyvärinen, 1999) to estimate the
    projecction vector w. This algorithm is a fixed point algorithm with orthogonalization and
    normalization steps, for a better estimation.

    Parameters
    ----------
        z ([[float]]): Whitened extended observation matrix.
        M (int): Number of iterations of the whole algorithm (that is, the maximum number of possibly estimated sources).
        max_iter (int): Maximum number of iterations that fastICA will run to find an estimated
            separation vector on the i-th iteration of main FOR loop.
        Tolx (float): The toleration or convergence criteria that sepration vectors from fastICA must
            satisfy.

    Returns
    ----------
        ([float]): Array correspondig to current estimation of the projection vector.

    """

    B: np.ndarray = np.zeros((z.shape[0], M), dtype=float)
    BB: np.ndarray = 0 * np.identity(z.shape[0])

    if cost == 1:
      cf = Skewness
    elif cost == 3:
      cf = ExpSquared
    else:
      cf = LogCosh

    for i in range(M):

        """
        1. Initialize the vector w_i(0) and w_i(-1) with unit norm
        """
        w_new = np.random.rand(z.shape[0])
        vec_norm = np.linalg.norm(w_new)
        if vec_norm > 0:
            w_new /= vec_norm

        """
                2. While |w_i(n)^{T}w_i(n - 1) - 1| > (0.0001 = Tolx)
            """
        n = 0
        while True and n < max_iter:

            w_old = np.copy(w_new)

            """
                    a. Fixed point algorithm
                        w_i(n) = E{zg[w_i(n - 1)^{T}z]} - Aw_i(n - 1)
                        with A = E{g'[w_i(n - 1)^{T}z]}
                """
            s = np.dot(w_old, z)
            w_new = (z * cf.g(s)).mean(axis=1) - cf.dg_dw(s).mean() * w_old

            """
                    b. Orthogonalization
                        w_i(n) = w_i(n) - BB^{T}w_i(n)
                """
            w_new -= np.dot(BB, w_new)

            """
                    c. Normalization
                        w_i(n) = w_i(n)/||w_i(n)||
                """
            vec_norm = np.linalg.norm(w_new)
            if vec_norm > 0:
                w_new /= vec_norm

            # Recalculate convergece criterion
            tolx = np.absolute(np.dot(w_new, w_old) - 1)

            if tolx <= Tolx:
              break

            """
                    d. Set n = n + 1
                """
            n += 1

        B[:, i] = w_new

    return B

