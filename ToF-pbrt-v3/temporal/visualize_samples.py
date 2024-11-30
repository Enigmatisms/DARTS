import numpy as np
import matplotlib.pyplot as plt
from transient_read import colors

def isotropic_sample(long_axis2 = 40, foci_dist = 20, samples = None, show = False):
    if samples is None:
        samples = np.random.rand(1000000000) * 2 - 1
    t_samples = (long_axis2 ** 2 - foci_dist ** 2) / (2 * long_axis2 - 2 * foci_dist * samples)
    
    if show:
        plt.hist(t_samples, np.linspace((long_axis2 - foci_dist) * 0.5, 0.5 * (long_axis2 + foci_dist), 200))
        plt.show()
    return t_samples
    
def uniform_sample_dist(num_samples = 50000, show = False):
    long_axis2 = 40
    foci_dist = 20
    samples = long_axis2 / foci_dist - (long_axis2 ** 2 - foci_dist ** 2) / (2 * foci_dist * ((long_axis2 - foci_dist) * 0.5 + foci_dist * np.random.rand(num_samples)))

    if show:
        plt.hist(samples, np.linspace(-1, 1, 200))
        plt.show()
    return samples

def sample_cos(target_t, foci_dist, half_power_diff):
    a2c2 = target_t - foci_dist
    inv_d = 1 / foci_dist
    nominator = half_power_diff * inv_d
    denominator = foci_dist * np.random.rand(100000) + a2c2 * 0.5
    cosTheta = np.clip(target_t * inv_d - nominator / denominator, -1, 1)
    return cosTheta

def visualize_cos_theta():
    T = 40
    ds = [5, 15, 25]
    y = np.linspace(-1, 1, 2000)
    samples = []
    for i, d in enumerate(ds):
        half_pd = 0.5 * (T + d) * (T - d)
        cos_thetas = sample_cos(T, d, half_pd)
        samples.append(cos_thetas)
    plt.subplot(2, 1, 1)
    plt.violinplot(samples, showmeans = True, showextrema = True)
    plt.xticks([1, 2, 3], labels = [f'd = {x}' for x in ds])
    plt.title(f'Target time = {T}, cos theta samples')
    plt.grid(axis = 'both')
    plt.legend()

    plt.subplot(2, 1, 2)
    distances = []
    for sample, d in zip(samples, ds):
        distances.append(isotropic_sample(T, d, sample))
    plt.violinplot(distances, showmeans = True, showextrema = True)
    plt.xticks([1, 2, 3], labels = [f'd = {x}' for x in ds])
    plt.title(f'Target time = {T}, distance samples')
    plt.grid(axis = 'both')
    plt.show()

def visualize_cos_p():
    T = 40
    ds = [5, 15, 25]
    y = np.linspace(-1, 1, 2000)
    plt.subplot(1, 2, 1)
    for i, d in enumerate(ds):
        cdf_values = (T**2 - d**2) / (2*d*(T - d * y)) - 0.5 * T / d + 0.5
        plt.plot(y, cdf_values, c = colors[i], label = f"T = 40, d = {d}")
        plt.xlabel('cos theta values')
        plt.ylabel('CDF values')
        plt.title('cos theta CDF for elliptical sampling')
        plt.grid(axis = 'both')
        plt.legend()

    plt.subplot(1, 2, 2)
    for i, d in enumerate(ds):
        pdf_values = 0.5 * (T**2 - d**2) / (T - d * y) ** 2
        plt.plot(y, pdf_values, c = colors[i], label = f"T = 40, d = {d}")
        plt.xlabel('cos theta values')
        plt.ylabel('PDF values')
        plt.title('cos theta PDF for elliptical sampling')
        plt.grid(axis = 'both')
        plt.legend()
    plt.show()

def pdf_comparison(samples, T = 40, d = 20):
    bins = np.linspace(-1, 1, 200)
    hist, _ = np.histogram(samples, bins)
    hist = hist.astype(np.float64)
    hist /= hist.max()
    xs = np.linspace(-1, 1, 10000)
    pdf = (T**2 - d**2) * 0.5 / (T - d * xs) ** 2
    pdf /= pdf.max()
    plt.plot(bins[:-1], hist, label = 'r')
    plt.plot(xs, pdf, label = 'b')
    plt.grid(axis = 'both')
    plt.legend()
    plt.show()

    
if __name__ == "__main__":
    samples = uniform_sample_dist(show = False)
    pdf_comparison(samples)
    # isotropic_sample(samples = samples, show = True)
    # visualize_cos_p()
    # visualize_cos_theta()
    