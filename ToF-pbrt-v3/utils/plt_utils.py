import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Union, Iterable, List
from matplotlib.ticker import StrMethodFormatter

COLORS  = ['#B31312', '#1640D6', '#1AACAC', '#EE9322', '#2B2A4C', '#AC87C5']

def get_asym_deviation(mean_v: np.ndarray, data: np.ndarray):
    """ Get asymmetric deviation (positive and negative)
        mean_v: of shape (N_param_sets)            ndim = 1
        data:   of shape (N_param_sets, N_trials)  ndim = 2 
    """
    y_std_pos = []
    y_std_neg = []
    if isinstance(data, np.ndarray):
        deviations = data - mean_v[..., None]
        for dev in deviations:
            y_std_pos.append(np.mean(dev[dev >= 0]))
            y_std_neg.append(-np.mean(dev[dev < 0]))
    else:
        for i, row in enumerate(data):
            deviation = row - mean_v[i]     # vector
            y_std_pos.append(np.mean(deviation[deviation >= 0]))
            y_std_neg.append(-np.mean(deviation[deviation < 0]))
    return np.float32(y_std_neg), np.float32(y_std_pos)    

def drop_max(y_data: Union[np.ndarray, list]):
    if isinstance(y_data, np.ndarray):
        # When there are some values being too big, it will have negative influence on the outcome of the figure
        y_data = np.sort(y_data, axis = 1)
        return y_data[..., :-1]           # the rid of the maximum value
    return [np.sort(data_row)[:-1] for data_row in y_data]
    
def apply_operation(y_data: Union[np.ndarray, list], func = "mean"):
    if isinstance(y_data, np.ndarray):
        return getattr(y_data, func)(axis = 1)
    return np.float32([getattr(np, func)(data) for data in y_data])

def assert_dim_2d(y_data: Union[np.ndarray, list]):
    is_array = isinstance(y_data, np.ndarray)
    if is_array and y_data.ndim == 2: return
    if not is_array and not isinstance(y_data[0][0], Iterable) == 2: return
    shape = y_data.shape if is_array else "<3D python list>"
    raise ValueError("All the Y data patch should be 2D, yet the current one is: ", shape)

def try_stack(inputs: list):
    try:
        result = np.stack(inputs, axis = 0)
    except ValueError:      # if the inner lists are of different length, stacking will fail
        return inputs
    return result

def tolist(y_data: Union[np.ndarray, List[np.ndarray]]):
    if isinstance(y_data, np.ndarray): return y_data.tolist()
    return [data.tolist() for data in y_data]

def lineplot_with_errbar(
    all_x_labels: list, all_y_data: np.ndarray, 
    title = 'Line Graph with Error Bars', xlabel = 'X-axis Label', 
    ylabel = 'Y-axis Label', labels = [], label_suffix = None, 
    label_as_tick = False, drop_maximum = True, make_even = True,
    complete_figure = False, violin_plot = False, draw_now = False, 
    save_image = False, legend_loc = "upper left", tmfp_ref = 0.0, 
    plot_title = False, set_precision = False, disable_min_max = False, 
    extra_label = [], ylog_scale = False
):
    """ Note that x_labels are list of strings, but they should be able to convert from str to float """
    same_xlabel  = not isinstance(all_x_labels[0], (list, tuple))
    for i, y_data in enumerate(all_y_data):
        x_labels = all_x_labels if same_xlabel else all_x_labels[i]
        if type(x_labels[0]) == str:
            xs = np.float32([float(string) for string in x_labels])
        else:
            xs = x_labels if isinstance(x_labels, np.ndarray) else np.float32(x_labels)
        if tmfp_ref > 1e-3:
            xs *= tmfp_ref
        assert_dim_2d(y_data)
        y_mean = apply_operation(y_data, 'mean')          # mean AMSE will not be influenced 
        if drop_maximum:
            drop_max(y_data)
        
        label = f'Series {i + 1}' if not labels else labels[i]
        if label_suffix:
            label = f'{label} {label_suffix[i]}'
        xs_input = xs.copy()
        if make_even:
            xs_diff = xs[1:] - xs[:-1]
            if np.std(xs_diff) > 1e-4:
                xs_input = np.arange(0, xs.size)
            else:
                make_even = False
                
        # y_std_neg, y_std_pos = get_asym_deviation(y_mean, y_data)
        # std_args = {'x': xs_input, 'y': y_mean, 'yerr': [y_std_neg, y_std_pos], 'fmt':'none', 'elinewidth':5, 'color':COLORS[i], 'alpha':0.5}

        std_args = {'x': xs_input, 'y': y_mean, 'fmt':'none', 'elinewidth':5, 'color':COLORS[i], 'alpha':0.5}
        
        if violin_plot:
            # note that violin plot takes columns as data, since our y_data is of shape (N_param_set, N_trials), we need to transpose it
            width = (xs[1:] - xs[:-1]).mean() * 0.2
            std_args['elinewidth'] = 3
            plt.errorbar(**std_args)
            if not isinstance(y_data, np.ndarray): 
                raise ValueError("Violin plotting mode can not be used in convergence analysis where `y_data` is not np.ndarray.")
            parts = plt.violinplot(y_data.T, positions = xs_input, showmeans = False, showextrema = False, widths = width)
            for pc in parts['bodies']:
                pc.set_alpha(0.5)
            plt.plot(xs_input, y_mean,label = label, color = COLORS[i])
        else:
            if disable_min_max:
                range = None
            else:
                yerr_low = y_mean - apply_operation(y_data, 'min')
                yerr_up  = apply_operation(y_data, 'max') - y_mean
                range = [yerr_low, yerr_up]
            plt.errorbar(**std_args)
            plt.errorbar(x = xs_input, y = y_mean, yerr = range, fmt='-o', label=label, ecolor = COLORS[i], color = COLORS[i],
                 elinewidth = 1, capsize = 2.5, markerfacecolor = COLORS[i], markersize = 7, markeredgecolor = 'none')
        if make_even:
            plt.xticks(xs_input, xs if not extra_label else extra_label)
        else:
            if extra_label:
                plt.xticks(xs, extra_label)
            elif label_as_tick:
                plt.xticks(xs, xs)

    if complete_figure:
        sns.despine()
        if set_precision:
            plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
        if title and plot_title:
            plt.title(title)
        if ylog_scale:
            plt.yscale('log')
            plt.grid(True, which = "both", ls = "--")       # for log scale graph, manual grid should be activated
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if legend_loc != "disable":
            plt.legend(loc = legend_loc)
    if draw_now:
        if save_image:
            plt.savefig("./output.png", dpi = 400)
        plt.show()

# Sample data
if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 5, 6])
    all_y = []
    for i in range(3):
        y = np.random.rand(6, 16) * 0.5
        y += np.random.rand(6, 1) * 2
        all_y.append(y)
    labels       = ['$\sigma_s = $'] * 3
    label_suffix = [0.2, 0.4, 0.6]
    lineplot_with_errbar(x, all_y, labels = labels, label_suffix = label_suffix, complete_figure = True, draw_now = True)
    