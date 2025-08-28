
#! ************************************************
#! Imports
#! ************************************************

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from yaml import safe_load
from datetime import datetime
from types import SimpleNamespace
from sys import exit, path
path.append(str(Path(__file__).resolve().parents[2]))
from config_root import project_root
import warnings
import inspect
#! ************************************************
#! Custom Warning function
#! ************************************************

def customWarningFormat(message, category, filename, lineno, line=None):
    """
    Format warnings in a compact way by including only the category, 
    function name, and message.

    Parameters
    ----------
    message : str
        The warning message that is generated.
    category : Warning
        The category of the warning (e.g., `UserWarning`, `DeprecationWarning`).
    filename : str
        The file name where the warning originated.
    lineno : int
        The line number where the warning was triggered.
    line : str, optional
        The line of source code that generated the warning (if available).

    Returns
    -------
    str
        A formatted warning string containing the warning category, 
        the function name where it was raised, and the message.

    Notes
    -----
    This custom formatter uses the `inspect` module to extract the 
    function name from the call stack. It overrides the default 
    `warnings.formatwarning` output, which normally includes the 
    file path and line number.
    """
    func_name = "<unknown>"
    for frameinfo in inspect.stack()[2:]:
        mod = inspect.getmodule(frameinfo.frame)
        if mod and mod.__name__ != "warnings":
            func_name = frameinfo.function
            break

    short_file = Path(filename).name
    return f"{category.__name__} in {func_name}() [{short_file}:{lineno}]: {message}\n"

warnings.formatwarning = customWarningFormat

#! ************************************************
#! Utility Functions
#! ************************************************

def loadConfig(filename=''):
    """
    This function safely loads the yaml configuration file into a dictionary.
    It also converts the dict to a Namespace object wherein keys become attributes and can be accessed by '.' operator.

    Parameters
    ----------
    filename : str
        name of the config file
    
    Author
    ------
    Author: Kartik Chari \n
    Last changed: 21.08.2025 (by Kartik Chari)
    """
    try:
        project_root = getProjectRoot()
        config_dir = project_root / 'config'
        with open(config_dir / filename, 'r') as file:
            dict_obj = safe_load(file)
        return convertDictToNamespace(dict_obj)
    except IsADirectoryError:
        print("ERROR: Please enter correct config filename!!")
        exit(1)

def convertDictToNamespace(data_inp):
    if isinstance(data_inp, dict):
        return SimpleNamespace(**{k: convertDictToNamespace(v) for k, v in data_inp.items()})
    elif isinstance(data_inp, list):
        return [convertDictToNamespace(i) for i in data_inp]
    else:
        return data_inp


def getProjectRoot(marker=""):
    """
    This function returns the project root folder path

    Parameters
    -----
    marker: str, optional
        The dir name that helps to looks for the root folder in any project structure, by default "".

    Author
    ------
    Author: Kartik Chari \n
    Last changed: 28.08.2025 (by Kartik Chari)
    """
    filepath = Path(__file__).resolve()
    if not marker:
        return filepath.parent.parent.parent
    else:
        for parent in filepath.parents:
            if parent.name == marker:
                return parent.parent
        raise RuntimeError(f"Project root not found. Looked for {marker}")


def getAbsolutePath(input_path=''):
    """
    This function ensures that the input path is absolute. It first gets the root of the project and then appends the input path to it. This function does not work with relative paths.

    Parameters
    -----
    input_path: str
        Path (wrt project root) to be made absolute, by default empty.
    Author
    ------
    Author: Kartik Chari \n
    Last changed: 27.08.2025 (by Kartik Chari)
    """
    if not input_path:
        raise ValueError("Please provide input path and/or project root path!")
    # get project root
    project_root = getProjectRoot()

    input_path = Path(input_path).expanduser()
    if input_path.is_absolute():
        try:
            # check if input_path is already inside project_root
            relative_part = input_path.resolve().relative_to(project_root)
        except ValueError:
            # break the path into pieces and strip the first '/'
            relative_part = Path(*input_path.parts[1:])
    else:
        relative_part = input_path

    return (project_root / relative_part).resolve()


def createOutputDir(param_obj=None, suffix_str=''):
    """
    This function checks whether the fig_save_path exists or not. If it does not exist, it will create the directory. In addition, it will create the output dir where the output plots and readme will be saved.
    The output dir will look like fig_save_path/sub_code/date_time_plot_1. It also checks the last integer and increments it to prevent overwrite.
    
    Parameters
    ----------
    param_obj: dict
        param dict imported from the .yaml file, by default None.
    suffix_str: str
        suffix for the output dir, by default "".
        
    Returns
    -----
    Path
        path of the output directory.
    
    Author
    ------
    Author: Kartik Chari \n
    Last changed: 13.11.2025 (by Kartik Chari)
    """
    if param_obj is None:
        raise RuntimeError("Please provide the yaml config object!!")
    if not suffix_str:
        suffix_str = "plot"
    
    inp_parent_dir = getAbsolutePath(param_obj.filepath.fig_save_path)
    parent_dir = inp_parent_dir / f"{param_obj.data_param.subject_code}"
    # ensure the input directory exists
    inp_parent_dir.mkdir(parents=True, exist_ok=True)
    # prepare timestamp-based folder name
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    base_name = f"{timestamp}_{suffix_str}"
    # check existing folders to find max x
    existing_dirs = [d for d in inp_parent_dir.iterdir() if d.is_dir() and d.name.startswith(base_name)]
    numbers = []
    for d in existing_dirs:
        suffix = d.name[len(base_name):].lstrip("_")  # get the number after '_'
        if suffix.isdigit():
            numbers.append(int(suffix))
    next_number = max(numbers) + 1 if numbers else 1
    new_dir = inp_parent_dir / f"{base_name}{next_number}"
    # create the new folder
    new_dir.mkdir()
    return new_dir


def createReadme(param_obj=None, dir_path=None):
    """
    This function creates a readme file in the same dir as the plots with all necessary hyperparameters of the model.

    Parameters
    ----------
    param_obj : dict
        param dict imported from the .yaml file, by default None
    dir_path: pathlib Path obj
        path for the readme file
    
    Author
    ------
    Author: Kartik Chari \n
    Last changed: 13.11.2025 (by Kartik Chari)
    """
    if dir_path is None:
        print("WARNING: No dir_path specified! Creating readme file in the fig_save_path...")
        dir_path = Path(param_obj.filepath.fig_save_path)
    
    try:
        readme_file = dir_path / "readme.txt"
        with readme_file.open("w") as f:
            f.write(f"Scenario: {param_obj.data_param.weights, param_obj.data_param.mov_type}\n") 
            f.write(f"Subject Code: {param_obj.data_param.subject_code}\n")
            f.write(f"Batch Size: {param_obj.model_param.batch_size}\n")
            f.write(f"Epochs: {param_obj.model_param.n_epochs}")
            f.write(f"Feature Selection: {param_obj.preprocess_param.feature_select}\n")
            f.write(f"Window Size X: {param_obj.preprocess_param.window_size_x}\n")
            f.write(f"Window Size Y: {param_obj.preprocess_param.window_size_y}\n")
            f.write(f"Window Step: {param_obj.preprocess_param.window_step}\n")
            f.write(f"\n")
            f.write(f"HPF Filter: {param_obj.preprocess_param.f_cutoff_hpf}Hz\n")
            f.write(f"Variance Filter width: {param_obj.preprocess_param.var_filter_width}\n")
            f.write(f"MVC: {param_obj.preprocess_param.mvc}\n")
            f.write(f"LPF Filter: {param_obj.preprocess_param.f_cutoff_lpf}Hz\n")
            f.write(f"Force Activation: {param_obj.preprocess_param.act_delay, param_obj.preprocess_param.act_beta1, param_obj.preprocess_param.act_beta2, param_obj.preprocess_param.act_gamma, param_obj.preprocess_param.act_A}\n")
            f.write(f"\n")
            f.write(f"BPNN Neurons: {param_obj.model_param.neurons_h1, param_obj.model_param.neurons_h2}\n")
            f.write(f"BPNN Act functions: {param_obj.model_param.act_inp, param_obj.model_param.act_h1, param_obj.model_param.act_h2}\n")
            f.write(f"Early Stopping: {param_obj.model_param.is_early_stop}\n")
            f.write(f"Post Processing Filter type: {param_obj.post_train_param.filter_type}\n")
            f.write(f"Post Processing Filter size: {param_obj.post_train_param.filter_size}\n")
    
    except KeyError:
        print("ERROR!! Please ensure you pass non-empty dict object!")
        exit(1)
    
    finally:
        f.close()


def plotResults(data_ref=[], label_ref="real", data_out=[], label_out="predicted", title="", xlabel="Time in s", ylabel="", is_grid_on=True):
    """
    This function plots the result of the BPNN model

    Parameters
    ----------
    data_ref : array
        array of reference data
    label_ref : str, optional
        label for ref legend, by default "real"
    data_out : array
        array of predicted data
    label_out : str, optional
        label for predicted data legend, by default "predicted"
    title : str, optional
        title for the plot, by default ""
    xlabel : str, optional
        xlabel for the plot, by default ""
    ylabel : str, optional
        ylabel for the plot, by default ""
    is_grid_on : bool, optional
        boolean to decide grid lines visibility, by default True
    
    Author
    ------
    Author: Kartik Chari \n
    Last changed: 13.11.2025 (by Kartik Chari)
    """
    plt.figure()

    x_samples = np.arange(0, len(data_ref),1)
    plt.plot(x_samples, data_ref, ls="dashed", label=label_ref)
    plt.plot(x_samples, data_out, label=label_out)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if is_grid_on:
        plt.grid()


