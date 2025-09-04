
#! ************************************************
#! Imports
#! ************************************************

import numpy as np
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from yaml import safe_load
from datetime import datetime
from types import SimpleNamespace
from typing import Type, Union, List

import warnings
import inspect
#! ************************************************
#! Custom Warning function
#! ************************************************

def customWarningFormat(message: str, category: Type[Warning], filename: str, lineno: int, line: str=None) -> str:
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

def loadConfig(filename: str = None) -> dict:
    """
    This function safely loads the yaml configuration file into a dictionary.
    It also converts the dict to a Namespace object wherein keys become attributes and can be accessed by '.' operator.

    Parameters
    ----------
    filename : str
        Name of the config file
    
    Returns
    -------
    dict
        Dictionary containing all config parameters from the yaml file
    
    Raises
    ------
    IsADirectoryError
        Raised if the filename is a dir and not a file
    TypeError
        Raised if the filename is None or not provided

    
    Author
    ------
    Author : Kartik Chari \n
    Last changed : 21.08.2025 (by Kartik Chari)
    """
    try:
        project_root = getProjectRoot()
        # prepend 'config/' only if filename does not consist of this
        if Path(filename).parts[0] != 'config':
            filename = 'config/' + filename

        with open(project_root / filename, 'r') as file:
            dict_obj = safe_load(file)
        return convertDictToNamespace(dict_obj)
    except (IsADirectoryError, TypeError):
        print(f"ERROR: Please enter correct config filename in loadConfig!!")
        exit(1)

def convertDictToNamespace(data_inp: Union[dict, list]) -> SimpleNamespace:
    """
    This function converts the input into a namespace that can be accessed like an attribute using a '.' operator.

    Parameters
    ----------
    data_inp : Union[dict, list]
        Input dict or list

    Returns
    -------
    SimpleNamespace
        Namespace of dict/list elements
    
    Author
    ------
    Author : Kartik Chari \n
    Last changed : 21.08.2025 (by Kartik Chari)
    """
    if isinstance(data_inp, dict):
        return SimpleNamespace(**{k: convertDictToNamespace(v) for k, v in data_inp.items()})
    elif isinstance(data_inp, list):
        return [convertDictToNamespace(i) for i in data_inp]
    else:
        return data_inp


def convertNamespaceToDict(data_inp: SimpleNamespace) -> dict:
    """
    This function converts the input nested namespace/dict  into a dict.

    Parameters
    ----------
    data_inp : SimpleNamespace
        Input NameSpace object

    Returns
    -------
    dict
        Dictionary converted from SimpleNamespace
    
    Author
    ------
    Author : Kartik Chari \n
    Last changed : 04.09.2025 (by Kartik Chari)
    """
    if isinstance(data_inp, SimpleNamespace):
        return {k: convertNamespaceToDict(v) for k, v in vars(data_inp).items()}
    elif isinstance(data_inp, dict):
        return {k: convertNamespaceToDict(v) for k, v in data_inp.items()}
    elif isinstance(data_inp, (list, tuple)):
        return [convertNamespaceToDict(v) for v in data_inp]
    else:
        return data_inp


def getProjectRoot(marker: str = None) -> Path:
    """
    This function returns the project root folder path.

    Parameters
    -----
    marker: str, optional
        The dir name that helps to looks for the root folder in any project structure, by default ""
    
    Returns
    -------
    Path
        Project root filepath
    
    Raises
    ------
    RuntimeError
        Raised if the marker is not found in the filepath stems

    Author
    ------
    Author : Kartik Chari \n
    Last changed : 28.08.2025 (by Kartik Chari)
    """
    filepath = Path(__file__).resolve()
    if not marker:
        print("No markers provided to get project root! Assuming that there are not further sub-divisions in your folder structure!!")
        return filepath.parent.parent.parent
    else:
        for parent in filepath.parents:
            if parent.name == marker:
                return parent.parent
        raise RuntimeError(f"Project root not found. Looked for {marker}")


def getAbsolutePath(input_path: str = None) -> Path:
    """
    This function ensures that the input path is absolute. It first gets the root of the project and then appends the input path to it. This function does not work with relative paths.

    Parameters
    -----
    input_path: str
        Path (wrt project root) to be made absolute, by default empty
    
    Returns
    -------
    Path
        Absolute path of the input_path
    
    Returns
    -------
    ValueError
        Raised if the input_path is None
    ValueError
        Raised if project root is not generated properly

    Author
    ------
    Author : Kartik Chari \n
    Last changed : 27.08.2025 (by Kartik Chari)
    """
    if input_path is None:
        raise ValueError("Please provide a valid input path !!")
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


def createOutputDir(param_obj: SimpleNamespace = None, suffix_str: str = None) -> Path:
    """
    This function checks whether the fig_save_path exists or not. If it does not exist, it will create the directory. In addition, it will create the output dir where the output plots and readme will be saved.
    The output dir will look like fig_save_path/sub_code/date_time_plot_1. It also checks the last integer and increments it to prevent overwrite.
    
    Parameters
    ----------
    param_obj: dict
        param dict imported from the .yaml file, by default None
    suffix_str: str
        suffix for the output dir, by default ""
        
    Returns
    -----
    Path
        path of the output directory
    
    Raises
    ------
    RuntimeError
        Raised if param_obj is None
    
    Author
    ------
    Author : Kartik Chari \n
    Last changed : 13.11.2024 (by Kartik Chari)
    """
    #TODO: Make it general
    if param_obj is None:
        raise RuntimeError("Please provide the yaml config object!!")
    if suffix_str is None:
        warnings.warn(f"Suffix_str is {suffix_str}. Using \"temp\" instead!!")
        suffix_str = "temp"
    
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


def createReadme(param_obj: Union[SimpleNamespace, dict] = None, dir_path: Path = None, sections_to_include: List[str] = None) -> None:
    """
    This function creates a readme file in the same dir as the plots with all necessary hyperparameters of the model.

    Parameters
    ----------
    param_obj : dict or SimpleNamespace
        Object imported from the .yaml file, by default None
    dir_path: Path
        Path of dir where readme file needs to be created
    sections_to_include: list of str
        Names of the sections that need to be included in the readme
    
    Raises
    ------
    TypeError
        Raised if param_obj is neither a dict nor SimpleNamespace
    KeyError
        Raised if the dict keys filepath or fig_save_path do not exist
    
    Author
    ------
    Author : Kartik Chari \n
    Last changed : 04.09.2025 (by Kartik Chari)
    """
    try:
        # convert SimpleNamespace → dict only if needed
        if isinstance(param_obj, SimpleNamespace):
            param_dict = convertNamespaceToDict(param_obj)
        elif isinstance(param_obj, dict):
            param_dict = convertNamespaceToDict(param_obj)
        else:
            raise TypeError("param_obj must be a dict or SimpleNamespace!!")

        if dir_path is None:
            warnings.warn("Missing dir_path argument!! Trying to access fig_save_path from config instead!!")
            if "filepath" not in param_dict or "fig_save_path" not in param_dict["filepath"]:
                raise KeyError("Missing filepath.fig_save_path in config file!!!")
            dir_path = getAbsolutePath(param_dict["filepath"]["fig_save_path"])
    
        readme_file = dir_path / "readme.txt"
        readme_file.parent.mkdir(parents=True, exist_ok=True)

        # filter sections if needed
        if sections_to_include is not None:
            filter_dict = {}
            for sec in sections_to_include:
                if sec in param_dict:
                    filter_dict[sec] = param_dict[sec]
                else:
                    filter_dict[sec] = "(NOT FOUND)"
        else:
            filter_dict = param_dict
        
        with readme_file.open("w") as f:
            f.write("Configuration Used for Training!!\n\n")
            yaml.dump(filter_dict, f, sort_keys=False, default_flow_style=False)
        
        print(f"Readme created at: {readme_file}")

    except KeyError:
        print("ERROR!! Please ensure you pass correct args to createReadme!!")
        exit(1)
    

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
    Author : Kartik Chari \n
    Last changed : 13.11.2024 (by Kartik Chari)
    """
    #TODO: Improve to make it more general
    plt.figure()

    x_samples = np.arange(0, len(data_ref),1)
    plt.plot(x_samples, data_ref, ls="dashed", label=label_ref)
    plt.plot(x_samples, data_out, label=label_out)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if is_grid_on:
        plt.grid()


