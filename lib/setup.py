import setuptools

setuptools.setup(
    include_package_data=True,
    name='biosignal_toolbox',
    version='0.1.0',
    description='biosignal_toolbox',
    url='https://git.hb.dfki.de/dfki_ude/biosignal_toolbox.git',
    author='DFKI',
    packages=setuptools.find_packages(),
    install_requires=[
        'matplotlib',
        'mne',
        'numpy',
        'scikit_learn',
        'scipy',
        'setuptools',
        'tensorflow==2.9.3',
        'seaborn==0.12.2', 
        'numba', 
        'PyWavelets', 
        'pandas'
    ],
    classifiers=["Programming langugage :: Python :: Version > 3.6",
                 "Operating System :: OS independent"],
)
