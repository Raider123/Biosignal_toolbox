from biosignal_toolbox.eego_amp_lib import ImpedancePlotting


plot = ImpedancePlotting(path_to_so_file='/home/dfki.uni-bremen.de/nkueper/Dokumente/DFKI_Job/EXPECT/biosignal_toolbox/clients_new',
                         electrodes='CW-05626',
                         preset_high=[15, 25],
                         preset_low=[5, 15],
                         refresh_delay=0.5,
                         markersize=1500,
                         fontsize=15)

plot.startImpedanceCheck()



