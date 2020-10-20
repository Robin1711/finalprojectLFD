from analyze_data import *
import matplotlib.pyplot as plt
import numpy as np

TOTAL_NEWSPAPERS = False
TOTAL_COUNTRIES = False
TOTAL_ORIENTATIONS = True
COP_COUNTRIES = False
COP_ORIENTATIONS = False

CUSTOM_NEWSPAPER_NAMES = [
    'The Australian' , 'Sydney Morning Herald', 'The Age'
    , 'The Times', 'The Hindu'
    , 'The Times', 'Mail & Guardian'
    , 'The Washington Post', 'The New York Times'
    ]

def set_plot_font_sizes(plt, scalar):
    small = scalar * 8
    medium = scalar * 10
    big = scalar * 12
    plt.rc('font', size=small)  # controls default text sizes
    plt.rc('axes', titlesize=medium)  # fontsize of the axes title
    plt.rc('axes', labelsize=medium)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=medium)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=medium)  # fontsize of the tick labels
    plt.rc('legend', fontsize=small)  # legend fontsize
    plt.rc('figure', titlesize=big)  # fontsize of the figure title
    return plt

################################################################################################

if TOTAL_NEWSPAPERS or TOTAL_COUNTRIES or TOTAL_ORIENTATIONS:
    # cop_stats = get_stats([20,21,22,23,24])
    cop_stats = get_stats()
    cop_stats = cop_stats['counts']


################################################################################################
################################################################################################
################################################################################################

if TOTAL_NEWSPAPERS:
    print("CREATING PLOT TOTAL NEWSPAPERS")
    scalar = 1
    plt = set_plot_font_sizes(plt, scalar)
    fs = tuple([scalar*x for x in (6.4,4.8)])
    fig_all_newspapers = plt.subplots(figsize=fs)
    fig_all_newspapers[0].suptitle('Amount of newspapers over all COPs')

    # Set width bar
    barWidth = 1

    # Set height bar
    BARS_newspaper = cop_stats['newspapers'].values()

    # Set position of bar on X-axis
    br_newspaper = [x + barWidth/2 for x in np.arange(len(BARS_newspaper))]

    # Make the plot
    colors_newspaper=['b', 'b', 'b', 'y', 'y', 'g', 'g', 'r', 'r']
    plt.bar(br_newspaper, BARS_newspaper, color=colors_newspaper, width=barWidth, edgecolor='grey')

    # Adding Xticks
    plt.xlabel('newspaper', fontweight='bold')
    plt.ylabel('# articles', fontweight='bold')
    plt.xticks([r + barWidth/2 for r in range(len(BARS_newspaper))],
               CUSTOM_NEWSPAPER_NAMES, rotation=80, fontsize=9)
    plt.legend()
    plt.show()


################################################################################################
################################################################################################
################################################################################################

if TOTAL_COUNTRIES:
    print("CREATING PLOT TOTAL COUNTRIES")
    scalar = 0.8
    plt = set_plot_font_sizes(plt, scalar)
    fs = tuple([scalar*x for x in (6.4,4.8)])
    fig_all_countries = plt.subplots(figsize=fs)
    fig_all_countries[0].suptitle('Amount of newspapers from countries over all COPs')

    # Set width bar
    barWidth = 1

    # Set height bar
    BARS_countries = cop_stats['country'].values()

    # Set position of bar on X-axis
    br_countries = [x + barWidth / 2 for x in np.arange(len(BARS_countries))]

    # Make the plot
    colors_countries=['b', 'y', 'g', 'r']
    plt.bar(br_countries, BARS_countries, color=colors_countries, width=barWidth, edgecolor='grey')

    # Adding Xticks
    plt.xlabel('country', fontweight='bold')
    plt.ylabel('# articles', fontweight='bold')
    plt.xticks([r + barWidth / 2 for r in range(len(BARS_countries))],
               cop_stats['country'].keys(), rotation=80)
    plt.legend()
    plt.show()


################################################################################################
################################################################################################
################################################################################################

if TOTAL_ORIENTATIONS:
    print("CREATING PLOT TOTAL ORIENTATIONS")
    scalar = 0.5
    plt = set_plot_font_sizes(plt, scalar)
    fs = tuple([scalar*x for x in (6.4,4.8)])
    fig_all_orientations = plt.subplots(figsize=fs)
    fig_all_orientations[0].suptitle('Amount of political orientations over all COPs')

    # Set width bar
    barWidth = 0.9

    # Set height bar
    BARS_orientations = cop_stats['labels'].values()

    # Set position of bar on X-axis
    br_orientations = [x + barWidth / 2 for x in np.arange(len(BARS_orientations))]

    # Make the plot
    colors_orientations=['b', 'r']
    plt.bar(br_orientations, BARS_orientations, color=colors_orientations, width=barWidth, edgecolor='grey')

    # Adding Xticks
    plt.xlabel('Policital Orientation', fontweight='bold')
    plt.ylabel('# articles', fontweight='bold')
    plt.xticks([r + barWidth / 2 for r in range(len(BARS_orientations))],
               cop_stats['labels'].keys())
    plt.legend()
    plt.show()


################################################################################################
################################################################################################
################################################################################################

if COP_COUNTRIES or COP_ORIENTATIONS:
    cop_stats = get_stats_per_cop()
    # cop_stats = get_stats_per_cop([20,21,22,23,24])

if COP_ORIENTATIONS:
    print("CREATING PLOT COP ORIENTATIONS")
    scalar = 3
    plt = set_plot_font_sizes(plt, scalar)
    fs = tuple([scalar*x for x in (6.4,4.8)])
    fig_cop_orientations = plt.subplots(figsize=fs)
    fig_cop_orientations[0].suptitle('The political orientation divided over de COPs')

    # Set width bar
    barWidth = 1/3

    # Set height bars
    BARS_orientations = list()
    for cop in cop_stats:
        orientation = cop_stats[cop]['counts']['labels']
        BARS_orientations.append(orientation)
    BARS_left = [d['Left-Center'] for d in BARS_orientations]
    BARS_right = [d['Right-Center'] for d in BARS_orientations]

    # Set position of bar on X-axis
    br_left_orientations = [x - barWidth/2 for x in np.arange(len(BARS_left))]
    br_right_orientations = [x + barWidth/2 for x in np.arange(len(BARS_right))]

    # Make the plot
    plt.bar(br_left_orientations, BARS_left, color='b', width=barWidth, edgecolor='grey', label='Left, total = '+str(sum(BARS_left)))
    plt.bar(br_right_orientations, BARS_right, color='r', width=barWidth, edgecolor='grey', label='Right, total = '+str(sum(BARS_right)))

    # Adding Xticks
    plt.xlabel('COP number', fontweight='bold')
    plt.ylabel('# articles', fontweight='bold')
    plt.xticks([r for r in range(len(BARS_right))],
               cop_stats.keys())
    plt.legend()
    plt.show()


################################################################################################
################################################################################################
################################################################################################

if COP_COUNTRIES:
    print("CREATING PLOT COP COUNTRIES")
    scalar = 3
    plt = set_plot_font_sizes(plt, scalar)
    fs = tuple([scalar*x for x in (6.4,4.8)])
    fig_cop_countries = plt.subplots(figsize=fs)
    fig_cop_countries[0].suptitle('The countries divided over de COPs')

    # Set width bar
    barWidth = 1/5

    # Set height bars
    BARS_countries = list()
    for cop in cop_stats:
        countries = cop_stats[cop]['counts']['country']
        BARS_countries.append(countries)

    BARS_australia = [d['Australian'] for d in BARS_countries]
    BARS_india = [d['India'] for d in BARS_countries]
    BARS_sa = [d['South Africa'] for d in BARS_countries]
    BARS_us = [d['United States'] for d in BARS_countries]

    # Set position of bar on X-axis
    barWidth=0.2
    br_country_australia = [x - barWidth*6/4 for x in np.arange(len(BARS_australia))]
    br_country_india = [x - barWidth*2/4 for x in np.arange(len(BARS_india))]
    br_country_sa = [x + barWidth*2/4 for x in np.arange(len(BARS_sa))]
    br_country_us = [x + barWidth*6/4 for x in np.arange(len(BARS_us))]

    # Make the plot
    plt.bar(br_country_australia, BARS_australia, color='b', width=barWidth, edgecolor='grey', label='Australia, total = '+str(sum(BARS_australia)))
    plt.bar(br_country_india, BARS_india, color='y', width=barWidth, edgecolor='grey', label='India, total = '+str(sum(BARS_india)))
    plt.bar(br_country_sa, BARS_sa, color='g', width=barWidth, edgecolor='grey', label='Souht Africa, total = '+str(sum(BARS_sa)))
    plt.bar(br_country_us, BARS_us, color='r', width=barWidth, edgecolor='grey', label='United States, total = '+str(sum(BARS_us)))

    # Adding Xticks
    plt.xlabel('COP number', fontweight='bold')
    plt.ylabel('# articles', fontweight='bold')
    plt.xticks([r for r in range(len(BARS_australia))],
               cop_stats.keys())
    plt.legend()
    plt.show()
