import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

def plot_heat_flux(csv_path, save_plot=True, output_dir="~/Desktop/het_fuels_proj/plots", smooth_param=7): #,smooth=True, smooth_type="gaussian", smooth_param=10): #csv_path, save_plot, ouput_dir): #
    """
    reads heat flux data from a csv file and generates a time resolved lineplot.

    Parameters:
    - csv_path (str): path to the CSV file containing heat flux data.
    - save_plot (bool): whether to save the plot as an image (default=True).
    - output_dir (str): directory to save the plot if save_plot=True.
    - smooth (bool): whether to smooth the data (default=True).
    - smooth_param (int): smoothing strength (sigma for gaussian, window size for moving average).
    """

    # expand `~` to the full home directory path
    csv_path = os.path.expanduser(csv_path)
    output_dir = os.path.expanduser(output_dir)

    # ensure the csv file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # read csv
    df = pd.read_csv(csv_path, header=0) 
    #print(f'csv path: {csv_path}')

    # ensure 'time [s]' column exists
    if "time [s]" not in df.columns:
        raise ValueError("Error: CSV does not contain a 'time [s]' column.")

    # filter data to only include time ≤ 120s (row index ≤ 83)
    df = df[df["time [s]"] <= 121]

    # extract time column for x-axis
    time_values = df["time [s]"]

    # initialize plot
    plt.figure(figsize=(10, 6))

    # plot each simulation's heat flux data
    for column in df.columns:
        if column != "time [s]":  # exclude time column
            data = df[column].copy()
            # 1: interpolate missing values to smooth step-like regions
            data = data.interpolate(method="linear")

            # 2: apply savitzky-golay filter for additional smoothing
            data = savgol_filter(data, window_length=min(smooth_param, len(data)//2*2-1), polyorder=2)
            
            # apply smoothing if enabled
            # if smooth:
            #     if smooth_type == "gaussian": 
            #         #print(f"applying {smooth_type} smoothing with parameter: {smooth_param}")
            #         #print(f"before smoothing (first 5 values of {column}): {data[:5]}")
            #         data = gaussian_filter1d(data, smooth_param) #sigma=smooth_param)
            #     elif smooth_type == "moving_average":
            #         data = data.rolling(window=smooth_param, center=True).mean()
      
            plt.plot(time_values, data[column], label=column, linewidth=2)

    # formatting the plot
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Fire Intensity [MW/m²]", fontsize=12)
    plt.title("Post Ignition Fire Intensity for Base Cases and 10(live)/90(dry) % Fuel Loading", fontsize=14)
    plt.legend(title='% Moisture') #, fontsize=10)
    plt.xlim(50, 120)
    plt.ylim(-1, 17.5)
    #plt.grid(True)

    # save the plot
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)  # create the directory if it doesn't exist
        plot_path = os.path.join(output_dir, "heat_flux_____________________.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

    plt.show()
    
def plot_fuel_consumption(csv_path, output_dir, save_plot): 
    """
    reads csvs, categorizes, and plots data. 
    """
    
    # load csv
    # csv_path = "postprocessing_data_all_5625.csv"
    output_dir = os.path.expanduser(output_dir)  # expand ~ to absolute path
    df = pd.read_csv(csv_path, skiprows=0)  # read the csv first row as headers

    # rename columns for clarity
    df.columns = ["Category", "Subcategory", "Grid FC [%]", "FC Normalized by 0% Humidity 100% Dead Fuel Case [%]"]#, "Overall Spread Rate [m/s]", "Max Flame Depth [m]"] #, "Total Grid FC [%]"]

    # drop any completely empty rows
    df = df.dropna(how="all")

    # convert numerical columns to floats, ignoring errors for non-numeric rows 
    # df["FC Normalized by 0% MC case [%]"] = pd.to_numeric(df["FC Normalized by 0% MC case [%]"], errors="coerce")
    #df["Total Grid FC [%]"] = pd.to_numeric(df["Total Grid FC [%]"], errors="coerce")

    # forward-fill categories to assign proper labels
    # df["Category"] = df["Category"].ffill()
    # drop any remaining NaN values (e.g., empty subcategories)
    df = df.dropna()
    
    # initialize plot
    plt.figure(figsize=(10, 6))

    # get unique categories
    categories = df["Category"].unique()
    
    #fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
    fig, axes = plt.plot

    # define colors for differentiation
    #colors = ["red", "green", "blue", "purple"]#, "orange"]
    # category_colors = {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}
    
    # **Subplot 1: Base Case (Scatter Plot Only)**
    base_subset = df[df["Category"] == "base"]
    axes[0].bar(base_subset["Subcategory"], base_subset["FC Normalized by 0% Humidity 100% Dead Fuel Case [%]"])#, color="red", label="Base", s=50)
    axes[0].set_title("Base Cases Fuel Consumption")
    axes[0].set_xlabel("% Moisture Content")
    axes[0].set_ylabel("FC Normalized by 0% Humidity 100% Dead Fuel Case [%]")

    # loop through each category and plot
    for category in categories:
        if category != "base":  # exclude base case
            subset = df[df["Category"] == category]

            # convert subcategories to numeric if possible
            try:
                subset["Subcategory"] = pd.to_numeric(subset["Subcategory"])
            except:
                pass  # if subcategories contain text, leave them as is
            
             # scatter plot
            axes[1].scatter(subset["Subcategory"], subset["FC Normalized by 0% Humidity 100% Dead Fuel Case [%]"]) #, color=category_colors[category], label=category, s=50)

            # line plot
            # axes[1].plot(subset["Subcategory"], subset["FC Normalized by 0% Humidity 100% Dead Fuel Case [%]"], color=category_colors[category], linestyle="-")

            # scatter plot (individual points)
            #plt.scatter(subset["Subcategory"], subset["FC Normalized by 0% MC case [%]"], color=category_colors[category], label=category, s=50)

            # line plot (trend within category)
            #plt.plot(subset["Subcategory"], subset["FC Normalized by 0% MC case [%]"], color=category_colors[category], linestyle="-")
    axes[1].set_title("Study Cases Fuel Consumption")
    axes[1].set_xlabel(" ")
    axes[1].legend(title="Study (live/dead)")
    
    plt.tight_layout()

    # formatting
    # plt.xlabel("Subcategory")
    # plt.ylabel("FC Normalized by 0% MC case [%]")
    # plt.title("Fuel Consumption Normalized by 0% Moisture Content Case")
    # plt.legend(title="Study")
    # # plt.grid(True)
    # plt.show()
    
    plt.show()
    os.makedirs(output_dir, exist_ok=True)  # create the directory if it doesn't exist
    plot_path = os.path.join(output_dir, "fuel_consumption_subplot.png")
    fig.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

def plot_fsr(csv_path="~/Desktop/het_fuels_proj/fire_spread_rate_data.csv", save_plot=True, output_dir="~/Desktop/het_fuels_proj/plots"): 
    """
    reads csvs, categorizes, and plots data. 
    """
    
    # load csv
    csv_path = "~/Desktop/het_fuels_proj/fire_spread_rate_data.csv"
    output_dir = os.path.expanduser(output_dir)  # expand pf to absolute path
    df = pd.read_csv(csv_path, skiprows=0)  # read the first row as headers

    # rename columns for clarity
    df.columns = ["Category", "Subcategory", "Fire Spread Rate [m/s]"] 

    # drop any completely empty rows
    df = df.dropna(how="all")

    # convert numerical columns to floats, ignoring errors for non-numeric rows 
    df["Fire Spread Rate [m/s]"] = pd.to_numeric(df["Fire Spread Rate [m/s]"], errors="coerce")
    #df["Total Grid FC [%]"] = pd.to_numeric(df["Total Grid FC [%]"], errors="coerce")

    # forward-fill categories to assign proper labels
    df["Category"] = df["Category"].ffill()
    # drop any remaining NaN values (e.g., empty subcategories)
    df = df.dropna()
    
    # initialize plot
    plt.figure(figsize=(10, 6))

    # get unique categories
    categories = df["Category"].unique()
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)

    # define colors for differentiation
    colors = ["red", "green", "blue", "purple"]#, "orange"]
    category_colors = {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}
    
    # subplot 1: base case (scatter plot only)
    base_subset = df[df["Category"] == "base"]
    axes[0].scatter(base_subset["Subcategory"], base_subset["Fire Spread Rate [m/s]"], color="red", label="Base", s=50, marker='^')
    axes[0].set_title("Base Cases Fire Spread Rate")
    axes[0].set_xlabel("% Moisture Content")
    axes[0].set_ylabel("Fire Spread Rate [m/s]")

    # loop through each category and plot
    for category in categories:
        if category != "base":  # Exclude base case
            subset = df[df["Category"] == category]

            # convert subcategories to numeric if possible
            try:
                subset["Subcategory"] = pd.to_numeric(subset["Subcategory"])
            except:
                pass  # if subcategories contain text, leave them as is
            
             # scatter plot
            axes[1].scatter(subset["Subcategory"], subset["Fire Spread Rate [m/s]"], color=category_colors[category], label=category, s=50, marker='^')

            # line plot
            axes[1].plot(subset["Subcategory"], subset["Fire Spread Rate [m/s]"], color=category_colors[category], linestyle="-")

    axes[1].set_title("Study Cases Fire Spread Rate")
    axes[1].set_xlabel(" ")
    axes[1].legend(title="Study (live/dead)")
    
    plt.tight_layout()
    plt.show()
    os.makedirs(output_dir, exist_ok=True)  # create the directory if it doesn't exist
    plot_path = os.path.join(output_dir, "fsr_plot.png")
    fig.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
def plot_study_rep():
    """
    Plots representative moisture contents for heterogeneous fuels study.
    """
    
    # studies = ["Study 1", "Study 2", "Study 3"]
    # cases = ["Case 1", "Case 2", "Case 3", "Case 4", "Case 5"]
    
    num_studies = 3
    num_cases = 5
    
    # Moisture contents for each study: [wet, dry]
    moisture_contents = [
        [[25, 25], [70, 20], [115, 15], [160, 10], [250, 0]],  # Study 1
        [[25, 25], [39.6, 18.7], [54.2, 12.5], [68.8, 6.22], [83.4, 0]],  # Study 2
        [[25, 25], [31.3, 18.8], [37.5, 12.5], [43.8, 6.25], [50, 0]]  # Study 3
    ]
    
    plt.figure(figsize=(12, 6)) 
    
    for study_idx in range(num_studies):
        x = range(1, num_cases + 1)
        
        # Plot wet fuel moisture content
        wet_mc = [mc[0] for mc in moisture_contents[study_idx]]
        plt.scatter([xi + study_idx * (num_cases + 1) for xi in x], wet_mc, color='blue', label='Wet Fuel' if study_idx == 0 else "")
        plt.plot([xi + study_idx * (num_cases + 1) for xi in x], wet_mc, color='blue')
        
        # Plot dry fuel moisture content
        dry_mc = [mc[1] for mc in moisture_contents[study_idx]]
        plt.scatter([xi + study_idx * (num_cases + 1) for xi in x], dry_mc, color='red', label='Dry Fuel' if study_idx == 0 else "")
        plt.plot([xi + study_idx * (num_cases + 1) for xi in x], dry_mc, color='red')
    
    # for study_idx, study in enumerate(studies):
    #     x = [f"{study}\n{case}" for case in cases]
        
    #     # Plot wet fuel moisture content
    #     wet_mc = [mc[0] for mc in moisture_contents[study_idx]]
    #     plt.scatter(x, wet_mc, color='blue', label='Wet Fuel' if study_idx == 0 else "")
    #     plt.plot(x, wet_mc, color='blue')
        
    #     # Plot dry fuel moisture content
    #     dry_mc = [mc[1] for mc in moisture_contents[study_idx]]
    #     plt.scatter(x, dry_mc, color='red', label='Dry Fuel' if study_idx == 0 else "")
    #     plt.plot(x, dry_mc, color='red')
    
    plt.xlabel('Case #')
    plt.ylabel('Moisture Content [%]')
    #plt.title('Moisture Content for Wet and Dry Fuels Across Studies and Cases')
    plt.legend()
    
    # Set x-ticks and labels
    all_ticks = [i + j*(num_cases+1) for j in range(num_studies) for i in range(1, num_cases+1)]
    plt.xticks(all_ticks, [str(i) for i in range(1, num_cases+1)] * num_studies)
    
    # Add study labels
    for i in range(num_studies):
        plt.text((i+0.5)*(num_cases+1), plt.ylim()[1], f'Study {i+1}', 
                 horizontalalignment='center', verticalalignment='bottom')
    
    # Add vertical lines to separate studies
    for i in range(1, num_studies):
        plt.axvline(x=i*(num_cases+1) , color='gray', linestyle='--') #+ 0.5
    
    # Rotate x-axis labels for better readability
    #plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()    
    
# define in and out paths
csv_path="~/Desktop/het_fuels_proj/data/postprocessing_fc_all_5625.csv"
# save_plot=True
output_dir="~/Desktop/het_fuels_proj/plots"

# call functions/plot
# plot_heat_flux(csv_path, save_plot, output_dir, smooth_param=7)#smooth_type="moving average", smooth_param=10) 
#plot_fuel_consumption(csv_path, output_dir, save_plot=True)
# plot_fsr()
plot_study_rep()