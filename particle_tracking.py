import glob
import os
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import time
import sys
import numba as nb 
from tabulate import tabulate
import pandas as pd


import re


path_home = os.getcwd()

output_columns = ['atom_id', 'x', 'y', 'z', 'radius', 'vmag']

#Specify the header marker for the input data
header_marker='ITEM: ATOMS'

#Specify the column numbers for the input data
cn_atom_id = 0
cn_x = 1
cn_y = 2
cn_z = 3
cn_radius = 4
#set vx = -1 if the input data does not contain velocity information
#also set output_columns to exclude 'vmag' from the output
cn_vx = 5
cn_vy = 6
cn_vz = 7

#obtain all files with the file_prefix 'dump'
file_prefix = 'dump'

#number of samples to take from each group
#set equal to 0 for no sampling
sample_size = 500

#flag to use previous particle's location to fill as a placeholder for the current particle's location
#this is used so that Paraview pathlines do not disappear when the particle is not present in the current file
particle_position_placeholder = True


def obtain_filepaths(root_path, ext):
    import os
    all_files = []
    file_names = []
    for root, dirs, files in os.walk(root_path):
        for filename in files:
            if filename.lower().startswith(ext):
                all_files.append(os.path.join(root, filename))
                file_names.append(filename)
    return all_files,file_names

def npy_write(data,filename):
    np.save('out_'+filename+'.npy', data)

def npy_read(filename):
    return np.load('out_'+filename+'.npy')

def find_data_start_line(file_path, header_marker=header_marker):
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if header_marker in line:
                return i + 1  # Return the line number after the header line
    return None  # Return None if the header line is not found


def read_input_file(file, one_file_flag=True, header_marker=header_marker):
    if one_file_flag:
        data_start_line = find_data_start_line(file, header_marker)
        if data_start_line is not None:
            data = np.genfromtxt(file, skip_header=data_start_line, invalid_raise=False)
        else:
            print(f"Header line '{header_marker}' not found. Unable to read the data.")
            return None
    else:
        data = np.concatenate([np.genfromtxt(f, skip_header=11, invalid_raise=False) for f in file_list])

    num_rows, num_cols = data.shape
    print("number of rows (particle number): ", num_rows)

    atom_id = np.array(data[:, cn_atom_id])
    x = np.array(data[:, cn_x])
    y = np.array(data[:, cn_y])
    z = np.array(data[:, cn_z])
    radius = np.array(data[:, cn_radius])
    if not cn_vx < 0:
        print("calculating vmag...")
        vx = np.array(data[:, cn_vx])
        vy = np.array(data[:, cn_vy])
        vz = np.array(data[:, cn_vz])

        vmag = np.sqrt(vx**2 + vy**2 + vz**2)

        alldata = np.column_stack((atom_id, x, y, z, radius, vmag))
    else:
        alldata = np.column_stack((atom_id, x, y, z, radius))

    return alldata

def random_sample_row(data, num_samples = 1000, replacement = True):
    return data[np.random.choice(data.shape[0], num_samples, replace=replacement)]


def convert_to_dataframe(data, df_columns = output_columns,printID=False):

    df = pd.DataFrame(data)
    df.columns = df_columns

    if printID:
        df[['atom_id']].to_csv('initial_atom_id.csv', index=False,header = False)

    return df

#This function filters all files for ID's in the random sample
def filter_by_atom_id(df):
    atom_id_filter = np.genfromtxt('initial_atom_id.csv', delimiter=',')
    print(atom_id_filter)
    return df[df['atom_id'].isin(atom_id_filter)]

@nb.jit(nopython=True)
def custom_filter(arr, columnNumber,min,max):
        n, m = arr.shape
        result = np.empty((n, m), dtype=arr.dtype)
        k = 0
        for i in range(n):
                if arr[i, columnNumber] >= min:
                        if arr[i, columnNumber] < max:
                                result[k, :] = arr[i, :]
                                k += 1

        return result[:k, :].copy()

def group_and_output(df, groupby_cat='radius', columns_out=output_columns, filename='no_name'):
    # Grouping by unique values in the 'Category' column
    grouped = df.groupby(groupby_cat)

    # Printing the new DataFrames
    for category, df_group in grouped:
        create_folder = os.path.join(path_home, str(category))
        try:
            os.makedirs(create_folder, exist_ok=True)
            # print("Folder %s created!" % create_folder)
        except FileExistsError:
            pass

        name = 'dump_' + filename + '.csv'
        file_path = os.path.join(create_folder, name)
        df_group.to_csv(file_path, index=False, header=False)


#Function to group and output a random sample (as csv) of the data for each category (particle size)
#column headers are particle size and columns are the random sample ids
def group_and_output_write_samples(df, groupby_cat='radius', columns_out=output_columns, sample_size=100):
    # Grouping by unique values in the 'Category' column
    grouped = df.groupby(groupby_cat)

    # DataFrame to store random sample atom_ids
    sample_df = pd.DataFrame()

    # Printing the new DataFrames
    for category, df_group in grouped:

        # Take 100 random samples from each group
        random_samples = df_group.sample(n=min(sample_size, len(df_group)), replace=False)

        # Extract atom_ids from the random samples
        atom_ids_to_filter = random_samples['atom_id'].tolist()

        # Add a new column to the sample_df with category (particle size) as the header
        sample_df[category] = atom_ids_to_filter

    # Save random sample atom_ids to CSV
    sample_df.to_csv('random_sample_atom_ids.csv', index=False)

    stacked_sample_df = stacked_dataframe_columns(sample_df)

    return stacked_sample_df

def stacked_dataframe_columns(dataframe0):
    # Use pd.melt to stack multiple columns into a single column dynamically
    stack_df = pd.melt(dataframe0, var_name='radius', value_name='atom_id')
    stack_df.to_csv('stacked_sample_atom_ids.csv', index=False)
    return stack_df


#Function to group, read random sample ids and then outputs the filtered data
def group_and_read_filter_samples_write(df, previous_df, groupby_cat='radius', columns_out=output_columns, filename = 'no_fucking_name'):
    # Grouping by unique values in the 'Category' column
    grouped = df.groupby(groupby_cat)

    # DataFrame to store random sample atom_ids
    sample_df = pd.DataFrame()

    # Printing the new DataFrames
    for category, df_group in grouped:
        create_folder = os.path.join(path_home, str(category))
        try:
            os.makedirs(create_folder, exist_ok=True)
        except FileExistsError:
            pass
        sample_df = read_sample_df()
        print(sample_df)
        if sample_df is not None:
            atom_ids_to_filter = extract_atom_ids(sample_df, category)
            print("category: ", category)
            print(atom_ids_to_filter)
            df_group = filter_by_atom_id_with_placeholder(df_group, atom_ids_to_filter,previous_df)
        name = 'dump_' + filename + '.csv'
        file_path = os.path.join(create_folder, name)
        df_group.to_csv(file_path, index=False, header=False)

def read_sample_df(file_path='random_sample_atom_ids.csv'):
    try:
        sample_df = pd.read_csv(file_path)
        return sample_df
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

def extract_atom_ids(sample_df, category):
    print("sample_df:", sample_df)
    print("category:", category)
    print("category type:", type(str(category)))
    print("category in sample_df:", str(category) in sample_df.columns)
    if sample_df is not None and str(category) in sample_df.columns:
        atom_ids = sample_df[str(category)].tolist()
        return atom_ids
    else:
        print(f"Category '{category}' not found in the sample DataFrame.")
        return []

def determine_filename_number(filename):
    numeric_values = re.findall(r'\d+\.\d+|\d+', filename)
    numeric_values = [float(value) for value in numeric_values]
    return str(int(numeric_values[-1]))

# Modify filter_by_atom_id function to accept atom_id_filter directly
def filter_by_atom_id(df, atom_id_filter):
    print(atom_id_filter)
    return df[df['atom_id'].isin(atom_id_filter)]

def filter_by_atom_id_with_placeholder(df, atom_id_filter, previous_df):
    current_ids = df['atom_id'].unique()
    missing_ids = set(atom_id_filter) - set(current_ids)

    print("missing_ids: ", missing_ids)

    current_rows = df[df['atom_id'].isin(atom_id_filter)]
    
    if not missing_ids:
        # No missing IDs, return only current rows
        return current_rows

    missing_rows_from_previous = previous_df[previous_df['atom_id'].isin(missing_ids)]
    result_df = pd.concat([current_rows, missing_rows_from_previous], ignore_index=True)

    return result_df

def main():

    home_dir = os.getcwd()

    csv_files = glob.glob(os.path.join(home_dir ,'**/*/*.csv'), recursive=True)

    for file_path in csv_files:
        os.remove(file_path)


    #first obtains all filepaths that start with dump
    filepath_list,filename_list = obtain_filepaths(path_home,file_prefix)
    numFiles=len(filepath_list)
    print('number of files:',numFiles)
    print("filepath list: ", filepath_list)


    # obtain initial particle IDs from the first file (printID)
    initialdata = read_input_file(filepath_list[0])
    # initialdata = random_sample_row(initialdata, num_samples = 100, replacement = False)

    np.savetxt('initialdata.csv', initialdata, delimiter=',')
    # npy_write(jetstreamdata,'jetstreamdata')

    df = convert_to_dataframe(initialdata, df_columns = output_columns,printID=True)

    atom_id_filter = np.genfromtxt('initial_atom_id.csv', delimiter=',')

    if sample_size > 0:
        print("filtering by sampling")
        stacked_df_sample_ids = group_and_output_write_samples(df, groupby_cat='radius', columns_out=output_columns, sample_size=sample_size)

    #empty dataframe to store the previous particle's location
    previous_df = pd.DataFrame() 

    for filepath, filename in zip(filepath_list, filename_list):
        print("filepath: ", filepath)
        alldata = read_input_file(filepath)
        df = convert_to_dataframe(alldata, df_columns=output_columns, printID=False)
        
        file_number = determine_filename_number(filename)
        print(file_number)

        if sample_size > 0:
            print("filtering by sampling")
            atom_id_filter = stacked_df_sample_ids['atom_id'].tolist()
            print("atom_id_filter: ", atom_id_filter)

        if particle_position_placeholder:
            df = filter_by_atom_id_with_placeholder(df, atom_id_filter, previous_df)
        else: 
            df = filter_by_atom_id(df, atom_id_filter)
            
        group_and_output(df, groupby_cat='radius', columns_out=output_columns, filename=file_number)
        
        previous_df = df.copy() # Save the current DataFrame for the next iteration


if __name__ == "__main__":
    # stdoutOrigin=sys.stdout 
    # sys.stdout = open("logIt.txt", "w")
    st = time.time()
    main()
    elapsed_wall_time = time.time() - st
    print('Execution wall time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_wall_time)))
    # sys.stdout.close()
    # sys.stdout=stdoutOrigin
