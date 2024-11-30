import h5py

def get_all_data_from_hdf5(file_path):
    """
    This function loads all data from an HDF5 file.
    It recursively traverses the entire file structure and retrieves the data.

    Parameters:
        file_path (str): Path to the HDF5 file.

    Returns:
        dict: A dictionary with the HDF5 file structure and data.
    """
    def recursive_extract(group):
        """ Recursively extract all datasets from an HDF5 group. """
        data_dict = {}
        for key in group:
            item = group[key]
            if isinstance(item, h5py.Group):  # If it's a group, recurse
                data_dict[key] = recursive_extract(item)
            elif isinstance(item, h5py.Dataset):  # If it's a dataset, get the data
                data_dict[key] = item[()]
        return data_dict

    # Open the HDF5 file and recursively extract data
    with h5py.File(file_path, 'r') as f:
        return recursive_extract(f)

# Example usage
file_path = '/home/zhnh/Documents/project/aloha/act-plus-plus/results/episode_0.hdf5'
data = get_all_data_from_hdf5(file_path)

# Print the structure and data (just a sample)
print(data)
