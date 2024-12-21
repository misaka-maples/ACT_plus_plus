import h5py

def print_hdf5_structure(hdf5_file, indent=0):
    for key in hdf5_file.keys():
        item = hdf5_file[key]
        print('  ' * indent + key)
        if isinstance(item, h5py.Group):
            print_hdf5_structure(item, indent + 1)
        elif isinstance(item, h5py.Dataset):
            print('  ' * (indent + 1) + f"Dataset: {item.shape}, {item.dtype}")

def main():
    # hdf5_path = r'D:\人头马\Mobile ALOHA\数据集\public_mobile_aloha_datasets-20240531T002240Z-001\public_mobile_aloha_datasets\aloha_static_cotraining_datasets\aloha_coffee_new_compressed\episode_3.hdf5'  # 替换为您的 HDF5 文件路径
    # hdf5_path= r'E:\episode_44.hdf5'
    # hdf5_path  = r"F:\isaac_hdf5_save\episode_37.hdf5"
    hdf5_path= "D:\BYD\git_ku\ACT_plus_plus-master\ACT_plus_plus-master\hdf5_file\save_dir\episode_23.hdf5"
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        print(f"Structure of {hdf5_path}:")
        print_hdf5_structure(hdf5_file)

if __name__ == "__main__":
    main()