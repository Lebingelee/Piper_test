import h5py
import argparse
import numpy as np


def _matches_keywords(name, keywords):
    if not keywords:
        return True
    name_lower = name.lower()
    return any(kw in name_lower for kw in keywords)


def _read_dataset(obj):
    value = obj[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return np.asarray(value)


def build_h5_dataset_dict(file_path, keywords=None):
    """
    将 H5 中匹配关键词的数据集读取到普通 dict。

    返回的 dataset 是 materialized numpy/string 值，不依赖打开的 h5py.File，
    适合在后续代码行打断点后用 dataset.keys() / dataset["xxx"] 调试查看。
    """
    keywords = [kw.lower() for kw in (keywords or [])]
    dataset = {}
    dataset_attrs = {}
    file_attrs = {}

    def collect_if_match(name, obj):
        if isinstance(obj, h5py.Dataset):
            if not _matches_keywords(name, keywords):
                return
            dataset[name] = _read_dataset(obj)
            dataset_attrs[name] = dict(obj.attrs)
            print(f"Key: {name:<50} | Shape: {str(obj.shape):<25} | Dtype: {obj.dtype}")

    with h5py.File(file_path, 'r') as f:
        print(f"Inspecting HDF5 file: {file_path}")
        print(f"Keywords: {keywords or ['<all>']}")
        print("=" * 90)
        print(f"{'Key':<50} | {'Shape':<25} | Dtype")
        print("-" * 90)
        f.visititems(collect_if_match)
        file_attrs = dict(f.attrs)

    return dataset, dataset_attrs, file_attrs


def inspect_h5_keywords(file_path, keywords):
    return build_h5_dataset_dict(file_path, keywords)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect shapes of datasets in an .h5 file whose keys contain given keywords."
    )
    parser.add_argument("-p","--file_path", type=str, 
                        default='data/Rollout_cpiql/group_1/traj_3_0514_2348.h5',
                        help="Path to the .h5 file")
    parser.add_argument(
        "-k", "--keywords",
        nargs='+',
        default=None,
        help="One or more keywords to filter keys. Omit this to load all datasets."
    )
    args = parser.parse_args()

    dataset, dataset_attrs, file_attrs = build_h5_dataset_dict(
        args.file_path,
        args.keywords,
    )

    # 在下一行打断点后，可在调试控制台查看：
    #   dataset.keys()
    #   dataset["traj_0000/action"]
    #   dataset["traj_0000/action"][0]
    #   dataset_attrs["traj_0000/action"]
    #   file_attrs
    print(dataset['traj_0000/success'][-1])
    print(f"\nLoaded {len(dataset)} datasets into variable `dataset`.")
