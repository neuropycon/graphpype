"""
Support function for loading test datasets
"""
import os
import os.path as op


def load_test_data(name):
    """load test data"""
    data_dir = {"data_nii": "STtqD3Y4ZZYqpTm",
                "data_con": "kEXrxPs4DgAJaLs",
                "data_gather": "5SQpidYQJXqij6F",
                "data_con_meg": "yAbqiq5WqAzSRkC",
                "data_inv_ts": "HT85yik8iR5W5rF"}

    data_dirpath = op.join(op.split(__file__)[0], "..", "data")

    try:
        os.makedirs(data_dirpath)
    except OSError:
        print("data_dirpath {} already exists".format(data_dirpath))

    data_path = op.join(data_dirpath, name)
    data_zip = op.join(data_dirpath, "{}.zip".format(name))

    if op.exists(data_path):
        return data_path

    if not op.exists(data_zip):

        assert name in data_dir.keys(),\
            "Error, {} not found in data_dict".format(name)
        oc_path = "https://cloud.int.univ-amu.fr/index.php/s/{}/download"\
            .format(data_dir[name])
        os.system("wget -O {} --no-check-certificate  --content-disposition\
            {}".format(data_zip, oc_path))

    assert op.exists(data_zip),\
        "Error, data_zip = {} not found ".format(data_zip)

    os.system("unzip -o {} -d {}".format(data_zip, data_dirpath))
    os.remove(data_zip)

    data_path = op.join(data_dirpath, name)

    assert op.exists(data_path)

    return data_path
