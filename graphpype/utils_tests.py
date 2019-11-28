"""
Support function for loading test datasets
"""
import os
import os.path as op


def load_test_data(name):
    """load test data"""

    data_dir = {"data_nii": "RCjt2KioMBmHRxj",
                "data_nii_mask": "BsjCfxSrXXGEaRD",
                "data_con": "8mofHzzWAiTZCw8",
                "data_nii_HCP": "sEPCRnQbs3RFFmw",
                "data_nii_min": "ypWHfE2kocQ7zq6",
                # "data_gather": "AfYgQZ87zTGJDx9",
                "data_con_meg": "Z4iebjw8maLt3Sd",
                "data_inv_ts": "KWTNYXesE2isZYa"}
    """
    data_dir = {"data_nii": "8HP8YRACDK7nPH7",
                "data_nii_mask": "pnzeHjz3xGY7Zij",
                "data_con": "kEXrxPs4DgAJaLs",
                "data_gather": "5SQpidYQJXqij6F",
                "data_con_meg": "yAbqiq5WqAzSRkC",
                "data_inv_ts": "HT85yik8iR5W5rF"}
    """
    data_dirpath = op.join(op.split(__file__)[0], "..", "data")

    try:
        os.makedirs(data_dirpath)
    except OSError:
        pass

    data_path = op.join(data_dirpath, name)
    data_zip = op.join(data_dirpath, "{}.zip".format(name))

    if op.exists(data_path):
        return data_path

    if not op.exists(data_zip):

        assert name in data_dir.keys(),\
            "Error, {} not found in data_dict".format(name)
        oc_path = "https://amubox.univ-amu.fr/s/{}/download"\
            .format(data_dir[name])
        # oc_path = "https://cloud.int.univ-amu.fr/index.php/s/{}/download"\
        #    .format(data_dir[name])
        os.system("wget -O {} --no-check-certificate  --content-disposition\
            {}".format(data_zip, oc_path))

    assert op.exists(data_zip),\
        "Error, data_zip = {} not found ".format(data_zip)

    os.system("unzip -o {} -d {}".format(data_zip, data_dirpath))
    os.remove(data_zip)

    data_path = op.join(data_dirpath, name)

    assert op.exists(data_path)

    return data_path
