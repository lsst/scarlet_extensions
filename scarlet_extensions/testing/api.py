import errno
from io import BytesIO
import json
import os
import shutil
import sqlite3
from typing import List, Callable
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from . import settings
from . import aws


# Paths to directories for different file types
__ROOT__ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

__LOOKUP_PATH__ = __ROOT__

__DATA_PATH__ = os.path.join(__ROOT__, "data")
__BLEND_PATH__ = os.path.join(__DATA_PATH__, "blends")
__DOCS_PATH__ = os.path.join(__ROOT__, "docs")
__PLOT_PATH__ = os.path.join(__DOCS_PATH__, "plots")
__SCENE_PATH__ = os.path.join(__PLOT_PATH__)
__BRANCH_FILE__ = os.path.join(__ROOT__, "branches.json")


def get_blend_ids(set_id: int=None) -> List[str]:
    """Get all of the blend IDs contained in the set

    Either `path` or `set_id` must be given.

    :param path: Path to blends
    :param set_id: Set containing blend ids
    :return: List of blend IDs
    """
    assert set_id in [1,2,3]
    sql = "SELECT blend_id FROM blends"
    if set_id is not None:
        sql += " WHERE set_id={};".format(set_id)

    path = os.path.join(__ROOT__, "lookup.db")
    connect = sqlite3.connect(path)
    cursor = connect.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()
    connect.commit()
    connect.close()

    blend_ids = np.array(result)[:, 0]
    return blend_ids


def get_blend(blend_id: str, path:str=None):
    """Load a blend

    :param blend_id: ID of the blend
    :param path: Path to the blend. If `path` is `None`
        then the blend is loaded from AWS
    :return:
    """
    if blend_id[-4:] != ".npz":
        blend_id = blend_id + ".npz"
    if path is None:
        client = aws.get_client("s3")
        result = client.get_object(Bucket="scarlet-blends", Key=blend_id)
        data = np.load(BytesIO(result["Body"].read()), allow_pickle=True)
    else:
        data = np.load(os.path.join(path, blend_id))
    return data


def get_branches() -> List[str]:
    """Load all of the branches that have been processed

    :return: List of the branches in the order that they were merged
    """
    # Make sure that the set id is valid
    f = open(__BRANCH_FILE__, "r")
    branches = json.load(f)
    f.close()
    return branches["branches"]


def save_branch(branch: str) -> None:
    """Append a new branch to the branches list

    :param branch: The branch to add to the list
    """
    branches = get_branches()
    if branch not in branches:
        branches.append(branch)
        f = open(__BRANCH_FILE__, "w")
        json.dump({"branches": branches}, f)
        f.close()


def check_data_existence(set_id: str, branch: str, overwrite: bool) -> bool:
    """Check if the pr has already been processed and if it's ok to overwrite
    :param set_id: ID of the set to analyze
    :param branch: The scarlet branch that wants to be created
    :param overwrite: Whether or not it is ok to rewrite the existing branch data
    :return: `True` if the pr can be written
    """
    branches = get_branches()
    if branch in branches and not overwrite:
        msg = "Branch {} has already been analyzed for set {}, to overwrite set the `overwrite` flag"
        raise ValueError(msg.format(branch, set_id))
    return True


def create_path(path: str) -> None:
    """
    Create a path if it doesn't already exist
    :param path: The name of the path to check/create
    """
    try:
        os.makedirs(path)
        print("created path '{}'".format(path))
    except OSError as e:
        print("results will be written to {}".format(path))
        if e.errno != errno.EEXIST:
            raise e


def get_filename(pr: str) -> str:
    """Consistently set the filename for each PR, for each set

    :param pr: The scarlet PR number
    :return: The filename to store the measurements for each source
    """
    return "{}.npz".format(pr)


def deblend_and_measure(
        set_id: str = None,
        branch: str = None,
        overwrite: bool = False,
        data_path: str = None,
        save_records: bool = False,
        save_residuals: bool = False,
        plot_residuals: bool = False,
        deblender: Callable = None,
) -> np.rec.recarray:
    """Deblend an entire test set and store the measurements

    :param set_id: ID of the set to analyze.
        This is only needed if `data_path` is `None` and
        `save` is `True`.
    :param branch: The scarlet branch to test
        (only needed if `save_records` or `save_residuals` is `True`)
    :param overwrite: Whether or not it is ok to rewrite the existing branch
    :param data_path: The path to the blend data. If no `data_path is specified
        then __BLEND_PATH__ is used.
    :param save_records: Whether or not to save the measurements records.
    :param save_residuals: Whether or not to save the residual plots
        (only necessary when `plot_residuals=True`).
    :param plot_residuals: Whether or not to plot the residuals.
    :param deblender: The function to use to deblend. This function should only take
        1 argument:
        
        * `data` The data from the npz file for the blend.
        
        The function should return a tuple with the following three items:
        
        * `measurements`: The measurement dictionary entry for the blend
        * `observation`: The observation used for deblending.
        * `sources`: The deblended source models.

    :return: The measurement `records` for each blend.
    """
    if data_path is None:
        data_path = os.path.join(__BLEND_PATH__, set_id)
        blend_ids = get_blend_ids(set_id=set_id)
    else:
        blend_ids = get_blend_ids(path=data_path)
    if save_records:
        check_data_existence(set_id, branch, overwrite)
    # Use the default `scarlet_extensions` `deblend` if the user hasn't specified their own
    if deblender is None:
        # import here to avoid circular dependence
        from . import deblend

        deblender = partial(
            deblend.deblend,
            max_iter=settings.max_iter,
            e_rel=settings.e_rel,
        )

    # Deblend the scene
    all_measurements = []

    num_blends = len(blend_ids)
    for bidx, blend_id in enumerate(blend_ids):
        print("blend {} of {}: {}".format(bidx, num_blends, blend_id))
        print(blend_id)
        filename = os.path.join(data_path, "{}.npz".format(blend_id))
        data = np.load(filename)
        results = deblender(data)
        measurements, observation, sources = results
        all_measurements += measurements

        if plot_residuals:
            import scarlet.display as display
            images = observation.images
            norm = display.AsinhMapping(minimum=np.min(images), stretch=np.max(images) * 0.055, Q=10)
            display.show_scene(sources, observation, show_model=False, show_observed=True, show_rendered=True,
                               show_residual=True, norm=norm, figsize=(15, 5))
            plt.suptitle(branch, y=1.05)

            if save_residuals:
                filename = os.path.join(__SCENE_PATH__, "{}.scene.png".format(blend_id))
                if os.path.exists(filename):
                    # Copy the current version as the old filename
                    new_filename = os.path.join(__SCENE_PATH__, "old_{}.scene.png".format(blend_id))
                    shutil.move(filename, new_filename)
                plt.savefig(filename)
                plt.close()
            else:
                plt.show()

    # Combine all of the records together and save
    _records = [tuple(m.values()) for m in all_measurements]
    keys = tuple(all_measurements[0].keys())
    records = np.rec.fromrecords(_records, names=keys)
    # Save the data if a path was provided
    if save_records:
        np.savez(os.path.join(__DATA_PATH__, set_id, get_filename(branch)), records=records)
        save_branch(branch)
    return records
