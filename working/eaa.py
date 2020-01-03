import os
import sys
import multiprocessing as mp
import string
import platform
import shutil
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
import calendar
import pyemu
import flopy


# some global config for plotting
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
abet = string.ascii_uppercase

# some global config for path/directory structure
old_h_dir = os.path.join("..", "ver")
h_nam_file = "eaa_ver.nam"
h_dir = "history"
h_start_datetime = "1-1-2001"
h_end_datetime = "12-31-2015"

old_s_dir = os.path.join("..", "pred")
s_dir = "scenario"
s_nam_file = "eaa_pred.nam"

# history and scenarion simulation start datetimes
s_start_datetime = "1-1-1947"
s_end_datetime = "12-31-1958"

# files with history and scenario observation locations and states
h_hds_file = os.path.join("_data", "reformatted_head_obs.smp")
h_drn_file = os.path.join("_data", "springflow_obs.smp")
h_crd_file = os.path.join("_data", "head_obs.crd")
s_hds_file = os.path.join("_data", "pred_head_obs.smp")
s_drn_file = os.path.join("_data", "pred_springflow_obs.smp")
s_crd_file = os.path.join("_data", "pred_head_obs.crd")

# value of dry cells
hdry = -1.0e+20

# platform-specific binary information
exe_name = "mf2005"
ies_name = "pestpp-ies"

if "window" in platform.platform().lower():
    bin_path = os.path.join("bin", "win")
    exe_name = exe_name + ".exe"
    ies_name = ies_name + ".exe"
elif "darwin" in platform.platform().lower():
    bin_path = os.path.join("bin", "mac")
else:
    bin_path = os.path.join("bin", "linux")

# the numeric IDs of J-17 and J-27
j17_id = 6837203
j27_id = 6950302


def _setup_model(old_dir, new_dir, start_datetime, nam_file, run=False):
    """load an existing model (either history or scenario) and configure it for
    PEST interface construction

    Args:
        old_dir (str): directory location where the original model resides
        new_dir (str): directory location where the new model files will be written
        start_datetime (str): string rep of model starting datetime
        nam_file (str): MODFLOW-2005 nam file
        run (bool): flag to run the model once it is written to new_dir. Default is False

    """

    # load the existing model and set some attributes
    m = flopy.modflow.Modflow.load(nam_file, model_ws=old_dir, check=False, 
                                   verbose=True, forgive=False)
    m.start_datetime = start_datetime
    m.lpf.hdry = hdry
    m.bas6.hnoflo = hdry

    # change the workspace to new_dir
    m.change_model_ws(new_dir, reset_external=True)
    # set the external path so that arrays and lists are outside of the
    # terrible MODFLOW file formats
    m.external_path = "."
    # write the inputs
    m.write_input()
    # run?
    if run:
        shutil.copy2(os.path.join(bin_path, exe_name), os.path.join(new_dir, exe_name))
        pyemu.os_utils.run("{0} {1}".format(exe_name, nam_file), cwd=new_dir)


def _rectify_wel(model_ws, nam_file, run=True):
    """rectify the stress period WEL file entries so that every
    stress period has the same entries (filling missing wells with
    "dummy" entries with zero pumping)

    Args:
        model_ws (str): model workspace
        nam_file (str): MODFLOW-2005 nam file
        run (bool): flag to run model once the WEL file has been rectified.
            Default is True.

    """
    # load the model
    m = flopy.modflow.Modflow.load(nam_file, model_ws=model_ws, check=False, 
                                   verbose=True, forgive=False)
    # get the current WEL file datasets
    spd = m.wel.stress_period_data
    df_dict = {}
    all_kij = set()
    # run thru all stress periods to get the union of well locations
    for kper in range(m.nper):
        ra = spd[kper]
        df = pd.DataFrame.from_records(ra)
        df.loc[:, "kij"] = df.apply(lambda x: (x.k, x.i, x.j), axis=1)
        df.loc[:, "kij_str"] = df.kij.apply(lambda x: "{0:01.0f}_{1:03.0f}_{2:03.0f}".format(*x))
        df.index = df.kij_str
        all_kij.update(set(df.kij_str.tolist()))
        print(kper)
        df_dict[kper] = df

    # work up fast-lookup containers for well location indices
    new_index = list(all_kij)
    new_k = {s: int(s.split('_')[0]) for s in new_index}
    new_i = {s: int(s.split('_')[1]) for s in new_index}
    new_j = {s: int(s.split('_')[2]) for s in new_index}
    new_index.sort()

    # process each stress period
    new_spd = {}
    for kper, df in df_dict.items():
        # reindex with the full kij locations index
        df = df.reindex(new_index)
        # map the new kijs to the old kijs
        for f, d in zip(["k", "i", "j"], [new_k, new_i, new_j]):
            isna = df.loc[:, f].isna()
            df.loc[isna, f] = [d[kij] for kij in df.loc[isna, :].index.values]

        # fill the nans with 0.0
        isna = df.flux.isna()
        df.loc[isna, "flux"] = 0.0

        # deal with the platform numpy int casting issue
        if "window" in platform.platform():
            df.loc[:, "i"] = df.i.astype(np.int32)
            df.loc[:, "j"] = df.j.astype(np.int32)
            df.loc[:, "k"] = df.k.astype(np.int32)
        else:
            df.loc[:, "i"] = df.i.astype(np.int)
            df.loc[:, "j"] = df.j.astype(np.int)
            df.loc[:, "k"] = df.k.astype(np.int)

        spd[kper] = df.loc[:, ["k", "i", "j", "flux"]].to_records(index=False)
    # create a new WEL package and replace the old one
    flopy.modflow.ModflowWel(m, stress_period_data=spd, ipakcb=m.wel.ipakcb)
    # write to a new model_ws with a "_wel" suffix
    m.change_model_ws("{0}_wel".format(model_ws))
    m.external_path = '.'
    m.write_input()
    # run?
    if run:
        shutil.copy2(os.path.join(bin_path, exe_name), os.path.join("{0}_wel".format(model_ws), exe_name))
        pyemu.os_utils.run("{0} {1}".format(exe_name, nam_file), cwd="{0}_wel".format(model_ws))
        # just to make sure the model ran
        new_lst = flopy.utils.MfListBudget(os.path.join("{0}_wel".format(model_ws), nam_file.replace(".nam", ".list")))


def build_rch_zone_array(model_ws, nam_file, plot=False):
    """build a recharge zone integer array for zone-based parameters
    using unique values in the in recharge arrays

    Args:
        model_ws (str): model workspace
        nam_file (str): MODFLOW-2005 nam file
        plot (bool): flag to plot the zone array.  Default is False
    """
    m = flopy.modflow.Modflow.load(nam_file, model_ws=model_ws, load_only=["rch"], check=False, 
                                   verbose=True, forvive=False)
    arr = m.rch.rech[0].array
    full_arr = m.rch.rech.array
    mn = full_arr.mean(axis=0)[0, :, :]
    mn_u, mn_c = np.unique(mn, return_counts=True)
    zn_arr = np.zeros_like(arr, dtype=np.int)
    for i, u_val in enumerate(mn_u):
        # this contional makes sure we keep zeros as zero in the zone array
        if u_val == 0.0:
            continue
        zn_arr[mn == u_val] = i
    np.savetxt(os.path.join("_data", "rch_zn_arr.dat"), zn_arr, fmt="%3d")
    if plot:
        zn_arr = zn_arr.astype(np.float)
        zn_arr[zn_arr == 0] = np.NaN
        cb = plt.imshow(zn_arr)
        plt.colorbar(cb)
        plt.show()


def _setup_pst(org_model_ws, new_model_ws, nam_file):
    """construct the PEST interface, set parameter bounds and
    generate the prior ensemble

    Args:
        org_model_ws (str): original model workspace
        new_model_ws (str): new model workspace/directory where the
            PEST interface will be constructed
        nam_file (str): MODFLOW-2005 nam file
    """

    # make sure the model simulated heads file exists - need this for observations
    if not os.path.exists(os.path.join(org_model_ws, nam_file.replace(".nam", ".hds"))):
        raise Exception("need to call _setup_model()")

    # load the model from org_model_ws
    m= flopy.modflow.Modflow.load(nam_file, model_ws=org_model_ws,
                                  load_only=["dis"], check=False,
                                  verbose=True, forgive=False)
    # load the recharge zone array
    rch_zn_arr = np.loadtxt(os.path.join("_data", "rch_zn_arr.dat"), dtype=np.int)

    # array-based model inputs to parameterize by layer (zero-based)
    props = [["lpf.hk", 0], ["lpf.ss", 0], ["lpf.sy", 0], ["bas6.strt", 0]]

    # copy to constant (global props)
    const_props = props.copy()

    # fill a zone-based array inputs container with recharge
    # zone pars for each stress period
    zone_props = []
    zone_props.extend([["rch.rech", kper] for kper in range(m.nper)])
    # extend the global parameter container with recharge for each stress period
    const_props.extend([["rch.rech", kper] for kper in range(m.nper)])
    # include the final simulated groundwater level in every active
    # model cell as an "observation" in PEST interface
    hds_kperk = [[m.nper - 1, 0]]

    # parameterize WEL flux and DRN cond spatially (one par for each entry)
    spatial_bc_props = [["wel.flux", 0], ["drn.cond", 0]]
    # parameterize WEL flux with a single global multiplier for ecah stress period
    temporal_bc_props = [["wel.flux", kper] for kper in range(m.nper)]

    #create the pest interface...
    ph = pyemu.helpers.PstFromFlopyModel(nam_file, org_model_ws=org_model_ws, new_model_ws=new_model_ws,
                                         grid_props=props,
                                         hds_kperk=hds_kperk, zone_props=zone_props, hfb_pars=True,
                                         remove_existing=True, build_prior=False, k_zone_dict={0: rch_zn_arr},
                                         spatial_bc_props=spatial_bc_props, temporal_bc_props=temporal_bc_props,
                                         model_exe_name=exe_name, pp_props=props, pp_space=30, const_props=const_props)


    # set the parameter bounds to Edwards-based physically-plausible values
    _set_par_bounds(ph.pst, nam_file)

    # geostatistcal draws from the prior
    pe = ph.draw(num_reals=300, use_specsim=True)
    #add the control file initial values as a realization
    pe.add_base()
    # enforce parameter bounds on the ensemble
    pe.enforce()
    # save the ensemble to compressed (PEST extended binary) format
    pe.to_binary(os.path.join(new_model_ws, "prior.jcb"))
    # save the control file
    ph.pst.write(os.path.join(new_model_ws, nam_file.replace(".nam", ".pst")))
    # read the array parameter multiplier config file and set a hard upper bound
    # on specific yield
    df = pd.read_csv(os.path.join(new_model_ws, "arr_pars.csv"))
    df.loc[:, "upper_bound"] = np.NaN
    df.loc[:, "lower_bound"] = np.NaN
    df.loc[df.org_file.apply(lambda x: "sy_" in x), "upper_bound"] = 0.25
    df.to_csv(os.path.join(new_model_ws, "arr_pars.csv"))

    # put the MODFLOW-2005 and PESTPP-IES binaries in the new_model_ws
    shutil.copy2(os.path.join(bin_path, exe_name), os.path.join(new_model_ws, exe_name))
    shutil.copy2(os.path.join(bin_path, ies_name), os.path.join(new_model_ws, ies_name))


def _set_par_bounds(pst, nam_file):
    """set the parameter bounds to expert-knowledge-based
    ranges

    Args:
        pst (pyemu.Pst): PEST control file instance
        nam_file (str): MODFLOW-2005 nam file

    """

    par = pst.parameter_data

    # special case for WEL flux pars: more recent time has metering, so less uncertainty
    names = par.loc[par.pargp.apply(lambda x: "welflux" in x), "parnme"]
    if nam_file == h_nam_file:
        par.loc[names, "parlbnd"] = 0.9
        par.loc[names, "parubnd"] = 1.1
    else:
        par.loc[names, "parlbnd"] = 0.7
        par.loc[names, "parubnd"] = 1.3

    # DRN conductance
    names = par.loc[par.pargp.apply(lambda x: "drncond" in x), "parnme"]
    par.loc[names, "parlbnd"] = 0.5
    par.loc[names, "parubnd"] = 1.5

    # initial conditions
    names = par.loc[par.pargp.apply(lambda x: "strt" in x), "parnme"]
    par.loc[names, "parlbnd"] = 0.9
    par.loc[names, "parubnd"] = 1.1

    # recharge
    names = par.loc[par.pargp.apply(lambda x: "rech" in x), "parnme"]
    par.loc[names, "parlbnd"] = 0.8
    par.loc[names, "parubnd"] = 1.2

    # HK
    names = par.loc[par.pargp.apply(lambda x: "hk" in x), "parnme"]
    par.loc[names, "parlbnd"] = 0.01
    par.loc[names, "parubnd"] = 100


def _add_smp_obs_to_pst(org_model_ws, new_model_ws, pst_name, nam_file, hds_crd_file):
    """add observations to the control file for the locations where groundwater levels
    have been measured.  The actual value of the observations will be set elsewhere

    Args:
        org_model_ws (str): original model workspace
        new_model_ws (str): new model workspace
        pst_name (str): PEST control file name
        nam_file (str): MODFLOW-2005 nam file
        hds_crd_file (str): PEST-style coordinate file that has been processed
            to include k,i,j indices

    """

    # make sure the control file exists
    pst_name = os.path.join(new_model_ws, pst_name)
    assert os.path.exists(pst_name)

    # load the model
    m = flopy.modflow.Modflow.load(nam_file, model_ws=new_model_ws,
                                   load_only=["dis"], check=False,
                                   forgive=False)

    # load the control file
    pst = pyemu.Pst(pst_name)

    # load GW level location dataframe
    crd_df = pd.read_csv(hds_crd_file + ".csv")

    #load DRN location dataframe
    drn_df = pd.read_csv(os.path.join("_data", "DRN_dict.csv"), delim_whitespace=True,
                         header=None, names=["name", "k", "i", "j"])

    # build a dict of name-index location for DRN locations
    kij_dict = {n: [0, i, j] for n, i, j in zip(drn_df.name, drn_df.i, drn_df.j)}

    # the name of the DRN budget file
    cbd_file = nam_file.replace(".nam", ".cbd")

    # get one from the org model workspace and update the path to it
    shutil.copy2(os.path.join(org_model_ws, cbd_file), os.path.join(new_model_ws, cbd_file))
    cbd_file = os.path.join(new_model_ws, cbd_file)

    # setup the forward run DRN budget post processor
    prec = "double"
    if "win" not in platform.platform().lower(): # not win or darwin
        prec = "singl"
    cbd_frun, cbd_df = pyemu.gw_utils.setup_hds_timeseries(cbd_file, kij_dict, prefix="drn",
                                                           include_path=True, fill=-1.0e+30,
                                                           text="drains", precision=prec,
                                                           model=m)

    # make sure the new DRN instruction file exists
    ins_file = "{0}_timeseries.processed.ins".format(cbd_file)
    assert os.path.exists(ins_file), ins_file

    # add the new DRN observations to the control file
    pst.add_observations(ins_file=ins_file, pst_path=".")

    # set meaningful obs group names
    pst.observation_data.loc[cbd_df.index, "obgnme"] = cbd_df.obgnme

    # build a dict of name-index locations for the GW level observations locations
    kij_dict = {n: [0, i, j] for n, i, j in zip(crd_df.name, crd_df.i, crd_df.j)}

    # setup GW level post processor
    hds_file = os.path.join(new_model_ws, nam_file.replace(".nam", ".hds"))
    assert os.path.exists(hds_file)
    hds_frun, hds_df = pyemu.gw_utils.setup_hds_timeseries(hds_file, kij_dict, prefix="hds",
                                                           include_path=True, fill=-1.0e+30, model=m)

    # make sure the GW level instruction file exists
    ins_file = "{0}_timeseries.processed.ins".format(hds_file)
    assert os.path.exists(ins_file), ins_file
    # add the GW level obs to the control file and set meaningful
    # obs group names
    pst.add_observations(ins_file=ins_file, pst_path=".")
    pst.observation_data.loc[hds_df.index, "obgnme"] = hds_df.obgnme
    # write the updated control file
    pst.write(pst_name)

    # add the post processor commands to the forward run script
    frun_file = os.path.join(new_model_ws, "forward_run.py")
    with open(frun_file, 'r') as f:
        lines = f.readlines()
    idx = None
    for i, line in enumerate(lines):
        if "__name__" in line:
            idx = i
    assert idx is not None
    lines.insert(idx, "    " + cbd_frun + '\n')
    lines.insert(idx, "    " + hds_frun + '\n')
    with open(frun_file, 'w') as f:
        for line in lines:
            f.write(line)


def add_ij_to_hds_smp(crd_file):
    """intersect the GW level observation coordinates against the
    model grid to get k,i,j index information

    Args:
        crd_file (str): PEST-style "bore coordinates" file


    """
    from shapely.geometry import Point
    # read the bore coord file
    crd_df = pd.read_csv(crd_file, delim_whitespace=True, header=None, names=["name", "x", "y", "layer"])
    # set a shapely point attribute
    crd_df.loc[:, "pt"] = crd_df.apply(lambda x: Point(x.x, x.y), axis=1)
    # load the history model
    m = flopy.modflow.Modflow.load(h_nam_file, model_ws=h_dir, 
                                   load_only=["dis"], check=False,
                                   forgive=False)
    # use the flopy grid intersect functionality
    gi = flopy.utils.GridIntersect(m.modelgrid)
    crd_df.loc[:, 'ij'] = crd_df.pt.apply(lambda x: gi.intersect_point(x)[0][0])
    # split out the i and j indices
    crd_df.loc[:, 'i'] = crd_df.ij.apply(lambda x: x[0])
    crd_df.loc[:, 'j'] = crd_df.ij.apply(lambda x: x[1])

    # remove extra columns
    crd_df.pop("ij")
    crd_df.pop("pt")

    # save the new dataframe to a CSV file
    crd_df.to_csv(crd_file + ".csv")

def _set_obsvals(d, nam_file, hds_file, drn_file, pst_file, run=True):
    """samples the groundwater and spring discharge observations to
    the model stress periods and sets the "obsval" attribute in the control
    file. Also plots up org obs and sampled obs in a multipage pdf

    Args:
        d (str): directory where the control file exists
        nam_file (str): MODFLOW-2005 nam file
        hds_file (str): PEST-style site sample file with groundwater
            level observations
        drn_file (str): PEST-style site sample file with spring discharge
            observations
        pst_file (str): PEST control file
        run (bool): flag to run PESTPP-IES with NOPTMAX=0 after the
            observation values have been updated.  Default is True.

    """

    # load the model
    m = flopy.modflow.Modflow.load(nam_file, model_ws=d, load_only=["dis"], 
                                   check=False, forgive=False)

    # work out the stress period ending datetime
    sp_end_dts = pd.to_datetime(m.start_datetime) + pd.to_timedelta(np.cumsum(m.dis.perlen.array), unit='d')

    # cast the model start_datetime from a str to a datetime instance
    start_datetime = pd.to_datetime(m.start_datetime)

    # load the gw level and spring discharge site sample files
    # into pandas dataframes
    hds_df = pyemu.smp_utils.smp_to_dataframe(hds_file)
    drn_df = pyemu.smp_utils.smp_to_dataframe(drn_file)

    # plotting limits
    xmn, xmx = pd.to_datetime(start_datetime), pd.to_datetime(sp_end_dts[-1])
    ymn, ymx = hds_df.value.min(), hds_df.value.max()

    # containers for the sampled observation series
    hds_sampled_dfs = []
    drn_sampled_dfs = []

    # a function to sample each observation in a given site
    # dataframe to the model stress period ending datetimes
    # uses nearest neighbor
    def sample_to_model(udf):
        d, v = [], []
        for dt, val in zip(udf.index.values, udf.value.values):
            # difference between this obs datetime and the
            # stress period end datetime
            diff = (sp_end_dts - dt).map(np.abs).values
            # the index of the minimum diff (nearest neighbor)
            idxmin = np.argmin(diff)

            # minimum diff in days
            day_diff = diff[idxmin].astype('timedelta64[D]')

            # the diff is greater than a month, something is wrong...
            if day_diff > np.timedelta64(31, 'D'):
                print(idxmin, sp_end_dts[idxmin], dt, day_diff)
                continue
            # save the datetime and value
            d.append(sp_end_dts[idxmin])
            v.append(val)
        # form a new dataframe and return
        udf_mod = pd.DataFrame({"value": v}, index=d)
        return udf_mod

    # save a multipage PDF for inspection
    with PdfPages(os.path.join("_data", "obs.pdf")) as pdf:
        ax_per_page = 10
        fig, axes = plt.subplots(ax_per_page, 1, figsize=(8.5, 11))
        ax_count = 0
        # process each unique GW level site entry
        for usite in hds_df.name.unique():
            print(usite)
            # get a dataframe of just this site
            udf = hds_df.loc[hds_df.name == usite, ["datetime", "value"]].copy()
            # set the index to datetime
            udf.index = udf.pop("datetime")
            # sample to stress period ending datetimes
            udf_mod = sample_to_model(udf)
            #set a name attribute
            udf_mod.loc[:, "name"] = usite
            # store new sample site dataframe
            hds_sampled_dfs.append(udf_mod)
            # plot
            ax = axes[ax_count]
            ax.plot(udf.index, udf.value, lw=0.5, marker='.', color='0.5', ms=5, alpha=0.5)
            ax.plot(udf_mod.index, udf_mod.value, lw=0.5, marker='.', color='b', ms=5, alpha=0.5)
            ax.set_title("site:{0}, org count:{1}, reindexed count:{2}".format(usite, udf.shape[0], udf_mod.shape[0]),
                         loc="left")
            ax.set_xlim(xmn, xmx)
            # ax.set_ylim(ymn,ymx)
            ax_count += 1
            if ax_count >= ax_per_page:
                plt.tight_layout()
                pdf.savefig()
                plt.close(fig)
                fig, axes = plt.subplots(ax_per_page, 1, figsize=(8.5, 11))
                ax_count = 0


        #process each unqiue DRN site entry
        for usite in drn_df.name.unique():
            print(usite)
            # get a dataframe of just this site
            udf = drn_df.loc[drn_df.name == usite, ["datetime", "value"]].copy()
            # use the datetime as the index
            udf.index = udf.pop("datetime")
            # sample to stress period ending datetime
            udf_mod = sample_to_model(udf)
            # set a name attribute
            udf_mod.loc[:, "name"] = usite
            # store
            drn_sampled_dfs.append(udf_mod)
            # plot
            ax = axes[ax_count]
            ax.plot(udf.index, udf.value, lw=0.5, marker='.', color='0.5', ms=5, alpha=0.5)
            ax.plot(udf_mod.index, udf_mod.value, lw=0.5, marker='.', color='b', ms=5, alpha=0.5)
            ax.set_title("site:{0}, org count:{1}, reindexed count:{2}".format(usite, udf.shape[0], udf_mod.shape[0]),
                         loc="left")
            ax.set_xlim(xmn, xmx)
            ax_count += 1
            if ax_count >= ax_per_page:
                plt.tight_layout()
                pdf.savefig()
                plt.close(fig)
                fig, axes = plt.subplots(ax_per_page, 1, figsize=(8.5, 11))
                ax_count = 0

        plt.tight_layout()
        pdf.savefig()

    # concatenate the sampled GW level dataframes into one large dataframe
    hds_df = pd.concat(hds_sampled_dfs)
    # set the datetime index as a column
    hds_df.loc[:, "datetime"] = hds_df.index
    # set a generic and nonduplicated index
    hds_df.index = np.arange(hds_df.shape[0])
    # save the sampled dataframe
    pyemu.smp_utils.dataframe_to_smp(hds_df, hds_file.replace(".smp", "_sampled.smp"))

    # concatenate the sample spring discharge dataframes into one large dataframe
    drn_df = pd.concat(drn_sampled_dfs)
    # set the datetime index as a column
    drn_df.loc[:, "datetime"] = drn_df.index
    # set a generic and nonduplicated index
    drn_df.index = np.arange(drn_df.shape[0])
    # save the sampled dataframe
    pyemu.smp_utils.dataframe_to_smp(drn_df, drn_file.replace(".smp", "_sampled.smp"))

    # build up observation names ("obsnme") in the sampled GW level dataframe
    # these are the same names that are in the control file
    hds_df.loc[:, "dt_str"] = hds_df.datetime.apply(lambda x: x.strftime("%Y%m%d"))
    hds_df.loc[:, "site_name"] = hds_df.name
    hds_df.loc[:, "obsnme"] = hds_df.apply(lambda x: "hds_{0}_{1}".format(str(x.site_name), x.dt_str), axis=1)
    hds_df.loc[:, "obsnme"] = hds_df.obsnme.apply(str.lower)
    hds_df.index = hds_df.obsnme

    # load the control file
    pst = pyemu.Pst(os.path.join(d, pst_file))
    obs = pst.observation_data
    # set all observations to zero weight
    obs.loc[:, "weight"] = 0.0
    # get set containers for observation names in the
    # control file and in the GW level dataframe
    pnames = set(list(obs.obsnme.values))
    snames = set(list(hds_df.obsnme.values))
    # make sure all GW level dataframe names are in the
    # control file
    print(snames - pnames)
    assert len((snames - pnames)) == 0
    # set the obsval attribute for space-time locations where
    # we have actual GW level observations
    obs.loc[hds_df.obsnme, "obsval"] = hds_df.value
    # set a generic non-zero weight for the actual
    # GW level observation locations
    obs.loc[hds_df.obsnme, "weight"] = 1.0

    # build up observation names ("obsnme") in the sampled spring discharge dataframe
    # these are the same names that are in the control file
    drn_df.loc[:, "dt_str"] = drn_df.datetime.apply(lambda x: x.strftime("%Y%m%d"))
    drn_df.loc[:, "site_name"] = drn_df.name
    drn_df.loc[:, "obsnme"] = drn_df.apply(lambda x: "drn_{0}_{1}".format(str(x.site_name), x.dt_str), axis=1)
    drn_df.loc[:, "obsnme"] = drn_df.obsnme.apply(str.lower)
    drn_df.index = drn_df.obsnme
    # get set container for observation names in the
    # spring discharge dataframe
    snames = set(list(drn_df.obsnme.values))
    # make sure all spring discharge dataframe names are in the
    # control file
    print(snames - pnames)
    assert len((snames - pnames)) == 0

    # set the obsval attribute for space-time locations where
    # we have actual spring discharge observations
    # negative 1 since drn out is negative, convert from cfs to cfd
    obs.loc[drn_df.obsnme, "obsval"] = -1.0 * drn_df.value * (60. * 60. * 24.)
    # set a generic non-zero weight
    obs.loc[drn_df.obsnme, "weight"] = 1.0
    # set noptmax to 0 for testing
    pst.control_data.noptmax = 0
    # save the updated control file
    pst.write(os.path.join(d, pst_file))
    # run PESTPP-IES?
    if run:
        pyemu.os_utils.run("pestpp-ies {0}".format(pst_file), cwd=d)


def run_local(b_d, m_d, pst_name, num_workers=10):
    """run PESTPP-IES in parallel on the current machine

    Args:
        b_d (str): "base" directory that contains all the files needed
            to run PESTPP-IES (MODFLOW file and PEST interface files)
        m_d (str): "master" directory that will be created and where the
            PESTPP-IES master instance will be started
        pst_name (str): control file name. Must exist in b_d
        num_workers (int): number of parallel workers to start.
            Default is 10.

    """
    pyemu.os_utils.start_workers(b_d, "pestpp-ies", pst_name, num_workers=num_workers,
                                 master_dir=m_d, worker_root=".",
                                 reuse_master=True)


def plot_obs_vs_sim_case(m_d, case="eaa_ver", post_iter=None,
                         plt_name="obs_v_sim.pdf", focus=False):
    """plot ensemble-based observed vs simulated GW level and spring discharge time
    series for a given PEST "case".

    Args:
        m_d (str): "master" directory that holds the simulated output ensembles
        case (str): the PEST "case" name.  Default is "eaa_ver".  various suffixes are
            appended to this case to form control file and ensemble file names
        post_iter (int): the PESTPP-IES iteration to use as the "posterior" ensemble.
            If None, no posterior will be plotted.  If True, only the maximum of the
            prior is plotted (to help with figure "busy-ness").  Default is None.
        plt_name (str): the name of the multi-page PDF to create.  It is written in the
            m_d directory.  Default is :"obs_v_sim.pdf:.
        focus (bool): flag to plot only the four locations of management interest.  If
            True, then only 4 axes are plotted - this creates the figures shown in the
            manuscript.  If False, all locations are plotted - this creates the
            multipage PDFs shown in the supplementary material
    Notes:
        calls plot_obs_vs_sim()

    """


    pst = pyemu.Pst(os.path.join(m_d, case + ".pst"))

    base_en = pyemu.ObservationEnsemble.from_binary(pst=pst,
                                                    filename=os.path.join(m_d, case + ".base.obs.jcb"))
    pr_en = pyemu.ObservationEnsemble.from_binary(pst=pst,
                                                  filename=os.path.join(m_d, case + ".0.obs.jcb"))
    pt_en = None
    if post_iter is not None:
        pt_en = pyemu.ObservationEnsemble.from_binary(pst=pst,
                                                      filename=os.path.join(m_d, "{0}.{1}.obs.jcb". \
                                                                            format(case, post_iter)))
    if "eaa_ver" in case:
        s, e = h_start_datetime, h_end_datetime
    elif "eaa_pred" in case:
        s, e = s_start_datetime, s_end_datetime
    else:
        raise Exception()
    plot_obs_vs_sim(pst=pst, start_datetime=s, end_datetime=e,
                    base_en=base_en, pr_en=pr_en, pt_en=pt_en,
                    plt_name=os.path.join(m_d, plt_name), focus=focus)


def plot_obs_vs_sim(pst, start_datetime, end_datetime, base_en=None, pr_en=None, pt_en=None,
                    plt_name="obs_v_sim.pdf", mask_invalid=True, focus=False):
    """plot ensemble-based observed vs simulated

    Args:
        pst (pyemu.Pst): control file instance
        start_datetime (str): model start datetime string
        end_datetime (str): model end datetime string
        base_en (pyemu.ObservationEnsemble): the observed plus noise ensemble.
            Default is None (dont plot)
        pr_en (pyemu.ObservationEnsemble): prior simulated output ensemble.
            Default is None (dont plot)
        pt_en: (pyemu.ObservationEnsemble): posterior simulated output ensmeble.
            Default is None (dont plot)
        plt_name (str): name of plot to generate.  Default is "obs_v_sim.pdf"
        mask_invalid (bool): flag to mask invalid values in the simulated output
            ensembles (defined by hdry).  Default is True.
        focus (bool): flag to plot only the four locations of management interest.  If
            True, then only 4 axes are plotted - this creates the figures shown in the
            manuscript.  If False, all locations are plotted - this creates the
            multipage PDFs shown in the supplementary material


    """
    # get the non-zero observation data
    obs = pst.observation_data
    nz_obs = obs.loc[pst.nnz_obs_names, :].copy()
    # set the datetimes for each non-zero observation
    nz_obs.loc[:, "datetime"] = pd.to_datetime(nz_obs.obsnme.apply(lambda x: x.split('_')[-1]))
    # spring discharge obs names
    drn_names = nz_obs.loc[nz_obs.obsnme.apply(lambda x: "drn" in x), "obsnme"]
    # convert from model units to (positive) CFS for plotting
    nz_obs.loc[drn_names, "obsval"] *= -1.0 / (60.0 * 60.0 * 24.0)

    # unique nonzero observation groups (site names)
    nz_grps = nz_obs.obgnme.unique()

    # if focus is True, drop the non-focus sites
    focus_sites = ["comal", "sanmar", str(j17_id), "j-17", str(j27_id), "j-27"]
    focus_labels = ["Comal", "San Marcos", "J-17", "J-17", "J-27", "J-27"]
    nz_grps.sort()
    if focus:
        keep = []
        labels = []
        for nz_grp in nz_grps:

            for fs, lab in zip(focus_sites, focus_labels):
                print(nz_grp, fs, fs in nz_grp)
                if fs in nz_grp:
                    keep.append(nz_grp)
                    labels.append(lab)
        nz_grps = keep



    with PdfPages(plt_name) as pdf:
        xmn, xmx = pd.to_datetime(start_datetime), pd.to_datetime(end_datetime)
        if focus:
            ax_per_page = 4
            fig, axes = plt.subplots(ax_per_page, 1, figsize=(7, 7))
        else:
            ax_per_page = 5
            fig, axes = plt.subplots(ax_per_page, 1, figsize=(8.5, 11))
        ax_count = 0

        # process each unique non-zero obs group
        for igrp, nz_grp in enumerate(nz_grps):
            ax = axes[ax_count]
            obs_grp = nz_obs.loc[nz_obs.obgnme == nz_grp, :].copy()
            obs_grp.sort_values(by="datetime", inplace=True)

            ireal_str = ""
            # process each ensemble
            for en, fc, en_name in zip([pr_en, pt_en, base_en], ['k', 'b', 'r'], ["prior", "post", "noise"]):
                if en is None:
                    continue
                # get the ensemble block for this group
                en_grp = en.loc[:, obs_grp.obsnme].copy()

                # convert the DRN outputs to positive CFS
                if "drn" in nz_grp:
                    en_grp = en_grp * -1.0 / (60.0 * 60.0 * 24.0)
                    # obs_grp.loc[:,"obsval"] *= -1.0 / (60.0 * 60.0 * 24.0)
                else:
                    # for GW level obs, check for invalid values
                    vals = en_grp.values
                    ivals = vals[vals == hdry]
                    if ivals.shape[0] > 0:
                        ireal = 0
                        for real in en_grp.index:
                            if hdry in en_grp.loc[real, :].values:
                                ireal += 1
                        ireal_str += ", {0} invalid values across {1} realizations in {2}". \
                            format(ivals.shape[0], ireal, en_name)

                    # mask invalid?
                    if mask_invalid:
                        en_grp.values[en_grp.values == hdry] = np.NaN
                # if the posterior ensemble was passed and this is the prior ensemble
                # dont plot it.
                if pt_en is not None and en_name == "prior":
                    pass
                else:
                    # plot each realization and thin-line trace
                    [ax.plot(obs_grp.datetime, en_grp.loc[i, :], color=fc, lw=0.025, alpha=0.1) for i in
                     en_grp.index.values]
                # if the "base" realization is found, plot it with a heavier line
                if "base" in en_grp.index:
                    print("base found",en_name,nz_grp)
                    if en_name == "prior":
                        ax.plot(obs_grp.datetime, en_grp.loc["base", :], color="k", dashes=(2,2),lw=1.5)
                    else:
                        ax.plot(obs_grp.datetime, en_grp.loc["base", :], color=fc,lw=1.5,alpha=0.35)

            # set axis limits
            ax.set_xlim(xmn, xmx)
            ymn, ymx = obs_grp.obsval.min() * 0.9, obs_grp.obsval.max() * 1.1
            # for the DRN series, set the y-axis limit to the observed spring flow
            if pr_en is not None and "base" in pr_en.index:
                en_grp = pr_en.loc["base", obs_grp.obsnme].copy()
                if "drn" in nz_grp:
                    en_grp = en_grp * -1.0 / (60.0 * 60.0 * 24.0)
                ymn = min(ymn, en_grp.min())
                ymx = max(ymx, en_grp.max())

            ax.set_ylim(ymn, ymx)

            # build up meaningful axis titles
            site_id = nz_grp.split('_')[1]
            try:
                site_id = int(site_id)
            except:
                pass
            title = "site: {0}".format(site_id)
            if site_id == j17_id:
                title += " (J-17)"
            elif site_id == j27_id:
                title += " (J-27)"
            if len(ireal_str) > 0:
                title += ireal_str
            if focus:
                ax.set_title("{0}) {1}".format(abet[igrp], labels[igrp]), loc="left")
            else:
                ax.set_title(title, loc="left")
            if "drn" in nz_grp:
                ax.set_ylabel("flow ($\\frac{ft^3}{s}$)")
            else:
                ax.set_ylabel("head (ft)")
            ax_count += 1
            if ax_count >= ax_per_page:
                plt.tight_layout()
                pdf.savefig()
                plt.close(fig)
                if not focus:
                    fig, axes = plt.subplots(ax_per_page, 1, figsize=(8.5, 11))
                    ax_count = 0

        if not focus:
            for rem_ax in range(ax_count, ax_per_page):
                axes[rem_ax].set_xticks([])
                axes[rem_ax].set_yticks([])
                axes[rem_ax].axis("off")
            plt.tight_layout()
            pdf.savefig()


def reweight_ensemble(m_d, t_d, case="eaa_ver"):
    """reweight the non-zero observations in the control file
    using a management-focused strategy with the ensemble mean
    residuals

    Args:
        m_d (str): master directory
        t_d (str): template directory
        case (str): the PEST case.  Default is "eaa_ver"


    """

    # load the control file
    pst = pyemu.Pst(os.path.join(m_d, case + ".pst"))
    obs = pst.observation_data

    # load the prior simulated output ensemble
    pr_en = pyemu.ObservationEnsemble.from_binary(pst=pst,
                        filename=os.path.join(m_d, case + ".0.obs.jcb"))
    assert "base" in pr_en.index

    # build a PEST-style residual dataframe using the mean ensmeble values
    res_df = pd.DataFrame({"modelled": pr_en.loc["base", :].values,
                           "group": pst.observation_data.loc[pr_en.columns, "obgnme"].values,
                           "name": pr_en.columns.values,
                           "measured": pst.observation_data.loc[pr_en.columns, "obsval"].values}, \
                          index=pr_en.columns)
    pst.set_res(res_df)
    init_nz_obs = pst.nnz_obs_names


    # drop the 5 realizations that cause the most dry values
    vals = []
    for real in pr_en.index:
        nz_real = pr_en.loc[real, init_nz_obs].values
        count = nz_real[nz_real == hdry].shape[0]
        vals.append(count)
    df = pd.DataFrame({"invalid_count": vals}, index=pr_en.index)
    df.sort_values(by="invalid_count", inplace=True, ascending=False)
    pr_en = pr_en.loc[df.iloc[5:].index]
    assert pr_en.shape == pr_en.dropna().shape

    # look for any nz obs yielding dry/inactive/insane values - zero weight these
    # so we dont have issues during the PESTPP-IES iterations with phi exploding
    nz_obs = obs.loc[pst.nnz_obs_names, :].copy()
    abs_mx = np.abs(pr_en.loc[:, nz_obs.obsnme].values).max(axis=0)
    # print(abs_mx)
    tol = 1.0e+10
    busted = abs_mx > tol
    busted_obs_names = nz_obs.obsnme.loc[busted]
    print(busted_obs_names)
    obs.loc[busted_obs_names, "weight"] = 0.0

    # re-get nz_obs now
    nz_obs = obs.loc[pst.nnz_obs_names, :].copy()
    print("removed {0} obs for insane values".format(busted_obs_names.shape[0]))
    print(len(init_nz_obs), pst.nnz_obs)

    # now use standard measurement based noise to check for prior-data conflict
    nz_hds_names = nz_obs.loc[nz_obs.obgnme.apply(lambda x: x.startswith("hds")), "obsnme"]
    obs.loc[nz_hds_names, "weight"] = 0.5
    nz_drn_names = nz_obs.loc[nz_obs.obgnme.apply(lambda x: x.startswith("drn")), "obsnme"]
    obs.loc[nz_drn_names, "weight"] = 1.0 / (obs.loc[nz_drn_names, "obsval"].apply(np.abs) * 0.25)
    # correct for drn flows that are zero
    drn_obs = nz_obs.loc[nz_drn_names, :]
    zero_flow_drn = drn_obs.loc[drn_obs.obsval == 0.0, "obsnme"]
    obs.loc[zero_flow_drn, "weight"] = 0.01
    # correct for giant weights
    too_high = drn_obs.loc[drn_obs.weight > 0.01, "obsnme"]
    obs.loc[too_high, "weight"] = 0.01
    print(obs.loc[nz_drn_names, "weight"].min(), obs.loc[nz_drn_names, "weight"].max())

    print(len(nz_hds_names) + len(nz_drn_names), pst.nnz_obs)
    # pst.write(os.path.join(m_d,"eaa_ver_reweight.pst"))

    # generate an obs + noise ensemble using measurement-error-based weights
    base_en = pyemu.ObservationEnsemble.from_gaussian_draw(pst=pst, num_reals=pr_en.shape[0])
    base_en._df.index = pr_en._df.index

    # check for prior-data conflict: for each non-zero weighted obs,
    # if the simulated output ensemble doesnt overlap with the
    # obs+noise ensemble, then we can't expect the DA process
    # to reproduce these observations.
    conflicted = []
    for oname in pst.nnz_obs_names:
        d_mn, d_mx = base_en.loc[:, oname].min(), base_en.loc[:, oname].max()
        p_mn, p_mx = pr_en.loc[:, oname].min(), pr_en.loc[:, oname].max()
        if d_mn > p_mx and p_mn > d_mx:
            print(oname, d_mn, d_mx, p_mn, p_mx)
            conflicted.append(oname)
    pst.observation_data.loc[conflicted, "weight"] = 0.0

    # re-get non-zero weighted observations
    obs = pst.observation_data
    nz_obs = obs.loc[pst.nnz_obs_names, :]

    # extract the non-zero weight obs blocks from the obs+noise and
    # prior simulated output ensembles and save them for restart (to
    # save runs)
    base_en = base_en.loc[:, nz_obs.obsnme]
    base_en.to_binary(os.path.join(t_d, "obs.jcb"))
    pr_en.to_binary(os.path.join(t_d, "restart_obs.jcb"))
    par_en = pyemu.ParameterEnsemble.from_binary(pst=pst, filename=os.path.join(t_d, "prior.jcb"))
    par_en = par_en.loc[pr_en.index, :]
    assert par_en.shape == par_en.dropna().shape
    par_en.to_binary(os.path.join(t_d, "restart_prior.jcb"))

    # enforce subjective management-focused weighting scheme
    tags = ["comal", "sanmar", str(j17_id), str(j27_id)]
    ratio = 100.0
    obsgrp_dict = {grp: 1.0 for grp in pst.nnz_obs_groups}
    tagged_groups = []
    for tag in tags:
        t_obs = nz_obs.loc[nz_obs.obsnme.apply(lambda x: tag in x), :]
        if t_obs.shape[0] == 0:
            raise Exception(tag)
        gnames = t_obs.obgnme.unique()
        if len(gnames) > 1:
            raise Exception(tag, gnames)
        obsgrp_dict[gnames[0]] = ratio
        tagged_groups.append(gnames[0])
    print(pst.phi)
    org_pc = pst.phi_components
    pst.adjust_weights(obsgrp_dict=obsgrp_dict)
    print(pst.phi)
    pc = pst.phi_components
    for grp in pst.nnz_obs_groups:
        print(grp, org_pc[grp], pc[grp])

    # set some ++ options for PESTPP-IES restart
    pst.pestpp_options["ies_par_en"] = "restart_prior.jcb"
    pst.pestpp_options["ies_obs_en"] = "obs.jcb"
    pst.pestpp_options["ies_restart_obs_en"] = "restart_obs.jcb"
    pst.control_data.noptmax = 1
    pst.write(os.path.join(t_d, case + "_reweight.pst"))


def _prep_for_parallel(b_d, pst_name, m_d=None, noptmax=6, with_loc=False, overdue_giveup_fac=5):
    """prepare a given template directory for parallel execution.  Makes a directory copy with
    temporary files removed, and (optionally) a master directory

    Args:
        b_d (str): base template directory.  This is copied to b_d+"_parallel"
        pst_name (str): control file name
        m_d (str): master directory to create.  Default is None (dont create)
        noptmax (int): number of PESTPP-IES DA iterations
        with_loc (bool): flag to use localization.  If True, "loc.jcb" must exists
            in b_d
        overdue_giveup_fac (float): factor to use to limit the amount of time waiting for
            slow runs to finish.  Default is 5, which means any run that takes longer than 5 times
            the mean run time will be marked as a run failure and killed.

    """

    # copy b_d to b_d+"_parallel" (e.g. the parallel template dir)
    ct_d = "{0}_parallel".format(b_d)
    if os.path.exists(ct_d):
        shutil.rmtree(ct_d)
    shutil.copytree(b_d, ct_d)

    # remove any temp files not needed since the parallel template dir will
    # need to be copied a bunch of times
    rm_ext_list = [".list", ".jcb", ".out", ".hds", ".rec", ".ftl", ".mt3d", \
                   ".ucn", ".rei", ".rns", ".rnj", ".cbb", ".cbd"]
    rm_tag_list = ["_setup_"]
    for rm_ext in rm_ext_list:
        rm_files = [f for f in os.listdir(ct_d) if f.lower().endswith(rm_ext)]
        [os.remove(os.path.join(ct_d, rm_file)) for rm_file in rm_files]

    for rm_tag in rm_tag_list:
        rm_files = [f for f in os.listdir(ct_d) if rm_tag in f.lower()]
        [os.remove(os.path.join(ct_d, rm_file)) for rm_file in rm_files]

    a_d = os.path.join(ct_d, "arr_mlt")
    [os.remove(os.path.join(a_d, f)) for f in os.listdir(a_d)[1:]]

    # copy the binaries into the parallel template dir
    for bin in os.listdir(bin_path):
        shutil.copy2(os.path.join(bin_path, bin), os.path.join(ct_d, bin))

    # copy pyemu and flopy into the parallel template dir
    shutil.copytree(os.path.join("flopy"), os.path.join(ct_d, "flopy"))
    shutil.copytree(os.path.join("pyemu"), os.path.join(ct_d, "pyemu"))

    # some platform specific things: condor is used only on windows
    if "window" in platform.platform().lower():
        shutil.copy2(os.path.join("python.zip"), os.path.join(ct_d, "python.zip"))
        agent_zip = "condor_agent"
        if os.path.exists(agent_zip + ".zip"):
            os.remove(agent_zip + ".zip")
        shutil.make_archive(agent_zip, "zip", ct_d)

    # if a master dir is requested, prepare it
    if m_d is not None:
        # make sure we dont stomp on an existing dir
        assert not os.path.exists(m_d), "master dir already exists {0}".format(m_d)
        # copy b_d to m_d
        shutil.copytree(b_d, m_d)
        # copy flopy, pyemu and binaries into m_d
        shutil.copytree("flopy", os.path.join(m_d, "flopy"))
        shutil.copytree("pyemu", os.path.join(m_d, "pyemu"))
        for bin in os.listdir(bin_path):
            shutil.copy2(os.path.join(bin_path, bin), os.path.join(m_d, bin))

        # load the control file
        pst = pyemu.Pst(os.path.join(m_d, pst_name))

        # if a localizer is being used
        if with_loc:
            pst.pestpp_options["ies_localizer"] = "loc.jcb"
            # make sure it exists
            loc_file = os.path.join(b_d, "loc.jcb")
            assert os.path.exists(loc_file)
            # load the localizer and make sure it is in sync with
            # the current weighting strategy (only non-zero obs can be
            # in the localizer row names)
            loc = pyemu.Matrix.from_binary(loc_file).to_dataframe()
            loc = loc.loc[pst.nnz_obs_names, :]
            pyemu.Matrix.from_dataframe(loc).to_coo(os.path.join(m_d, "loc.jcb"))
            # use 20 threads for localized solve
            pst.pestpp_options["ies_num_threads"] = 20

        # if the initial parameter ensemble is not in the ++ options,
        # set it to the prior ensemble
        if "ies_par_en" not in pst.pestpp_options:
            pst.pestpp_options["ies_par_en"] = "prior.jcb"
            pst.pestpp_options["ies_num_reals"] = 100

        # save binary formats
        pst.pestpp_options["ies_save_binary"] = True
        # set overdue_giveup_fac
        pst.pestpp_options["overdue_giveup_fac"] = overdue_giveup_fac
        # if we are iterating, set the bad phi to protect against
        # invalid (hdry) outputs
        if noptmax != -1:
            pst.pestpp_options["ies_bad_phi"] = 1e20

        # just a guess...
        pst.pestpp_options["ies_initial_lambda"] = 1000.0

        # number of iterations
        pst.control_data.noptmax = noptmax

        # save the control file into the master dir
        pst.write(os.path.join(m_d, pst_name))


def transfer_hist_pars_to_scenario(hist_en_filename, scen_en_filename):
    """transfer static (shared) parameters from a history parameter ensemble
    to a scenario parameter ensemble

    Args:
        hist_en_filename (str): a binary-format file holding a parameter ensemble
            for the history model
        scen_en_filename (str): a binary-format file holding a parameter ensemble
            for the scenario model



    """

    # load the control files and ensembles for both the history and scenario models
    hist_pst = pyemu.Pst(os.path.join(h_dir, "eaa_ver.pst"))
    hist_en = pyemu.ParameterEnsemble.from_binary(pst=hist_pst, filename=hist_en_filename)
    scen_pst = pyemu.Pst(os.path.join(s_dir, "eaa_pred.pst"))
    scen_en = pyemu.ParameterEnsemble.from_binary(pst=scen_pst, filename=scen_en_filename)

    # if the indices are not the same?
    if list(hist_en.index.values) != list(scen_en.index.values):
        # use the history en index
        scen_en = scen_en.loc[hist_en.index, :]
        assert scen_en.shape == scen_en.dropna().shape
    # tags for shared parameters
    props = {"hk", "ss", "sy", "dc", "hb"}
    hist_par = hist_pst.parameter_data
    scen_par = scen_pst.parameter_data

    # get lists of shared parameters in both ensembles
    hist_parnames = hist_par.loc[hist_par.parnme.apply(lambda x: True in [True for p in props if p in x]), "parnme"]
    scen_parnames = scen_par.loc[scen_par.parnme.apply(lambda x: True in [True for p in props if p in x]), "parnme"]

    # work out the common names between the two sets of names
    common_pars = list(set(hist_parnames).intersection(set(scen_parnames)))
    common_pars.sort()

    # update the static pars in the scenario ensemble from the history ensemble
    scen_en._df.loc[:, common_pars] = hist_en._df.loc[:, common_pars]

    # save the update scenario ensemble
    scen_en.to_binary(os.path.join(s_dir, "updated_par_en.jcb"))

    # update the scenario control file ++ option for the parameter ensemble
    scen_pst.pestpp_options["ies_par_en"] = "updated_par_en.jcb"
    # save
    scen_pst.write(os.path.join(s_dir, "eaa_pred.pst"))


def plot_below(pst_file, pr_en_file, pt_en_file=None,
               plt_name="months_below.pdf", thres=30.0):
    """plot the consecutive and cumulative Comal springs discharge
    months where springflow is below a given threshold

    Args:
        pst_file (str): control file name
        pr_en_file (str): binary-format prior scenario output ensemble
        pt_en_file (str): binary-format posterior scenario output ensemble
        plt_name (str): plot name.  Default is "months_below.pdf"
        thres (float): threshold (in positive CFS) for Comal springflow


    """

    # load the control file
    pst = pyemu.Pst(pst_file)

    # load the prior ensemble
    pr_en = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=pr_en_file)

    # load the posterior ensemble?
    pt_en = None
    if pt_en_file is not None:
        pt_en = pyemu.ObservationEnsemble.from_binary(pst=pst,
                                                      filename=pt_en_file)

    # get just the comal springs ensemble blocks
    # and convert to positive CFS
    cols = [c for c in pr_en.columns if "comal" in c]
    pr_en = pr_en.loc[:, cols] * -1. / 60. / 60. / 24.
    if pt_en is not None:
        pt_en = pt_en.loc[:, cols] * -1. / 60. / 60. / 24.
    obs = pst.observation_data
    comal_obs = obs.loc[cols, "obsval"] * -1. / 60. / 60. / 24.

    # get the "truth" number of months below thres
    comal_truth = comal_obs.loc[comal_obs < thres].shape[0]

    # count prior ensemble consecutive and cumulative months below
    pr_count, pr_max = [], []
    for real in pr_en.index:
        s = pr_en.loc[real, cols]
        isbelow = s < thres
        below = s.loc[isbelow]
        isbelow = isbelow.astype(int)
        # consecutive months below
        mx = 0
        longest = 0
        current = 0
        for num in isbelow.values:
            if num == 1:
                current += 1
            else:
                longest = max(longest, current)
                current = 0

        mx = max(longest, current)

        c = below.shape[0]
        pr_count.append(c)
        pr_max.append(mx)
    # count prior ensemble consecutive and cumulative months below
    if pt_en is not None:
        pt_count, pt_max = [], []
        for real in pt_en.index:
            s = pt_en.loc[real, cols]
            isbelow = s < thres
            below = s.loc[isbelow]
            isbelow = isbelow.astype(int)
            mx = 0
            longest = 0
            current = 0
            for num in isbelow.values:
                if num == 1:
                    current += 1
                else:
                    longest = max(longest, current)
                    current = 0

            mx = max(longest, current)

            c = below.shape[0]
            pt_count.append(c)
            pt_max.append(mx)

    # truth consecutive months below
    mx = 0
    longest = 0
    current = 0
    isbelow = comal_obs < thres
    isbelow = isbelow.astype(int)
    for num in isbelow.values:
        if num == 1:
            current += 1
        else:
            longest = max(longest, current)
            current = 0

    truth_mx = max(longest, current)
    print("comal",comal_truth,truth_mx)
    # use geometric series for binning bc appearent log distro
    bins = np.geomspace(0.1, max(pr_max), 20)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))
    ax = axes[0]
    ax.hist(pr_max, bins=bins, facecolor="0.5", edgecolor="none", alpha=0.35)
    if pt_en is not None:
        ax.hist(pt_max, bins=bins, facecolor="b", edgecolor="none", alpha=0.5)
    ax.set_ylabel("increasing probability density")
    ymin = ax.get_ylim()
    ax.plot([truth_mx, truth_mx], ymin, "r-")
    ax.set_xlabel("months")
    ax.set_yticks([])
    ax.set_ylim(ymin)
    ax.set_title("A) simulated consecutive Comal months below {0}".format(thres) + \
                 " $\\frac{ft^3}{s}$", loc="left")

    ax = axes[1]
    ax.hist(pr_count, bins=bins, facecolor="0.5", edgecolor="none", alpha=0.35)
    if pt_en is not None:
        ax.hist(pt_count, bins=bins, facecolor="b", edgecolor="none", alpha=0.5)
    ax.set_ylabel("increasing probability density")
    ymin = ax.get_ylim()
    ax.plot([comal_truth, comal_truth], ymin, "r-")
    ax.set_xlabel("months")
    ax.set_yticks([])
    ax.set_ylim(ymin)
    ax.set_title("B) simulated Comal months below {0}".format(thres) + \
                 " $\\frac{ft^3}{s}$", loc="left")
    plt.tight_layout()
    plt.savefig(plt_name)
    plt.show()


def build_temporal_localizer(t_dir, pst_name="eaa_ver.pst",save=True):
    """build a localizer for temporal parameters

    """
    m = flopy.modflow.Modflow.load(h_nam_file, model_ws=t_dir, load_only=["dis"], 
                                   check=False, forgive=False)
    totim = list(np.cumsum(m.dis.perlen.array))
    totim.insert(0, 0.0)
    # stress period begin datetime
    kper_dt = pd.to_datetime(m.start_datetime) + pd.to_timedelta(totim, unit="d")

    pst = pyemu.Pst(os.path.join(t_dir, pst_name))
    par = pst.parameter_data.loc[pst.adj_par_names, :].copy()
    obs = pst.observation_data.loc[pst.nnz_obs_names, :].copy()
    obs.loc[:, "datetime"] = pd.to_datetime(obs.obsnme.apply(lambda x: x.split("_")[-1]))

    # recharge constants
    df = pd.read_csv(os.path.join(t_dir, "arr_pars.csv"), index_col=0)
    df = df.loc[df.mlt_file.apply(lambda x: "rech" in x and "dat_cn" in x), :]
    df.loc[:, "kper"] = df.org_file.apply(lambda x: int(x.split('.')[0].split('_')[-1]))
    df.loc[:, "cnst"] = df.mlt_file.apply(lambda x: int(x.split('.')[0].split('rech')[-1]))
    zone_kper = {z: k for z, k in zip(df.cnst, df.kper)}
    cn_rech_par = par.loc[par.pargp.apply(lambda x: "cn_rech" in x), :].copy()
    cn_rech_par.loc[:, "kper"] = cn_rech_par.pargp.apply(lambda x: zone_kper[int(x.replace("cn_rech", ""))])
    cn_rech_par.loc[:, "datetime"] = cn_rech_par.kper.apply(lambda x: kper_dt[x])

    # recharge zones
    df = pd.read_csv(os.path.join(t_dir, "arr_pars.csv"), index_col=0)
    df = df.loc[df.mlt_file.apply(lambda x: "rech" in x and "dat_zn" in x), :]
    df.loc[:, "kper"] = df.org_file.apply(lambda x: int(x.split('.')[0].split('_')[-1]))
    df.loc[:, "zone"] = df.mlt_file.apply(lambda x: int(x.split('.')[0].split('rech')[-1]))
    zone_kper = {z: k for z, k in zip(df.zone, df.kper)}
    zn_rech_par = par.loc[par.pargp.apply(lambda x: "zn_rech" in x), :].copy()
    zn_rech_par.loc[:, "kper"] = zn_rech_par.pargp.apply(lambda x: zone_kper[int(x.replace("zn_rech", ""))])
    zn_rech_par.loc[:, "datetime"] = zn_rech_par.kper.apply(lambda x: kper_dt[x])

    # print(zn_rech_par.datetime.unique())
    # print(cn_rech_par.datetime)

    # wel pars
    wel_par = par.loc[par.pargp == "welflux", :].copy()
    wel_par.loc[:, "kper"] = wel_par.parnme.apply(lambda x: int(x.split('_')[-1]))
    wel_par.loc[:, "datetime"] = wel_par.kper.apply(lambda x: kper_dt[x])
    # print(wel_par.datetime)

    # init conditions
    strt_par = par.loc[par.pargp.apply(lambda x: "strt" in x), :].copy()
    strt_grps = strt_par.pargp.unique()

    temp_par = wel_par.copy()
    temp_par = temp_par.append(zn_rech_par)
    temp_par = temp_par.append(cn_rech_par)
    for sgrp in strt_grps:
        temp_par.loc[sgrp, "datetime"] = pd.to_datetime(m.start_datetime)
        temp_par.loc[sgrp, "pargp"] = sgrp
        temp_par.loc[sgrp, "parnme"] = sgrp

    temp_grps = set(temp_par.pargp.unique().tolist())

    static_groups = par.loc[par.pargp.apply(lambda x: x not in temp_grps), "pargp"].unique().tolist()

    loc_cols = static_groups
    loc_cols.extend(temp_par.parnme.tolist())
    print(len(loc_cols))

    loc = pyemu.Matrix.from_names(row_names=pst.nnz_obs_names, col_names=loc_cols).to_dataframe()
    loc.loc[:, :] = 1.0
    loc.loc[:, temp_par.parnme] = 0.0

    u_obs_dts = obs.datetime.unique()
    tol = pd.to_timedelta(550, unit="d")
    for udt in u_obs_dts:
        obs_dt = obs.loc[obs.datetime == udt, :].copy()
        # find pars that are within the previous 18 months
        utemp_par = temp_par.copy()
        utemp_par.loc[:, "td"] = udt - temp_par.datetime
        utemp_par = utemp_par.loc[utemp_par.td.apply(lambda x: x.days > 0 and x.days < tol.days), :]
        print(udt, utemp_par.shape[0])
        loc.loc[obs_dt.obsnme, utemp_par.parnme] = 1.0
    
    if save:
        locmat = pyemu.Matrix.from_dataframe(loc)
        locmat.to_coo(os.path.join(t_dir, "loc.jcb"))
    return loc

def run_condor(pst_name, m_d, port=4200):
    """run PESTPP-IES in parallel using an HTCondor array

    Args:
        pst_name (str): control file name
        m_d (str): existing master directory to start the master
            instance in
        port (int): tcp port for PESTPP-IES communication

    """

    # make sure condor logging directory exists
    if not os.path.exists("log"):
        os.mkdir("log")

    # read in the generic condor submit file lines
    f = open("_generic_.sub", 'r')
    lines = []
    # replace the "arguments" line with the desired port number and
    # control file name
    for line in f:
        if "arguments" in line:
            raw = line.strip().split()
            raw[-1] = pst_name
            raw[-2] = str(port)
            line = ' '.join(raw) + "\n"
        lines.append(line)

    # write out a new condor submit file
    sub_file = pst_name + ".sub"
    with open(sub_file, 'w') as f:
        for line in lines:
            f.write(line)

    # issue the condor submit command (async)
    os.system("condor_submit {0} > submit.dat".format(sub_file))

    # get the condor cluster number from the condor_submit stdout
    cluster_num = _get_cluster_num()

    # run the master PESTPP-IES instance
    pyemu.os_utils.run("pestpp-ies.exe {0} /h :{1} > master.stdout".format(pst_name, port), cwd=m_d)

    # timeout 1 min
    time.sleep(60)
    # issue condor_rm for the current cluster number
    os.system("condor_rm {0}".format(cluster_num))
    # timeout 1 min
    time.sleep(60)
    # issue condor_rm for current cluster number with forced removal
    os.system("condor_rm {0} -forcex".format(cluster_num))
    # timeout 2 mins
    time.sleep(120)


def _get_cluster_num():
    """get the cluster number from the condor_submit stdout capture

    Returns:
        cluster_num (int)


    """



    cluster_num = None
    with open("submit.dat", 'r') as f:
        for line in f:
            if "submitted to cluster" in line:
                cluster_num = int(line.strip().split()[-1].replace(".", ""))
    if cluster_num is None:
        raise Exception()
    return cluster_num


def setup_history_model():
    """high-level function to prepare the history model for PESTPP-IES

    """
    # add_ij_to_hds_smp(h_crd_file)
    # prep the model
    _setup_model(old_h_dir, "temp_history", h_start_datetime, h_nam_file)
    # fix the WEL file for parameterization
    _rectify_wel("temp_history", h_nam_file)
    # setup the pest interface and draw from the Prior
    _setup_pst("temp_history_wel", h_dir, h_nam_file)
    # add the GW level and spring discharge observation locations
    _add_smp_obs_to_pst("temp_history_wel", h_dir, "eaa_ver.pst", h_nam_file, h_crd_file)
    # set the GW level and spring discharge observation values
    _set_obsvals(h_dir, h_nam_file, h_hds_file, h_drn_file, "eaa_ver.pst", run=False)


def setup_scenario_model():
    """high-level function to prepare the scenario model for PESTPP-IES
    """
    # add_ij_to_hds_smp(s_crd_file)
    # prep the model
    _setup_model(old_s_dir, "temp_scenario", s_start_datetime, s_nam_file)
    # fix the WEL file for parameterization
    _rectify_wel("temp_scenario", s_nam_file)
    # setup the pest interface and draw from the Prior
    _setup_pst("temp_scenario_wel", s_dir, s_nam_file)
    # add the GW level and spring discharge observation locations
    _add_smp_obs_to_pst("temp_scenario_wel", s_dir, "eaa_pred.pst", s_nam_file, s_crd_file)
    # set the GW level and spring discharge observation values
    _set_obsvals(s_dir, s_nam_file, s_hds_file, s_drn_file, "eaa_pred.pst", run=False)


def setup_models_parallel():
    """call setup_history_model and setup_scenario_model in
    parallel using multiprocessing to save wall time
    """

    p_hist = mp.Process(target=setup_history_model)
    p_scen = mp.Process(target=setup_scenario_model)
    p_hist.start()
    p_scen.start()
    p_hist.join()
    p_scen.join()

    # use the static par values from the history ensemble for consistency
    transfer_hist_pars_to_scenario(os.path.join(h_dir,"prior.jcb"),
                                     os.path.join(s_dir,"prior.jcb"))


def plot_array_pars(post_iter):
    """plot the array multiplier parameterization summary plots
    as seen in the supplementary information

    Args:
        post_iter (int): PESTPP-IES iteration number to use as the posterior parameter ensemble

    """

    # the master history directory where PESTPP-IES ran
    m_d = "master_history_reweight"

    # load the history model
    m = flopy.modflow.Modflow.load("eaa_ver.nam", model_ws=m_d, 
                                   load_only=["dis", "bas6"], check=False,
                                   forgive=False)
    # get the MODFLOW ibound array as a numpy ndarray
    ib = m.bas6.ibound[0].array

    # load the array parameterization configuration dataframe
    arr_df = pd.read_csv(os.path.join(m_d, "arr_pars.csv"))

    # get a list of static (not recharge) model input arrays
    mi_files = arr_df.loc[arr_df.model_file.apply(lambda x: "rech" not in x), :].model_file.unique()

    # load the control file
    pst = pyemu.Pst(os.path.join(m_d, "eaa_ver_reweight.pst"))
    # load the prior observation ensemble
    o_pr_en = pyemu.ObservationEnsemble.from_binary(pst=pst,
        filename=os.path.join(m_d,"eaa_ver_reweight.0.obs.jcb"))
    # load the posterior observation ensemble
    o_pt_en = pyemu.ObservationEnsemble.from_binary(pst=pst,
        filename=os.path.join(m_d,"eaa_ver_reweight.{0}.obs.jcb".format(post_iter)))
    #get the prior and posterior phi vectors for axix labeling
    pr_pv = o_pr_en.phi_vector
    pt_pv = o_pt_en.phi_vector

    # load the prior parameter ensemble
    pr_en = pyemu.ParameterEnsemble.from_binary(pst=pst,
            filename=os.path.join(m_d, "eaa_ver_reweight.0.par.jcb"))
    # load the posterior parameter ensemble
    pt_en = pyemu.ParameterEnsemble.from_binary(pst=pst,
         filename=os.path.join(m_d, "eaa_ver_reweight.{0}.par.jcb".\
                                    format(post_iter)))

    # change the python workding directory into the master directory
    os.chdir(m_d)

    # property array name tags
    prop_tags = {"strt": "initial conditions", "hk": "hydraulic conductivity",
                 "ss": "specific storage", "sy": "specific yield"}
    # multiplier array name tags
    mlt_tags = {"dat_pp": "pilot points", "dat_gr": "grid-scale", "dat_cn": "global"}
    # property unit dictionary for colorbar labeling
    unit_dict = {'strt': "feet", "hk": "$log_{10}\\frac{ft}{day}$",
                 "ss": "$log_{10}\\frac{1}{foot}$", "sy": "$log_{10}\\frac{ft^3}{ft^3}$"}


    with PdfPages("array_pars.pdf") as pdf:

        ireal = 0
        # for each realization in the posterior ensemble
        for real in pt_en.index:
            # if not in the prior, continue
            if real not in pr_en.index:
                continue
            # reset the control file inital parameter values (e.g. "parval") to
            # the prior realization values
            pst.parameter_data.loc[:, "parval1"] = pr_en.loc[real, pst.par_names]

            # write the "model input values" using the PEST template files
            print(real, "writing pr inputs")
            pst.write_input_files()

            # apply the array multiplication process, including pilot point
            # interpolation to the grid
            print(real, "pr apply")
            pyemu.helpers.apply_array_pars()
            # save multiplier array directory since we are
            # gonna do the same ops for the posterior
            if os.path.exists("arr_mlt_pr"):
                shutil.rmtree("arr_mlt_pr")
            shutil.copytree("arr_mlt", "arr_mlt_pr")
            # also save the model input files
            for mi_file in mi_files:
                shutil.copy2(mi_file, "pr_" + os.path.split(mi_file)[-1])

            # now replace the initial parameter values with the cooresponding posterior realization
            pst.parameter_data.loc[:, "parval1"] = pt_en.loc[real, pst.par_names]
            # write model input files with the PEST template files
            print(real, "writing pt inputs")
            pst.write_input_files()

            # run the array multiplier process
            print(real, "pt apply")
            pyemu.helpers.apply_array_pars()
            print(real, "plot")

            # for each model input file (e.g hk array, sy array, ss array, etc)
            # plot the original (existing) model input array, then each of the
            # multiplier arrays for both the prior and posterior and finally the
            #resulting new model input array (prior and posterior).
            for mi_file in mi_files:

                fig, axes = plt.subplots(5, 2, figsize=(8.5, 11))

                # org_ax = axes[0, 0]
                org_ax = plt.subplot(5, 1, 1)
                prop_label = None
                for tag, lab in prop_tags.items():
                    if tag in mi_file:
                        prop_label = lab
                        cb_label = unit_dict[tag]
                if prop_label is None:
                    raise Exception()
                # load the original model input array
                org_arr = np.loadtxt(os.path.join("arr_org", mi_file))

                # load the corresponding processed (multiplied) prior and posterior
                # model input arrays
                pt_mi_arr = np.loadtxt(mi_file)
                pr_mi_arr = np.loadtxt("pr_" + os.path.split(mi_file)[-1])

                # all model input arrays except initial conditions are better represented
                # by log transform
                if "strt" not in mi_file:
                    org_arr = np.log10(org_arr)
                    pr_mi_arr = np.log10(pr_mi_arr)
                    pt_mi_arr = np.log10(pt_mi_arr)
                # mask inactive areas
                org_arr[ib == 0] = np.NaN
                pr_mi_arr[ib == 0] = np.NaN
                pt_mi_arr[ib == 0] = np.NaN

                # work on the min and max range for the color bar
                mi_vmin = min(np.nanmin(org_arr), np.nanmin(pr_mi_arr), np.nanmin(pt_mi_arr))
                mi_vmax = max(np.nanmax(org_arr), np.nanmax(pt_mi_arr), np.nanmax(pt_mi_arr))

                # plot the original and add a colorbar
                c = org_ax.imshow(org_arr, vmin=mi_vmin, vmax=mi_vmax)
                divider = make_axes_locatable(org_ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                c = plt.colorbar(c, cax=cax)
                c.set_label(cb_label)
                org_ax.set_title("A) original {1} model input for realization {2}".format(real, prop_label, real),
                                 loc="left")
                org_ax.set_xticks([])
                org_ax.set_yticks([])

                # plot the multiplier arrays
                mlt_files = arr_df.loc[arr_df.model_file == mi_file, "mlt_file"]
                pt_arrs, pr_arrs, mlt_labels = [], [], []
                mlt_vmin, mlt_vmax = 1.0e+10, -1.0e+10
                ax_c = 1
                for i, mlt_file in enumerate(mlt_files):
                    mlt_label = None
                    for tag, lab in mlt_tags.items():
                        if tag in mlt_file:
                            mlt_label = lab
                    # load the prior multipier array
                    pr_arr = np.loadtxt(os.path.join("arr_mlt_pr", os.path.split(mlt_file)[-1]))
                    # load the posterior multiplier array
                    pt_arr = np.loadtxt(mlt_file)
                    # log transform and mask
                    pr_arr = np.log10(pr_arr)
                    pt_arr = np.log10(pt_arr)
                    pr_arr[ib == 0] = np.NaN
                    pt_arr[ib == 0] = np.NaN
                    # update the range min and max
                    mlt_vmin = min(mlt_vmin, np.nanmin(pr_arr), np.nanmin(pt_arr))
                    mlt_vmax = max(mlt_vmax, np.nanmax(pr_arr), np.nanmax(pt_arr))
                    # store in container
                    pr_arrs.append(pr_arr)
                    pt_arrs.append(pt_arr)
                    mlt_labels.append(mlt_label)

                # for each multiplier array (prior and posterior)
                for i, (pr_arr, pt_arr, mlt_label) in enumerate(zip(pr_arrs, pt_arrs, mlt_labels)):
                    # get the prior and posterior axes
                    pr_ax, pt_ax = axes[i + 1, 0], axes[
                        i + 1, 1]  # fig.add_subplot(gs[i + 1, 0]), fig.add_subplot(gs[i + 1, 1])
                    # label them
                    pr_ax.set_title("{0}) prior {1} multiplier".format(abet[ax_c], mlt_label), loc="left")
                    ax_c += 1
                    pt_ax.set_title("{0}) posterior {1} multiplier".format(abet[ax_c], mlt_label), loc="left")
                    ax_c += 1

                    # plot the prior multiplier array and add a colorbar
                    c = pr_ax.imshow(pr_arr, vmin=mlt_vmin, vmax=mlt_vmax)
                    divider = make_axes_locatable(pr_ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    c = plt.colorbar(c, cax=cax)
                    c.set_label("$log_{10}$")

                    # plot the posterior multiplier array and add a colorbar
                    c = pt_ax.imshow(pt_arr, vmin=mlt_vmin, vmax=mlt_vmax)
                    divider = make_axes_locatable(pt_ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    c = plt.colorbar(c, cax=cax)
                    c.set_label("$log_{10}$")
                    pr_ax.set_xticks([])
                    pr_ax.set_yticks([])
                    pt_ax.set_xticks([])
                    pt_ax.set_yticks([])


                # now plot the resulting prior and posterior multiplied model input arrays
                # and add colobars
                pr_ax = axes[-1, 0]
                pt_ax = axes[-1, 1]
                c = pr_ax.imshow(pr_mi_arr, vmin=mi_vmin, vmax=mi_vmax)
                divider = make_axes_locatable(pr_ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                c = plt.colorbar(c, cax=cax)
                c.set_label(cb_label)
                c = pt_ax.imshow(pt_mi_arr, vmin=mi_vmin, vmax=mi_vmax)
                divider = make_axes_locatable(pt_ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                c = plt.colorbar(c, cax=cax)
                c.set_label(cb_label)
                # label these axes with the resulting phi values
                pr_ax.set_title("{0}) prior {1} model input\n$\\phi$: {3:1.2E}".\
                                format(abet[ax_c], prop_label, real, pr_pv.loc[real]), loc="left")
                ax_c += 1
                pt_ax.set_title("{0}) posterior {1} model input\n$\\phi$: {3:1.2E}".\
                                format(abet[ax_c], prop_label, real, pt_pv.loc[real]), loc="left")
                pr_ax.set_xticks([])
                pr_ax.set_yticks([])
                pt_ax.set_xticks([])
                pt_ax.set_yticks([])
                plt.tight_layout()
                axes[0, 1].axis("off")
                # plt.text(0.7, 0.97, "Realization {0} {1}".format(real,prop_label), fontsize=12, transform=plt.gcf().transFigure,ha="center")
                pdf.savefig()
                plt.close(fig)
                #break

            ireal += 1
            # only plot the first few b/c this is so slow!
            if ireal > 3:
                break

    os.chdir("..")


def plot_phi_hists(post_iter):
    """plot the history and scenario phi histograms for the 4 locations
    of management interest

    Args:
        post_iter (int): the PESTPP-IES iteration to use as the posterior

    """

    # the master history PESTPP-IES directory
    m_d = "master_history_reweight"

    # load the control file, prior ensemble and posterior ensembles
    pst = pyemu.Pst(os.path.join(m_d, "eaa_ver_reweight.pst"))
    pr_en = pyemu.ObservationEnsemble.from_binary(pst=pst,
        filename=os.path.join(m_d, "eaa_ver_reweight.0.obs.jcb"))
    pt_en = pyemu.ObservationEnsemble.from_binary(pst=pst, 
        filename=os.path.join(m_d, "eaa_ver_reweight.{0}.obs.jcb".format(post_iter)))



    fig, axes = plt.subplots(2, 4, figsize=(8, 5.0))
    org_obs = pst.observation_data.copy()
    bins = 20
    # for each of the four locations
    dfs = {}
    for i, (site, label) in enumerate(zip(["comal", "sanmar", str(j17_id), str(j27_id)],
                                          ["Comal", "San Marcos", "J-17", "J-27"])):

        pst.observation_data = org_obs.copy()
        obs = pst.observation_data
        # get the observation data for just this site
        site_obs = obs.loc[obs.obgnme.apply(lambda x: site in x), :].copy()
        # reset all obs to zero weight
        obs.loc[:, "weight"] = 0.0
        # set weight for just this site to 1.0 (so unweighted sum of squares)
        obs.loc[site_obs.obsnme, "weight"] = 1.0
        # update the control file attribute for the ensembles for phi calcs
        pr_en.pst = pst
        pt_en.pst = pst

        ax = axes[0, i]

        # log transform the phi values since they appear to be (atleast) log distributed
        pr_pv = pr_en.phi_vector.apply(np.log10)
        pt_pv = pt_en.phi_vector.apply(np.log10)
        dfs["History "+ label] = pt_pv
        # plot the prior and posterior historgrams
        ax.hist(pr_pv, bins=bins, facecolor="0.5", alpha=0.5, edgecolor="none")
        ax.hist(pt_pv, bins=bins, facecolor="b", alpha=0.5, edgecolor="none")

        # plot the "existing" phi (from the original models) as a
        # vertical line for reference
        ylim = ax.get_ylim()
        pr_bval = pr_pv["base"]
        ax.plot([pr_bval, pr_bval], ylim, color="0.5", ls="--")
        ax.set_title("{0}) History {1}".format(abet[i], label), loc="left")


    # now for the scenario phi values - the prior and posterior scenario output ensembles
    # are in two different directories!

    m_d = "master_scenario_posterior"
    pst = pyemu.Pst(os.path.join(m_d, "eaa_pred.pst"))
    # load the prior scenario output ensemble
    pr_en = pyemu.ObservationEnsemble.from_binary(pst=pst,
                                                  filename=os.path.join("master_scenario_prior", "eaa_pred.0.obs.jcb"))
    # load the posterior output ensemble
    pt_en = pyemu.ObservationEnsemble.from_binary(pst=pst,
                                                  filename=os.path.join(m_d, "eaa_pred.0.obs.jcb"))

    org_obs = pst.observation_data.copy()
    # for each site of management interest
    for i, (site, label) in enumerate(zip(["comal", "sanmar", 'j-17', "j-27"],
                                          ["Comal", "San Marcos", "J-17", "J-27"])):
        pst.observation_data = org_obs.copy()
        obs = pst.observation_data
        # obs data for just this site
        site_obs = obs.loc[obs.obgnme.apply(lambda x: site in x), :].copy()
        # set all obs weights to zero
        obs.loc[:, "weight"] = 0.0
        # set the weights for this site to 1.0 (unweighted)
        obs.loc[site_obs.obsnme, "weight"] = 1.0
        # update the control file instance for both ensembles
        pr_en.pst = pst
        pt_en.pst = pst

        ax = axes[1, i]

        # log transform the phi vectors as before
        pr_pv = pr_en.phi_vector.apply(np.log10)
        pt_pv = pt_en.phi_vector.apply(np.log10)
        dfs["Scenario " + label] = pt_pv
        # mask any resulting phi values that are greater than 10^35 (meaning an invalid value
        # was yielded by the simulation - didnt have this problem in the history ensemble because
        # of the reweighting process)
        pr_pv.loc[pr_pv>=35] = np.NaN
        pt_pv.loc[pt_pv >= 35] = np.NaN

        # plot the prior and posterior phi histograms
        ax.hist(pr_pv, bins=bins, facecolor="0.5", alpha=0.5, edgecolor="none",normed=True)
        ax.hist(pt_pv, bins=bins, facecolor="b", alpha=0.5, edgecolor="none",normed=True)

        ylim = ax.get_ylim()
        pr_bval = pr_pv["base"]
        ax.plot([pr_bval, pr_bval], ylim, color="0.5", ls="--")
        ax.set_title("{0}) Scenario {1}".format(abet[i + 4], label), loc="left")

    for ax in axes.flatten():
        # ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("$log_{10} \phi$")

    plt.tight_layout()
    plt.savefig(os.path.join(m_d, "phi_hists.pdf"))
    df = pd.DataFrame(dfs)
    df.to_csv("posterior_phis.csv")

def plot_par_change(post_iter):
    """plot prior to posterior parameter first (mean) and second (variance) moment changes
    for the supplementary material

    Args:
        post_iter (int): the PESTPP-IES iteration to use as the posterior

    """

    # the PESTPP-IES history master directory
    m_d = "master_history_reweight"

    # load the control file
    pst = pyemu.Pst(os.path.join(m_d, "eaa_ver_reweight.pst"))

    # regroup the time-varying recharge zone and constant (global) parameters into two
    # groups to limit the number of axes plots
    par = pst.parameter_data
    par.loc[par.pargp.apply(lambda x: "zn_rech" in x), "pargp"] = "zn_rech"
    par.loc[par.pargp.apply(lambda x: "cn_rech" in x), "pargp"] = "cn_rech"

    # load the prior and posterior parameter ensembles
    pr_en = pyemu.ParameterEnsemble.from_binary(pst=pst,
                                                filename=os.path.join(m_d, "eaa_ver_reweight.0.par.jcb"))
    pt_en = pyemu.ParameterEnsemble.from_binary(pst=pst,
                        filename=os.path.join(m_d, "eaa_ver_reweight.{0}.par.jcb".format(post_iter)))
    # plot par change summaries
    pyemu.plot_utils.ensemble_change_summary(pr_en, pt_en, pst=pst, filename=os.path.join(m_d, "par_change.pdf"))


def write_par_summary_table():
    """write a latex parameter summary table for the supplementary
    material

    """

    # a function to rename the parameter groups with user-friendly names
    def grp_namer(grp):
        name_dict = {"rech": "recharge", "hk": "hydraulic conductivity",
                     "ss": "specific storage", "sy": "specific yield",
                     "hfb": "hydraulic flow barrier conductance",
                     "welflux": " well extraction",
                     "strt": "initial conditions",
                     "drn": "drain conductance"
                     }
        type_dict = {"zn": "zone", "gr": "grid-scale", "pp": "pilot point", "cn": "global"}

        name_tag = None
        for tag, name in name_dict.items():
            if tag in grp:
                name_tag = name
                break
        if name_tag is None:
            raise Exception(grp)

        type_tag = ""
        for tag, name in type_dict.items():
            if tag in grp:
                type_tag = name
                break
        time_tag = ""
        if "scen" in grp:
            time_tag = "scenario"
        elif "hist" in grp:
            time_tag = "history"

        name = time_tag + " " + type_tag + " " + name_tag
        if grp == "welflux_k00":
            name = time_tag + " spatial well extraction"
        elif "recharge" in name or "wel" in name:
            name += "(per stress period)"
        return name


    # load the history and scenario control files
    h_pst = pyemu.Pst(os.path.join(h_dir, "eaa_ver.pst"))
    s_pst = pyemu.Pst(os.path.join(s_dir, "eaa_pred.pst"))

    # rename the time varying history parameters with a tag for "h"istory
    par = h_pst.parameter_data
    h_t_pars = par.loc[par.pargp.apply(lambda x: "wel" in x or "rech" in x or "strt" in x), "parnme"]
    par.loc[h_t_pars, "pargp"] = "hist" + par.loc[h_t_pars, "pargp"]
    par.loc[h_t_pars, "parnme"] = "hist" + par.loc[h_t_pars, "parnme"]

    # rename the time varying scenario parameters with a tag for "s"cenario
    par = s_pst.parameter_data
    s_t_pars = par.loc[par.pargp.apply(lambda x: "wel" in x or "rech" in x or "strt" in x), "parnme"]
    par.loc[s_t_pars, "pargp"] = "scen" + par.loc[s_t_pars, "pargp"]
    par.loc[s_t_pars, "parnme"] = "scen" + par.loc[s_t_pars, "parnme"]

    # update the scenario parameter data to have the time-varying history parameters
    print(s_pst.npar)
    s_pst.parameter_data = par.append(h_pst.parameter_data.loc[h_t_pars, :])
    print(s_pst.npar)

    # get user-friendly parameter group names
    s_pst.parameter_data.loc[:, "pargp"] = s_pst.parameter_data.pargp.apply(lambda x: grp_namer(x))
    print(s_pst.par_groups)
    # save the summary table
    s_pst.write_par_summary_table(filename="eaa_par_sum.tex", sigma_range=6)


def plot_parallel(post_iter):
    """function to do the post-process plotting in parallel
    using multiprocessing to save wall time

    Args:
        post_iter (int): the PESTPP-IES iteration to use as the posterior

    """
    procs = []

    #plot_obs_vs_sim_case("master_history_prior", "eaa_ver")
    p = mp.Process(target=plot_obs_vs_sim_case,args=["master_history_prior","eaa_ver"])
    procs.append(p)

    #plot_obs_vs_sim_case("master_scenario_prior", "eaa_pred")
    p = mp.Process(target=plot_obs_vs_sim_case, args=["master_scenario_prior", "eaa_pred"])
    procs.append(p)

    #plot_obs_vs_sim_case(m_d, "eaa_ver_reweight", post_iter=3,
    #                     focus=False,plt_name="obs_vs_sim_all.pdf")
    p = mp.Process(target=plot_obs_vs_sim_case, args=["master_history_reweight", "eaa_ver_reweight"],
                   kwargs={"post_iter":post_iter,"focus":False,"plt_name":"hist_obs_v_sim_all.pdf"})
    procs.append(p)

    #plot_obs_vs_sim_case(m_d, "eaa_ver_reweight", post_iter=3, focus=True)
    p = mp.Process(target=plot_obs_vs_sim_case, args=["master_history_reweight", "eaa_ver_reweight"],
                   kwargs={"post_iter": post_iter, "focus": True,"plt_name":"hist_obs_v_sim.pdf"})
    procs.append(p)

    #plot_par_change()
    p = mp.Process(target=plot_par_change,kwargs={"post_iter":post_iter})
    procs.append(p)

    #plot_array_pars()
    p = mp.Process(target=plot_array_pars,kwargs={"post_iter":post_iter})
    procs.append(p)

    #plot_phi_hists()
    p = mp.Process(target=plot_phi_hists,kwargs={"post_iter":post_iter})
    procs.append(p)

    pr_en_file = os.path.join("master_scenario_prior", "eaa_pred.0.obs.jcb")
    pt_en_file = os.path.join("master_scenario_posterior", "eaa_pred.0.obs.jcb")
    pst_file = os.path.join("master_scenario_posterior", "eaa_pred.pst")
    plt_name = os.path.join("master_scenario_posterior", "months_below.pdf")
    #plot_below(pst_file=pst_file, pr_en_file=pr_en_file,
    #           pt_en_file=pt_en_file, plt_name=plt_name)
    kwargs = {"pst_file":pst_file,"pr_en_file":pr_en_file,
              "pt_en_file":pt_en_file,"plt_name":plt_name}
    p = mp.Process(target=plot_below,kwargs=kwargs)
    procs.append(p)



    for p in procs:
        p.start()

    pst = pyemu.Pst(os.path.join("master_scenario_posterior", "eaa_pred.pst"))
    pr_en = pyemu.ObservationEnsemble.from_binary(pst=pst,
                                                  filename=os.path.join("master_scenario_prior", "eaa_pred.0.obs.jcb"))
    pt_en = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=os.path.join("master_scenario_posterior",
                                                                                 "eaa_pred.0.obs.jcb"))
    base_en = pyemu.ObservationEnsemble.from_binary(pst=pst, filename=os.path.join("master_scenario_prior",
                                                                                   "eaa_pred.base.obs.jcb"))
    args = [pst, s_start_datetime, s_end_datetime]

    kwargs = {"pr_en": pr_en, "pt_en": pt_en, "base_en": base_en, "focus": True}
    kwargs["plt_name"] = os.path.join("master_scenario_posterior","scen_obs_v_sim.pdf")
    plot_obs_vs_sim(*args,**kwargs)

    kwargs["plt_name"] = os.path.join("master_scenario_posterior", "scen_obs_v_sim_all.pdf")
    kwargs["focus"] = False
    plot_obs_vs_sim(*args, **kwargs)

    for p in procs:
        p.join()



def soup_to_nuts(noptmax=2):
    """very high level function that implements the entire analysis

    Args:
        noptmax (int): the number PESTPP-IES DA iterations to undertake and
            also the iteration to use as the posterior. Default is 2

    """
    # prepare both simulations and set up pest interfaces
    setup_models_parallel()

    # write the parameter summary table for SI
    write_par_summary_table()

    # run history prior monte carlo and plot obs vs sim
    m_d = "master_history_prior"
    _prep_for_parallel(h_dir, "eaa_ver.pst", m_d=m_d, noptmax=-1)
    run_condor("eaa_ver.pst",m_d)
    
    # run scenario prior monte carlo and plot obs vs sim
    m_d = "master_scenario_prior"
    _prep_for_parallel(s_dir, "eaa_pred.pst", m_d=m_d, noptmax=-1)
    run_condor("eaa_pred.pst",m_d)
    
    # reweight and PDC resolution
    reweight_ensemble("master_history_prior",h_dir,case="eaa_ver")

    # run DA iterations and plot
    build_temporal_localizer(h_dir,save=True)
    m_d = "master_history_reweight"
    _prep_for_parallel(h_dir,"eaa_ver_reweight.pst",m_d=m_d,noptmax=noptmax,overdue_giveup_fac=3,with_loc=True)
    run_condor("eaa_ver_reweight.pst",m_d)
    
    # transfer adjusted parameters from history to scenario
    transfer_hist_pars_to_scenario(os.path.join("master_history_reweight"
                                                ,"eaa_ver_reweight.{0}.par.jcb".format(noptmax)),
                                    os.path.join(s_dir,"prior.jcb"))
    m_d = "master_scenario_posterior"
    _prep_for_parallel(s_dir,"eaa_pred.pst",m_d=m_d,noptmax=-1)
    run_condor("eaa_pred.pst", m_d)

    plot_parallel(noptmax)



def write_site_map_shapefiles():
    """write a grid shapefile with the ibound array and a shapefile of HFB cell locations
    for use in the site map

    """
    import shapefile
    m = flopy.modflow.Modflow.load(h_nam_file,model_ws=h_dir,
                                   load_only=["dis","bas6","hfb6"],
                                   forgive=False)
    m.bas6.ibound.export("ibound.shp")
    df = pd.DataFrame.from_records(m.HFB6.hfb_data)

    xcenter = m.modelgrid.xcellcenters
    ycenter = m.modelgrid.ycellcenters
    df.loc[:,"x1"] = df.apply(lambda x: xcenter[int(x.irow1),int(x.icol1)],axis=1)
    df.loc[:, "y1"] = df.apply(lambda x: ycenter[int(x.irow1), int(x.icol1)], axis=1)
    df.loc[:, "x2"] = df.apply(lambda x: xcenter[int(x.irow2), int(x.icol2)], axis=1)
    df.loc[:, "y2"] = df.apply(lambda x: ycenter[int(x.irow2), int(x.icol2)], axis=1)

    w = shapefile.Writer(target="hfb.shp",shapeType=shapefile.POLYLINE)
    for col in df.columns:
        w.field(col,fieldType='N',decimal=2)
    df.apply(lambda x: w.line([[[x.x1,x.y1],[x.x2,x.y2]]]),axis=1)
    df.apply(lambda x: w.record(*x),axis=1)
    

if __name__ == "__main__":

    #soup_to_nuts(3)
    #hfb_shapefile()
    #setup_models_parallel()
    print("test")
