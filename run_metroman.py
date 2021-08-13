"""Program that runs MetroMan on sets of reaches and writes a NetCDF file that
contains A0, n, and Q time series.
"""

# Standard imports
import json
import os
from pathlib import Path
import sys

# Third party imports
from netCDF4 import Dataset
from numpy import linspace,reshape,diff,ones,array,empty,mean,zeros,putmask
import numpy as np

# Application imports
from metroman.CalcdA import CalcdA
from metroman.CalculateEstimates import CalculateEstimates
from metroman.GetCovMats import GetCovMats
from metroman.MetroManVariables import Domain,Observations,Chain,RandomSeeds,Experiment,Prior,Estimates
from metroman.MetropolisCalculations import MetropolisCalculations
from metroman.ProcessPrior import ProcessPrior
from metroman.SelObs import SelObs

def get_reachids(reachjson):
    """Extract and return a list of reach identifiers from json file.
    
    Parameters
    ----------
    reachjson : str
        Path to the file that contains the list of reaches to process
    
        
    Returns
    -------
    list
        List of reaches identifiers
    """

    #index = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX"))
    index=2
    with open(reachjson) as jsonfile:
        data = json.load(jsonfile)
    return data[index]

def get_domain_obs(nr):
    """Define and return domain and observations as a tuple."""

    DAll=Domain()
    DAll.nR=nr #number of reaches
    xkm=array([25e3, 50e3, 75e3, 25e3, 50e3, 75e3])
    DAll.xkm=xkm[0:nr] #reach midpoint distance downstream [m]
    L=array([50e3, 50e3, 50e3, 50e3, 50e3, 50e3])
    DAll.L=L[0:nr]  #reach lengths, [m]
    DAll.nt=25 #number of overpasses

    AllObs=Observations(DAll)
    AllObs.sigS=1.7e-5
    AllObs.sigh=0.1
    AllObs.sigw=10

    return DAll, AllObs

def retrieve_obs(reachlist, inputdir, DAll, AllObs):
    """ Retrieves data from SWOT and SoS files, populates observation object and
    returns qbar."""

    Qbar=empty([DAll.nR])
    i=0
    BadIS=False
    for reach in reachlist:
        swotfile=inputdir.joinpath('swot', reach["swot"])
        swot_dataset = Dataset(swotfile)
        AllObs.h[i,:]=swot_dataset["reach/wse"][0:DAll.nt].filled(np.nan)
        AllObs.w[i,:]=swot_dataset["reach/width"][0:DAll.nt].filled(np.nan)
        AllObs.S[i,:]=swot_dataset["reach/slope2"][0:DAll.nt].filled(np.nan)
        swot_dataset.close()

        nbad=np.count_nonzero(np.isnan(AllObs.h[0,:]))
        if DAll.nt-nbad < 6: #note: 6 is typically minimum needed observations for metroman 
            BadIS=True
        else:
             sosfile=inputdir.joinpath('sos', reach["sos"])
             sos_dataset=Dataset(sosfile)
             
             sosreachids=sos_dataset["reaches/reach_id"][:]
             sosQbars=sos_dataset["reaches/mean_q"][:]
             k=np.argwhere(sosreachids == reach["reach_id"])
     
             Qbar[i]=sosQbars[k]

             sos_dataset.close()

        i += 1


    #select observations that are NOT equal to the fill value
    iDelete=np.where(np.isnan(AllObs.h[0,:]))
    shape_iDelete=np.shape(iDelete)
    nDelete=shape_iDelete[1]
    AllObs.h=np.delete(AllObs.h,iDelete,1)
    AllObs.w=np.delete(AllObs.w,iDelete,1)
    AllObs.S=np.delete(AllObs.S,iDelete,1)

    DAll.nt -= nDelete

    DAll.t=reshape(linspace(1,DAll.nt,num=DAll.nt),[1,DAll.nt]) #time, [days]
    DAll.dt= reshape(diff(DAll.t).T*86400 * ones((1,DAll.nR)),(DAll.nR*(DAll.nt-1),1))

    # Reshape observations
    AllObs.hv=reshape(AllObs.h, (DAll.nR*DAll.nt,1))
    AllObs.Sv=reshape(AllObs.S, (DAll.nR*DAll.nt,1))
    AllObs.wv=reshape(AllObs.w, (DAll.nR*DAll.nt,1))
    return Qbar,iDelete,nDelete,BadIS

def set_up_experiment(DAll, Qbar):
    """Define and set parameters for experiment and return a tuple of 
    Chain, Random Seed, Expriment and Prior."""

    C=Chain()
    C.N=10000
    C.Nburn=2000
    R=RandomSeeds()
    R.Seed=9

    Exp=Experiment()
    Exp.tUse=array([1,	31])
    Exp.nOpt=5

    P=Prior(DAll)
    P.meanQbar=mean(Qbar)
    P.covQbar=0.5
    P.eQm=0.
    P.Geomorph.Use=False
    # this is for Laterals=false
    P.AllLats.q=zeros((DAll.nR,DAll.nt))
    return C, R, Exp, P

def process(DAll, AllObs, Exp, P, R, C):
    """Process observations and priors and return an estimate."""
    
    D,Obs,AllObs,DAll,Truth,Prior.Lats.q=SelObs(DAll,AllObs,Exp,[],Prior.AllLats)
    Prior.Lats.qv=reshape(Prior.Lats.q,(D.nR*(D.nt-1),1))
    Obs=CalcdA(D,Obs)
    AllObs=CalcdA(DAll,AllObs)
    ShowFigs=False
    DebugMode=True

    Smin=1.7e-5
    Obs.S[Obs.S<Smin]=putmask(Obs.S,Obs.S<Smin,Smin) #limit slopes to a minimum value
    AllObs.S[AllObs.S<Smin]=putmask(AllObs.S,AllObs.S<Smin,Smin)

    P,jmp=ProcessPrior(P,AllObs,DAll,Obs,D,ShowFigs,Exp,R,DebugMode)
    Obs,P2=GetCovMats(D,Obs,Prior)

    C=MetropolisCalculations(P,D,Obs,jmp,C,R,DAll,AllObs,Exp.nOpt,DebugMode)
    Estimate,C=CalculateEstimates(C,D,Obs,P,DAll,AllObs,Exp.nOpt) 
    return Estimate

def write_output(outputdir, reachids, Estimate, iDelete, nDelete, BadIS):
    """Write data from MetroMan run to NetCDF file in output directory."""

    fillvalue = -999999999999

    #add back in placeholders for removed data
    iInsert=iDelete-np.arange(nDelete)
    iInsert=reshape(iInsert,[nDelete,])

    Estimate.AllQ=np.insert(Estimate.AllQ,iInsert,fillvalue,1)
    Estimate.QhatUnc_HatAllAll=np.insert(Estimate.QhatUnc_HatAllAll,iInsert,fillvalue,1)

    setid = '-'.join(reachids) + "_metroman.nc"
    outfile = outputdir.joinpath(setid)
    dataset = Dataset(outfile, 'w', format="NETCDF4")
    dataset.set_id = setid    # TODO decide on how to identify sets
    dataset.valid =  1 if not BadIS else 0   # TODO decide what's valid if applicable
    dataset.createDimension("nr", len(reachids))
    dataset.createDimension("nt", len(Estimate.AllQ[0]))

    nr = dataset.createVariable("nr", "i4", ("nr",))
    nr.units = "reach"
    nr[:] = range(1, len(Estimate.A0hat) + 1)

    nt = dataset.createVariable("nt", "i4", ("nt",))
    nt.units = "time steps"
    nt[:] = range(len(Estimate.AllQ[0]))

    reach_id = dataset.createVariable("reach_id", "i8", ("nr",))
    reach_id[:] = np.array(reachids, dtype=int)

    A0 = dataset.createVariable("A0hat", "f8", ("nr",), fill_value=fillvalue)
    A0[:] = Estimate.A0hat

    na = dataset.createVariable("nahat", "f8", ("nr",), fill_value=fillvalue)
    na[:] = Estimate.nahat
    
    x1 = dataset.createVariable("x1hat", "f8", ("nr",), fill_value=fillvalue)
    x1[:] = Estimate.x1hat

    allq = dataset.createVariable("allq", "f8", ("nr", "nt"), fill_value=fillvalue)
    allq[:] = Estimate.AllQ

    qu = dataset.createVariable("q_u", "f8", ("nr", "nt"), fill_value=fillvalue)
    qu[:] = Estimate.QhatUnc_HatAllAll

    dataset.close()

def main():
#    inputdir = Path("/mnt/data/input")
#    outputdir = Path("/mnt/data/output")
    inputdir = Path("/Users/mtd/OneDrive - The Ohio State University/Analysis/SWOT/Discharge/Confluence/metroman_rundir")
    outputdir = Path("/Users/mtd/OneDrive - The Ohio State University/Analysis/SWOT/Discharge/Confluence/metroman_outdir")
    try:
        reachjson = inputdir.joinpath(sys.argv[1])
    except IndexError:
        reachjson = inputdir.joinpath("sets.json") 

    reachlist = get_reachids(reachjson)

    DAll, AllObs = get_domain_obs(len(reachlist))

    Qbar,iDelete,nDelete,BadIS = retrieve_obs(reachlist, inputdir, DAll, AllObs)

    if BadIS:
        fillvalue=-999999999999
	    #define and write fill value data
        print("fewer than minimum number of swot passes. not running metroman for this set.")
        Estimate=Estimates(DAll,DAll)
        Estimate.nahat=np.full([DAll.nR],fillvalue)
        Estimate.x1hat=np.full([DAll.nR],fillvalue)
        Estimate.QhatUnc_HatAllAll=np.full([DAll.nR,DAll.nt],fillvalue) 
    else:
    	C, R, Exp, P = set_up_experiment(DAll, Qbar)
    	Estimate = process(DAll, AllObs, Exp, P, R, C)
    
    reachids = [ str(e["reach_id"]) for e in reachlist ]
    write_output(outputdir, reachids, Estimate,iDelete,nDelete,BadIS)

if __name__ == "__main__":
   main()    
