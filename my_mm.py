import datetime
from numpy import empty,reshape
from datetime import timedelta
from metroman.MetroManVariables import Domain,Observations
import netCDF4 as nc
import geopandas as gpd
import numpy as np


def readData(reach_ids_str,shapefile,priorfile,swordfile):
    reach_ids = reach_ids_str.split(',')
    reach_ids = list(map(int,reach_ids))
    DAll = Domain()
    DAll.nR = len(reach_ids)
    gdf = gpd.read_file(shapefile)
    selected = gdf[['reach_id','slope2','wse','width','time_tai','time_str']]
    #get the times first

    obs_dict = {}
    unique_times = set()
    epoch = datetime.datetime(2000,1,1,0,0,0)
    # round the time to the precision of hours
    for reach_id in reach_ids:
        reach_data = selected.loc[selected['reach_id'] == str(reach_id)]
        
        for _,row in reach_data.iterrows():
            if row['time_tai'] > 0:
                t = epoch + timedelta(seconds=row['time_tai'])
                t_hour = (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)+timedelta(hours=t.minute//30) - epoch).total_seconds()

                unique_times.add(t_hour)
    overlap_ts = sorted(unique_times)
    
    nt = len(overlap_ts)
    DAll.nt = nt
    tall = [ epoch + datetime.timedelta(seconds=t) for t in overlap_ts ]


    talli=empty(DAll.nt)
    for i in range(DAll.nt):
        dt=(tall[i]-tall[0])
        talli[i]=dt.days + dt.seconds/86400.
        
    DAll.t=reshape(talli,[1,DAll.nt])

    #get all observations
    AllObs = Observations(DAll)
    for index, reach_id in enumerate(reach_ids):
        reach_data = selected.loc[selected['reach_id'] == str(reach_id)]
        obs_dict = {}
        # initialize the dictionary with missing data values
        for t in overlap_ts:
            obs_dict[t] = {'slope':None,'wse': None,'width':None,'time_str': None}
        
        # fill the dictionary with data read from the shapfile
        for _,row in reach_data.iterrows():
            if row['time_tai'] <= 0: continue
            t = epoch + timedelta(seconds=row['time_tai'])
            t_hour = (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)+timedelta(hours=t.minute//30) - epoch).total_seconds()
            obs_dict[t_hour] = {'slope':row['slope2'],'wse': row['wse'],'width':row['width'],'time_str': row['time_str']}
        # reach swot observation data
        sorted_dict = dict(sorted(obs_dict.items()))
        # use a shapefile instead of the netcdf data
        # get times 
        for i,key in enumerate(sorted_dict.keys()):
            AllObs.S[index,i] = sorted_dict[key]['slope']
            AllObs.h[index,i] = sorted_dict[key]['wse']
            AllObs.w[index,i] = sorted_dict[key]['width']



    if True:   
        AllObs.sigS=1.7e-5
        AllObs.sigh=0.1
        AllObs.sigw=10
    else:
        AllObs=0.
        
        
    print('Reading priors')
    priors = nc.Dataset(priorfile, format="NETCDF4") 
    reaches  = priors['reaches']['reach_id'][:]
    q_bars = priors['model/mean_q'][:]
    sword_dataset=nc.Dataset(swordfile)
    swordreachids=sword_dataset["reaches/reach_id"][:]
    

    QBar=empty(DAll.nR)
    reach_length=empty(DAll.nR)
    dist_out=empty(DAll.nR)
    for index, reach_id in enumerate(reach_ids):
        reach_index = np.where(reaches == reach_id)
        if reach_index[0].size > 0:
            QBar[index] = q_bars[reach_index]
        else:
             QBar[index] = -999999999999
        k=np.argwhere(swordreachids == reach_id)
        print(reach_id,k)

        reach_lengths=sword_dataset["reaches/reach_length"][:]
        reach_length[index]=reach_lengths[k]

        dist_outs=sword_dataset["reaches/dist_out"][:]
        dist_out[index]=dist_outs[k]
    DAll.L=reach_length
    DAll.xkm=np.max(dist_out)-dist_out + DAll.L[0]/2 #reach midpoint distance downstream [m]
    DAll.t=reshape(talli,[1,DAll.nt])

    # Reshape observations
    AllObs.hv=reshape(AllObs.h, (DAll.nR*DAll.nt,1))
    AllObs.Sv=reshape(AllObs.S, (DAll.nR*DAll.nt,1))
    AllObs.wv=reshape(AllObs.w, (DAll.nR*DAll.nt,1))
    return AllObs,DAll,QBar,tall,reach_ids

def main():
    AllObs,DAll,QBar,tall,reach_ids = readData(reach_ids_str='73282400081,73282400061,73282400051,73282400041,73282400031',shapefile=r'D:\tmp\unzipped\m1_m3_m7.shp',
             priorfile=r'D:\git\podaac_tutorials\notebooks\datasets\data_downloads\na_sword_v16_SOS_unconstrained_0001_20240726T123358_priors.nc',
             swordfile=r'D:\workspace\SWOT-Confluence\offline-discharge-data-product-creation\data\input\sword\SWORD_v16_netcdf\netcdf\na_sword_v16.nc')
    print(AllObs.h)
    print(QBar)
    print(tall)
    print(reach_ids)

if __name__ == '__main__':
    main()