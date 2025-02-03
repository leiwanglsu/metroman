import wx
import streamlit as st
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from metroman.MetroManVariables import Domain,Observations,Chain,RandomSeeds,Experiment,Prior,Estimates
from my_mm import readData
            
## i/o for streamlit
def file_selector_wx():
    app = wx.App(False)  # Create a wx application instance

    dialog = wx.FileDialog(
        None, 
        message="Choose a file", 
        wildcard="All files (*.*)|*.*", 
        style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
    )
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None
    dialog.Destroy()
    return path
import pickle,os
def save_session_state(file_path):
    with open(file_path, 'wb') as f:
            pickle.dump(st.session_state.to_dict(), f)


def load_session_state(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
                loaded_state = pickle.load(f)
                for k, v in loaded_state.items():
                    st.session_state[k] = v
    else:
        print("File not found.")
## Compute MM
from numpy import linspace,reshape,diff,ones,array,empty,mean,zeros,putmask
import numpy as np
def set_up_experiment(DAll, Qbar):
    """Define and set parameters for experiment and return a tuple of 
    Chain, Random Seed, Expriment and Prior."""

    C=Chain()
    C.N=10000 #according to the paper, it should be 100000
    C.Nburn=2000 #according to the paper, it should be 20000
    R=RandomSeeds()
    R.Seed=9

    Exp=Experiment()

    tUseMax=min(31,DAll.nt)

    #Exp.tUse=array([1,	31])
    Exp.tUse=array([1,	tUseMax-1])

    Exp.nOpt=5

    P=Prior(DAll)
    P.meanQbar=mean(Qbar)
    P.covQbar=0.5
    P.eQm=0.
    P.Geomorph.Use=False
    # this is for Laterals=false
    P.AllLats.q=zeros((DAll.nR,DAll.nt))
    return C, R, Exp, P
from metroman.SelObs import SelObs
from metroman.CalcdA import CalcdA
from metroman.ProcessPrior import ProcessPrior
from metroman.GetCovMats import GetCovMats
from metroman.MetropolisCalculations import MetropolisCalculations
from metroman.CalculateEstimates import CalculateEstimates
from run_metroman import process

# def process(DAll, AllObs, Exp, P, R, C, Verbose):
#     """Process observations and priors and return an estimate."""
    
#     D,Obs,AllObs,DAll,Truth,Prior.Lats.q=SelObs(DAll,AllObs,Exp,[],Prior.AllLats)

    
    
    
#     Prior.Lats.qv=reshape(Prior.Lats.q,(D.nR*(D.nt-1),1))
#     Obs=CalcdA(D,Obs)
#     AllObs=CalcdA(DAll,AllObs)
#     ShowFigs=False
#     DebugMode=False

#     Smin=1.7e-5
#     #Smin=5e-5
#     Obs.S[Obs.S<Smin]=putmask(Obs.S,Obs.S<Smin,Smin) #limit slopes to a minimum value
#     AllObs.S[AllObs.S<Smin]=putmask(AllObs.S,AllObs.S<Smin,Smin)

#     P,jmp=ProcessPrior(P,AllObs,DAll,Obs,D,ShowFigs,Exp,R,DebugMode,Verbose)
#     Obs,P2=GetCovMats(D,Obs,Prior)

#     C=MetropolisCalculations(P,D,Obs,jmp,C,R,DAll,AllObs,Exp.nOpt,DebugMode,Verbose)
#     Estimate,C=CalculateEstimates(C,D,Obs,P,DAll,AllObs,Exp.nOpt) 
#     return Estimate

## Disdplay

def displayMessages():
    st.write(f"The shape file is {st.session_state['shapeFile']}")
    st.write(f"The Prior file is {st.session_state['priorFile']}")
    st.write(f"The SWORD file is {st.session_state['swordFile']}")
    st.write(f"Reach ids: {st.session_state['reaches']}")
    if 'reach_coor' in st.session_state.keys():
        st.write(f"Reach coordinates:{st.session_state['reach_coor']}")
## Export

## Interface
def init():
    # Streamlit app

    st.title("SWOT discharge MetroMan")
    
    if not 'shapeFile' in st.session_state.keys():
        st.session_state['shapeFile'] = 'No file'
    if not 'priorFile' in st.session_state.keys():
        st.session_state['priorFile'] = 'No file'
    if not 'swordFile' in st.session_state.keys():
        st.session_state['swordFile'] = 'No file'
    if not 'reaches' in st.session_state.keys():
        st.session_state['reaches'] = None   
    if not 'river' in st.session_state.keys():
        st.session_state['river'] = 'No river'   
    if not 'reach' in st.session_state.keys():
        st.session_state['reach'] = None 
    if not 'AllQ' in st.session_state.keys():
        st.session_state['AllQ'] = None
    if not 'wse' in st.session_state.keys():
        st.session_state['wse'] = None    
    if not 'width' in st.session_state.keys():
        st.session_state['width'] = None   
    if not 'slope' in st.session_state.keys():
        st.session_state['slope'] = None         
    if not 'times' in st.session_state.keys():
        st.session_state['times'] = None
 
def widget():
    if st.sidebar.button("Select a shape file"):
        selected_file = file_selector_wx()
        if selected_file:
            st.success(f"Selected File: {selected_file}")
            st.session_state['shapeFile'] = selected_file

    if st.sidebar.button("Select a prior file"):
        selected_file = file_selector_wx()
        if selected_file:
            st.success(f"Selected File: {selected_file}")
            st.session_state['priorFile'] = selected_file
            
    if st.sidebar.button("Select a SWORD file"):
        selected_file = file_selector_wx()
        if selected_file:
            st.success(f"Selected File: {selected_file}")
            st.session_state['swordFile'] = selected_file 
               
    if reach_ids := st.sidebar.text_input("Enter the reach ids from upstream to downstream"):
        st.session_state['reaches'] = reach_ids
        save_session_state('streamlit.ss')
    
    if st.sidebar.button('Read data'):
        if  st.session_state['shapeFile'] == 'No file': return
        if  st.session_state['swordFile'] == 'No file': return
        if  st.session_state['priorFile'] == 'No file': return
        if  st.session_state['reaches'] == 'No reaches': return
        
        # get the reach ids
        try:
            AllObs,DAll,QBar,tall,reach_ids= readData(reach_ids_str= st.session_state['reaches'],shapefile=st.session_state['shapeFile'],priorfile=st.session_state['priorFile'],
                     swordfile=st.session_state['swordFile'])
            # Filter the width data using mean + 1 std
            data = AllObs.w
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)

            # Define allowable bounds
            lower_bound = mean - 0 * std
            upper_bound = mean + 0 * std

            # Clip values to bounds
            adjusted_data = np.clip(data, lower_bound, upper_bound)
            AllObs.w = adjusted_data
            
            # Filter the height data using mean + 1 std
            data = AllObs.h
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)

            # Define allowable bounds
            lower_bound = mean - 0 * std
            upper_bound = mean + 0 * std

            # Clip values to bounds
            adjusted_data = np.clip(AllObs.h, lower_bound, upper_bound)
            AllObs.h = adjusted_data            
            
            # Filter the height data using mean + 1 std
            data = AllObs.S
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)

            # Define allowable bounds
            lower_bound = 0.0001
            upper_bound = 0.0001

            # Clip values to bounds
            adjusted_data = np.clip(AllObs.S, lower_bound, upper_bound)
            AllObs.S = adjusted_data  
            
                        
            st.session_state['wse'] = AllObs.h
            st.session_state['width'] = AllObs.w
            st.session_state['slope'] = AllObs.S
            st.session_state['times'] = tall
            print(tall)
            print(AllObs.h)
            save_session_state('streamlit.ss')

        except Exception as e:
            st.write(e)
            return
    if st.sidebar.button("Process"):
        AllObs,DAll,QBar,tall,reach_ids=readData(reach_ids_str = st.session_state['reaches'],shapefile=st.session_state['shapeFile'],priorfile=st.session_state['priorFile'],
            swordfile=st.session_state['swordFile'])
        C, R, Exp, P = set_up_experiment(DAll, Qbar=QBar)

        Estimate = process(DAll, AllObs=AllObs, Exp=Exp, P=P, R=R, C=C, Verbose=True)
        
        st.write(f"Estimated discharge{Estimate.AllQ}")
        st.session_state['AllQ'] = Estimate.AllQ
        st.session_state['times'] = tall
        save_session_state('streamlit.ss')


                    
def plot():
    print(type(st.session_state['times']) != type(None))
    print(type(st.session_state['reaches']) != type(None))
    if type(st.session_state['AllQ']) != type(None) and type(st.session_state['times']) != type(None) and type(st.session_state['reaches']) != type(None):
        AllQ = st.session_state['AllQ']
        tall = st.session_state['times']
        reach_ids = st.session_state['reaches']
        for idx, row in enumerate(AllQ):
            fig = plt.figure()
            plt.plot(tall, row, marker='o')
            plt.title(f' reach {reach_ids[idx]}')
            plt.xlabel('Time')
            plt.ylabel('Q')
            plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
            plt.grid(True)
            plt.tight_layout()
            st.pyplot(fig)
 
    if type(st.session_state['wse']) != type(None) and type(st.session_state['times']) != type(None) and type(st.session_state['reaches']) != type(None):
        wse = st.session_state['wse']
        tall = st.session_state['times']
        
        reach_ids = st.session_state['reaches'].split(',')
        for idx, row in enumerate(wse):
            fig = plt.figure()
            plt.plot(tall, row, marker='o')
            plt.title(f' reach {reach_ids[idx]}')
            plt.xlabel('Time')
            plt.ylabel('Wse (m)')
            plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
            plt.grid(True)
            plt.tight_layout()
            st.pyplot(fig)
    if type(st.session_state['width']) != type(None) and type(st.session_state['times']) != type(None) and type(st.session_state['reaches']) != type(None):
        width = st.session_state['width']
        tall = st.session_state['times']
        reach_ids = st.session_state['reaches'].split(',')
        for idx, row in enumerate(width):
            fig = plt.figure()
            plt.plot(tall, row, marker='o')
            plt.title(f' reach {reach_ids[idx]}')
            plt.xlabel('Time')
            plt.ylabel('Width (m)')
            plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
            plt.grid(True)
            plt.tight_layout()
            st.pyplot(fig)     
def showTable():
    if type(st.session_state['reaches']) != str:
        return
    reach_ids_str = st.session_state['reaches']
    reach_ids = reach_ids_str.split(',')
    reach_ids = list(map(int,reach_ids))
    if type(st.session_state['AllQ']) != type(None):

        AllQ = st.session_state['AllQ']
        st.write('Discharge Q')
        try:
            df = pd.DataFrame(AllQ.T,columns=reach_ids)
            tall = st.session_state['times']
            if len(tall) == df.shape[0]:
                df.insert(0,'date',tall)
            st.dataframe(df)
 
        except Exception as e:
            print(e)
            
    if type(st.session_state['wse']) != type(None) :
        wse = st.session_state['wse']
        st.write('Water Surface elevation')
        try:
            
            df = pd.DataFrame(wse.T,columns=reach_ids)
            df.insert(0,'date',tall)
            st.dataframe(df)

        except Exception as e:
            print(e)
    if type(st.session_state['width']) != type(None) :
        width = st.session_state['width']
        st.write('River width')
        try:
            
            df = pd.DataFrame(width.T,columns=reach_ids)
            df.insert(0,'date',tall)
            
            st.dataframe(df)    
        except Exception as e:
                print(e)
    if type(st.session_state['slope']) != type(None) :
        slope = st.session_state['slope']
        st.write('River slope')
        try:
            
            df = pd.DataFrame(slope.T,columns=reach_ids)
            df.insert(0,'date',tall)
            
            st.dataframe(df)    
        except Exception as e:
                print(e)                
## main()

def main():
    load_session_state('streamlit.ss')
    init()
    widget()
    displayMessages()
    plot()
    showTable()
    save_session_state('streamlit.ss')
    
    
if __name__ == '__main__':
    main()
    
    