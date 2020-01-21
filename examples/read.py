# coding: utf-8
import state_io

with open("rundir/state.pkl", "rb")  as f:
    data = state_io.load(f)
    
print(data)
