# coding: utf-8
import state_io

with open("rundir/state.pkl", "rb")  as f:
    data = state_io.load_state(f)
    
print(data)