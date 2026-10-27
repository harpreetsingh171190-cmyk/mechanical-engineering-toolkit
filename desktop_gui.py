"""Dependency-light desktop GUI using Tkinter."""
import tkinter as tk
from tkinter import ttk, messagebox
from mechanical_ds.analytics import moving_average_forecast, simulate_mass_spring, optimize_beam_depth
from mechanical_ds.storage import HistoryStore

class App(tk.Tk):
    def __init__(self):
        super().__init__(); self.title("Mechanical Engineering Toolkit"); self.geometry("680x500")
        ttk.Label(self,text="Mechanical Engineering Toolkit",font=("Arial",18,"bold")).pack(pady=12)
        self.tabs=ttk.Notebook(self); self.tabs.pack(fill="both",expand=True,padx=12,pady=8)
        self.make_tab("Forecast", [("Measurements, comma separated","10,11,12,11.5,13"),("Forecast steps","3")], self.forecast)
        self.make_tab("Beam optimizer", [("Load (N)","1000"),("Span (m)","2"),("Width (m)","0.05"),("Allowable stress (Pa)","150000000")], self.beam)
        self.make_tab("Vibration", [("Mass (kg)","10"),("Stiffness (N/m)","1000"),("Damping (N s/m)","5")], self.vibration)
    def make_tab(self,title,fields,callback):
        frame=ttk.Frame(self.tabs,padding=16); self.tabs.add(frame,text=title); entries=[]
        for i,(label,default) in enumerate(fields):
            ttk.Label(frame,text=label).grid(row=i,column=0,sticky="w",pady=5); e=ttk.Entry(frame,width=42); e.insert(0,default); e.grid(row=i,column=1,pady=5); entries.append(e)
        ttk.Button(frame,text="Calculate",command=lambda: self.safe(callback,entries)).grid(row=len(fields),column=0,columnspan=2,pady=12)
        out=ttk.Label(frame,text="Results will appear here",wraplength=580); out.grid(row=len(fields)+1,column=0,columnspan=2,sticky="w"); frame.out=out
    def safe(self,fn,entries):
        try: fn([e.get() for e in entries], self.tabs.nametowidget(self.tabs.select()).out)
        except Exception as exc: messagebox.showerror("Input / calculation error",str(exc))
    def forecast(self,v,out):
        pred=moving_average_forecast([float(x) for x in v[0].split(",")],int(v[1])); result={"forecast":pred.tolist()}; HistoryStore().add("forecast",result); out.config(text=str(result))
    def beam(self,v,out):
        r=optimize_beam_depth(*map(float,v)); HistoryStore().add("beam_depth_optimization",r); out.config(text="\n".join(f"{k}: {val:.6g}" for k,val in r.items()))
    def vibration(self,v,out):
        r=simulate_mass_spring(float(v[0]),float(v[1]),float(v[2]),.05,0,duration_s=5); summary={"final_displacement_m":float(r['displacement_m'][-1]),"samples":len(r['time_s'])}; HistoryStore().add("vibration_simulation",summary); out.config(text=str(summary))
if __name__=="__main__": App().mainloop()
