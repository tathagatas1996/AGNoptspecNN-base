import pandas as pd

emission_lines = {
    1: ["Hgamma", 4340.471 ,'bn', 0.2 ],
    2: ["HeII"  , 4685.710 ,'b' , 0.1 ],
    3: ["Hbeta" , 4861.333 ,'bn', 0.3 ],
    4: ["OIII_1", 4958.911 ,'n' , 0.5 ],
    5: ["OIII_2", 5006.843 ,'n' , 1.0 ],
    6: ["NII_1" , 6548.050 ,'n' , 0.4 ],
    7: ["Halpha", 6562.819 ,'bn', 0.8 ],
    8: ["NII_2" , 6583.460 ,'n' , 0.4 ],
    9: ["SII_1" , 6716.440 ,'n' , 0.2 ],
    10:["SII_2", 6730.810  ,'n' , 0.2 ],}

def line_list():
    df = pd.DataFrame(emission_lines).T
    df = df.rename(columns={0:"Line", 1:"Wavelength", 2:"broad/narrow", 3:"amplitude"})
    return df