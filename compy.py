#__version__ = "0.1.0"
#__author__ = 'Francisco Iaconis, Jessica Del Punta, Adrian Jimenez Gandica'
#__credits__ = 'UNS - CONICET'

import pandas as pd
import glob
from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy.signal as sg

def Npattern(permdf,D,*arg):
    distx = get_patterns(arg[0],D,permdf)
    distx.columns = ['pat_pattern',arg[0].name]
    if len(arg)>1:
        for a in arg[1:]:
            disty = get_patterns(a,D,permdf)
            disty.columns = ['pat_pattern',a.name]
            distx = pd.merge(distx,disty,how ="outer",on = "pat_pattern") # Uno las distribuciones en un solo dataframe
    return(distx)

def patterns(filename,D,root):# D es el ws, el tamaño de la ventana
    
    permdf = make_patterns(D) # Fabrico los patrones. Funcion que se explica a continuacion
    
    for archivo in filename: # para cada elemento en "filename" hacer lo siguiente:    
        dft = pd.read_csv(archivo) # se l

    ## Por archivo voy comparando la serie temoral con los patrones. Esto lo hago con la funcion "get_patterns("serie temporal",'d' de patterns)"
        try:
            dist = pd.DataFrame()
            # saco la distribucion de probabilidades para 'x' y para 'y'
            disty = get_patterns(dft['y'],D,permdf)
            distx = get_patterns(dft['x'],D,permdf)
            
            dist = pd.merge(distx,disty,how ="outer",on = "pat_pattern") # Uno las distribuciones en un solo dataframe
            dist.columns = ['pattern','x','y'] # Asigno nombres a las columnas

            #tiempo_total = dft["t"].iloc[-1]  # Calculo el tiempo total del test
            f = archivo
            n = min([f[::-1].find('/'),f[::-1].find('\\')]) # esto es para encontrar la ultima barra del file
            archivo = f[-n:-4]# El nombre del file solamente
            #dist.to_csv(("..//Complejidad - Datos/Distribuciones/"+archivo+'_t'+str(int(tiempo_total/1000))+"_"+str(D)+"_patterns.csv"),index = False) #guardo las distribuciones en archivos
            dist.to_csv((root+archivo+"_D"+str(D)+"_patterns.csv"),index = False) #guardo las distribuciones en archivos
        except:
            print("Error en el archivo "+archivo+". No se pudo calcular los patrones por alguna razon")

def make_patterns(d):
    ''' 
    Esta funcion crea los patrones de 'd' puntos
    '''
    listname = []
    v = range(0,d) # esto me da una lista que va de cero a d-1
    perm = permutations(v) # calculo todas las permutaciones
    permdf = pd.DataFrame(list(perm)) #las pongo en un dataframe ordenadas. cada fila es una permutacion y cada columna es una posicion
    for i in range(1,d+1):
        listname.append("x"+str(i))
    permdf.columns=listname# la columnas se llaman ['x1','x2','x3']
    
    permdf = permdf.astype(str)
    permdf["pat_pattern"] = permdf["x1"]
    
    for i in range(2,d+1):
        permdf["pat_pattern"] = permdf["pat_pattern"]+permdf["x"+str(i)] # esto es para que me queden los patrones en una columna 

    return (permdf)

def get_patterns(dfx,d,permdf):
    tempdf = pd.DataFrame()
    tempdf["x1"]=dfx #dft['x']
    for i in range(1,d):
        tempdf["x"+str(i+1)]=dfx.shift(-i) 

    # Borro los primeros puntos que me quedan con NaN
    tempdf = tempdf.drop(tempdf.tail(d-1).index)#,inplace=True)
    
    sort_index = np.argsort(tempdf) # Esto me da directamente el patron. me ordena de menos a mayor y me da el indice (excelente!!!!)
    sortdf = pd.DataFrame(sort_index) # Lo convierto a dataframe
    sortdf = sortdf.astype(str)
    
    sortdf["pat_pattern"] = sortdf["x1"]
    for j in range(2,d+1):
        sortdf["pat_pattern"] = sortdf["pat_pattern"]+sortdf["x"+str(j)]

    sortdf["cant"] = 1
    sortdfgroup = sortdf.groupby("pat_pattern")
    pattDist = sortdfgroup.sum()/sortdfgroup.sum().sum()
    pattDist.reset_index(level=0, inplace=True)
    
    pattdistFull = pd.DataFrame()
    #pattdistFull = pd.merge(pattDist,permdf,how ="outer",on = "pat_pattern")
    pattdistFull = pd.merge(permdf,pattDist,how ="outer",on = "pat_pattern")
    pattdistFull = pattdistFull.fillna(0.0)
    pattDist1 = pattdistFull[["pat_pattern","cant"]]
    
    return(pattDist1)

def pattern_mod(filename,s,points,root):

    patterns_base = make_patt_dir(points)
    for archivo in filename: # para cada elemento en "filename" hacer lo siguiente:    
        dft = pd.read_csv(archivo)

    ## Por archivo voy comparando la serie temoral con los patrones. Esto lo hago con la funcion "get_patterns("serie temporal",'d' de patterns)"
        try:
            dist = pd.DataFrame()
            dist,ur = get_pattern_mod(dft,s,points,archivo,patterns_base) # Esta funcion es la que obtiene las distribuciones a partir de la señal
            
            f = archivo
            n = min([f[::-1].find('/'),f[::-1].find('\\')]) #Doy vuelta el nombre y busco donde esta la barra. Esta posicion con signo menos es la que me dice donde empuieza el nombre del file
            archivo = f[-n:-4]# El nombre del file solamente
            
            dist.to_csv((root+archivo+"_points"+str(points)+"_s"+str(s)+"_pattern_mod.csv"),index = False) #guardo las distribuciones en archivos
        except:
            print("Error en el archivo "+archivo+". No se pudo calcular los patrones por alguna razon")

def make_patt_dir(points):
    patternr =pd.DataFrame(list(itertools.product(('L','R','U','D','S'), repeat=points)))
    patternr['pattern'] = patternr[0] 
    for i in patternr.columns:
       # print i
        try:
            patternr['pattern'] = patternr['pattern']+patternr[i+1] 
        except:
            pass
    return(patternr)

def Npatt_mod(ws,patternr,datax,datay,diff_umbral):
     # calculo el desplazamiento punto a punto "r", solo para obtener un valor de umbral para definir el estado "stay"
    df_temp = pd.DataFrame({'x':datax,
                           'y':datay})
    df_temp['x_next'] = df_temp['x'].shift(-1)
    df_temp['y_next'] = df_temp['y'].shift(-1)
    df_temp['rx'] = df_temp['x_next'] - df_temp['x']
    df_temp['ry'] = df_temp['y_next'] - df_temp['y']
    df_temp['r'] =  np.sqrt(df_temp['rx']**2+df_temp['ry']**2)
    
    dft = df_temp.copy()
    # Defino las direcciones
    dft['dir'] = 0
    dft['dir'] = dft['dir'].mask((dft['ry']/dft['rx'] < 1) & (dft['ry']/dft['rx'] > -1) & (dft['rx'] > 0),'L')
    dft['dir'] = dft['dir'].mask((dft['ry']/dft['rx'] < 1) & (dft['ry']/dft['rx'] > -1) & (dft['rx'] < 0),'R')
    dft['dir'] = dft['dir'].mask(((dft['ry']/dft['rx'] < -1) | (dft['ry']/dft['rx'] > 1)) & (dft['ry'] > 0),'U')
    dft['dir'] = dft['dir'].mask(((dft['ry']/dft['rx'] < -1) | (dft['ry']/dft['rx'] > 1)) & (dft['ry'] < 0),'D')
    
    diff_umbral_value = dft.r.quantile(diff_umbral)
    dft['dir'] = dft['dir'].mask(dft['r']<=diff_umbral_value,'S')

    dft['dir0'] = dft['dir']
    del dft['dir']
    
    # no tengo idea del por qué de esto pero por alguna razon lo puse
    for i in range(ws-1):
        dft['dir'+str(i+1)] = dft['dir'+str(i)].shift(-1)
    dft = dft[1*ws:]
    dft = dft[:-ws*1]

    dft['patt_mod'] = dft['dir0']
    for i in range(1,ws):
        dft['patt_mod'] = dft['patt_mod'] + dft['dir'+str(i)]


    grouped_patt = dft.groupby("patt_mod")
    
    patt_mod_dist = pd.DataFrame()
    
    for group in grouped_patt.groups:
        patt_cant = pd.DataFrame([ [group , len(grouped_patt.groups[group])] ])
        patt_mod_dist = pd.concat([patt_mod_dist,patt_cant])
    
    patt_mod_dist.columns = ['pattern','cant']
    patt_mod_dist['cant'] = patt_mod_dist['cant']/patt_mod_dist['cant'].sum() 

    patt_mod_dist = patt_mod_dist.reset_index(drop=True)
    new_index = list(range(len(patt_mod_dist)))

    patt_mod_dist.reindex(new_index)
    patt_tot = pd.merge(patternr,patt_mod_dist,how = 'left', on ='pattern' )
    #patt_tot = pd.merge(patt_mod_dist,patternr,how = 'left', on ='pattern' )
    patt_tot = patt_tot.fillna(0)
    patt_tot = patt_tot[['pattern','cant']]
    return (patt_tot)

def get_pattern_mod(dft,s,points,archivo,patternr):
    '''
    dft: DataFrame de datos
    s: parametro de subsampleo
    points: cuantos puntos forman los patrones (windows size)
    archivo: nombre del file a evaluar
    patternr: posibles patrones
    '''
    # calculo el desplazamiento punto a punto "r", solo para obtener un valor de umbral para definir el estado "stay"
    df_temp = dft.copy()
    df_temp['x_next'] = df_temp['x'].shift(-1)
    df_temp['y_next'] = df_temp['y'].shift(-1)
    df_temp['rx'] = df_temp['x_next'] - df_temp['x']
    df_temp['ry'] = df_temp['y_next'] - df_temp['y']
    df_temp['r'] =  np.sqrt(df_temp['rx']**2+df_temp['ry']**2)
    
    # Defino un umblar para determinar cuando el sujeto esta realizando una fijacion
    if 'CARAS' in archivo: 
        ur  = float(df_temp['r'].quantile([0.85]))
    elif 'TMT' in archivo:
        ur  = float(df_temp['r'].quantile([0.75]))
    else:
        ur  = float(df_temp['r'].quantile([0.8])) # Este porcentaje es arbitrario, podria ser distinto
        
    dft = sub_samp(dft,s,'dec') # esta funcion subsamplea la señal
    # Vuelvo a calcular el desplazamiento parcial con la señal subsampleada
    
    dft['x_next'] = dft['x'].shift(-1)
    dft['y_next'] = dft['y'].shift(-1)
    dft['rx'] = dft['x_next'] - dft['x']
    dft['ry'] = dft['y_next'] - dft['y']
    dft['r'] =  np.sqrt(dft['rx']**2+dft['ry']**2)
    # Defino las direcciones
    dft['dir'] = 0
    dft['dir'] = dft['dir'].mask((dft['ry']/dft['rx'] < 1) & (dft['ry']/dft['rx'] > -1) & (dft['rx'] > 0),'L')
    dft['dir'] = dft['dir'].mask((dft['ry']/dft['rx'] < 1) & (dft['ry']/dft['rx'] > -1) & (dft['rx'] < 0),'R')
    dft['dir'] = dft['dir'].mask(((dft['ry']/dft['rx'] < -1) | (dft['ry']/dft['rx'] > 1)) & (dft['ry'] > 0),'U')
    dft['dir'] = dft['dir'].mask(((dft['ry']/dft['rx'] < -1) | (dft['ry']/dft['rx'] > 1)) & (dft['ry'] < 0),'D')
    
    dft['dir'] = dft['dir'].mask(dft['r']<=ur,'S')

    dft['dir0'] = dft['dir']
    del dft['dir']
    
    # no tengo idea del por qué de esto pero por alguna razon lo puse
    for i in range(points-1):
        dft['dir'+str(i+1)] = dft['dir'+str(i)].shift(-1)
    dft = dft[1*points:]
    dft = dft[:-points*1]

    dft['patt_mod'] = dft['dir0']
    for i in range(1,points):
        dft['patt_mod'] = dft['patt_mod'] + dft['dir'+str(i)]


    grouped_patt = dft.groupby("patt_mod")
    
    patt_mod_dist = pd.DataFrame()
    
    for group in grouped_patt.groups:
        patt_cant = pd.DataFrame([ [group , len(grouped_patt.groups[group])] ])
        patt_mod_dist = pd.concat([patt_mod_dist,patt_cant])
    
    patt_mod_dist.columns = ['pattern','cant']
    patt_mod_dist['cant'] = patt_mod_dist['cant']/patt_mod_dist['cant'].sum() 

    patt_mod_dist = patt_mod_dist.reset_index(drop=True)
    new_index = list(range(len(patt_mod_dist)))

    patt_mod_dist.reindex(new_index)
    patt_tot = pd.merge(patternr,patt_mod_dist,how = 'left', on ='pattern' )
    #patt_tot = pd.merge(patt_mod_dist,patternr,how = 'left', on ='pattern' )
    patt_tot = patt_tot.fillna(0)
    patt_tot = patt_tot[['pattern','cant']]
    return (patt_tot,ur)

def sub_samp(df,s,method):
    # Esta funcion subsamplea con dos metodos, "decimal" (dec) y "resample"(res). Usa de argumentos df que es el
    # dataframe a subsamplear,"s" que es cuantos puntos quitamos por cada punto del df y "method" puede
    # tener valores de "dec" o "res".
    if method=='dec':
        dec = pd.DataFrame()
        for col in df.columns:
            if ((col=='x') or (col=='y') or (col =='t')):
                dec[col] = sg.decimate(df[col],s+1)
        #dec['t'] = sg.decimate(df.t,10)
        return (dec)
    elif method== 'res':
        res = pd.DataFrame()
        res_len = int(len(df)/(s+1)) # cantidad de puntos que quiero que tenga la nueva señal
        for col in df.columns:
            if col!='trial':
                res[col] = sg.resample(df[col],res_len)
        #res['t'] = sg.resample(df.t,res_len)
        return(res)
    else:
        return df
        #print('no se eligio metodo de subsampling')

def Ntrinaria (data,ws):
    caras = getFace(data,0.034)
    trindist = getTrin(caras,ws)
    patternr = posible_trinaria(ws)

    tridistFull = pd.DataFrame()
    tridistFull = pd.merge(patternr,trindist,how ="outer",on = "pat_tern")
    tridistFull = tridistFull.fillna(0.0)

    tridistFull = tridistFull[["pat_tern","cant"]]
    tridistFull.columns = ['pattern','x']
    return (tridistFull)

def posible_trinaria(ws):
    import itertools
    prueba_df = pd.DataFrame(list(itertools.product(list(range(3)), repeat=ws))).astype('str')
    prueba_df["pat_tern"] = ''
    for j in range(ws):
        prueba_df["pat_tern"] += prueba_df[j]
    patternr = pd.DataFrame({'pat_tern':prueba_df['pat_tern'].astype('str')})
    return(patternr)
        
def trinaria(filename,Div,winSize,s,root):
    patternr =pd.DataFrame()
    #patternr es un dataframe de permutaciones con repeticion que se logran con "winSize" puntos
    patternr =pd.DataFrame(list(itertools.product(list(range(3)), repeat=winSize)))##6/8/19 toque el range. estaba range(winSize)
   
    patternr = patternr.astype(str) # Lo convierto a strings 
    patternr["pat_tern"] = patternr[0]
    for j in range(1,winSize): 
        patternr["pat_tern"] = patternr["pat_tern"]+patternr[j]
    filename = [k for k in filename if 'CARAS' in k] #este filtro me deja los archivos que tengan 'CARAS' en su nombre nada mas
    
    for archivo in filename: # para cada elemento en "filename" hacer lo siguiente:    

        dft = pd.read_csv(archivo)
        dft = sub_samp(dft,s,'dec') #Funcion que subsamplea la señal
       
    ################################ Aca esta el juguito ###############################################
        # calculo las cantidades de veces que aparcen cada patron

        tri = pd.DataFrame()
        tri[0] = getFace(dft["x"],Div) 
        trindist = getTrin(tri,winSize)
        #trindist.plot.bar(x = "pat_tern",y="cant")
        tridistFull =pd.DataFrame()
        tridistFull = pd.merge(patternr,trindist,how ="outer",on = "pat_tern")
        tridistFull = tridistFull.fillna(0.0)

        #print (archivo)
        
        #tridistFull = tridistFull.drop([0, 1, 2], axis=1)
        tridistFull = tridistFull[["pat_tern","cant"]]
        tridistFull.head()
        tridistFull.columns = ['pattern','x']
        try:
            f = archivo
            n = min([f[::-1].find('/'),f[::-1].find('\\')]) #Doy vuelta el nombre y busco donde esta la barra. Esta posicion con signo menos es la que me dice donde empuieza el nombre del file
            archivo = f[-n:-4]# El nombre del file solamente
            #tridistFull.to_csv("..//Complejidad - Datos/Distribuciones/"+archivo+'_t'+str(int(tiempo_total/1000))+"_s"+str(s)+"_trinaria_ws"+str(winSize)+".csv",index = False)
            tridistFull.to_csv(root+archivo+"_s"+str(s)+"_trinaria_ws"+str(winSize)+".csv",index = False)
        except:
            print("Error en el archivo "+archivo+". No se pudo calcular los patrones por alguna razon")

def getFace(xx,Div):
    #Div = Div/1080.0
    Drmin = np.zeros(len(xx), dtype=int) #Hago vector con todos ceros
    Drmed = np.where((xx >= (np.mean(xx) - Div)) & (xx <= (np.mean(xx) + Div)), 1, 0) #Los de en medio les doy el valor 1
    Drmax = np.where((xx >= (np.mean(xx) + Div)), 2, 0) # los de arriba el valor dos
    Lx = Drmin + Drmed + Drmax #Sumos los tres. ME DA LOS PATRONES PUNTO A PUNTO
    return (Lx)

def getTrin(tri,winSize):
    tri = pd.DataFrame(tri)
    for i in range(1,winSize):
        tri[i] = tri[i-1].shift(-1)
    tri=tri.drop(tri.index[-((winSize-1)):])
    tri = tri.astype(int)
    tri = tri.astype(str)
    tri["pat_tern"] = tri[0]
    for j in range(1,winSize):
        tri["pat_tern"] = tri["pat_tern"]+tri[j]
        
    tri["cant"] = 1
    trigroup = tri.groupby("pat_tern")
    trinDist = trigroup.sum()/trigroup.sum().sum()
    trinDist.reset_index(level=0, inplace=True)
    return (trinDist)

def Nhistograma(Nbins,*arg):
    hist_dist = pd.DataFrame()
    for a in arg:
        hx, b_ex = np.histogram(a, bins =  np.arange(0,1,1.0/Nbins))
        hx=hx.astype(float)/sum(hx)
        #histdist = pd.DataFrame({'x':hx,'y':hy})
        hist_dist[a.name] = hx
    return (hist_dist)

def histograma(filenames, Nbins,root):
    for archivo in [filenames]: # para cada elemento en "filename" hacer lo siguiente:    
        
        dft = pd.read_csv(archivo)

        try:
            dist = pd.DataFrame()
            #dft["t"] = dft["t"]-dft["t"].iloc[0]
        ################################ Aca esta el juguito ###############################################
            # calculo las cantidades de veces que aparcen cada patron
            hx, b_ex = np.histogram(dft["x"], bins =  np.arange(0,1,1.0/Nbins)) #Hago los histogramas de a Bins en x e y
            hy, b_ey = np.histogram(dft["y"], bins =  np.arange(0,1,1.0/Nbins))

            hx=hx.astype(float)/sum(hx) #Lo normalizo para tener distribución de prob
            hy=hy.astype(float)/sum(hy)
            histdist = pd.DataFrame({'x':hx,'y':hy}) #Lo almaceno en un dataframe
            

            #print (archivo)
            #tiempo_total = int(dft["t"].iloc[-1]/1000)

            f = archivo
            n = min([f[::-1].find('/'),f[::-1].find('\\')]) #Doy vuelta el nombre y busco donde esta la barra. Esta posicion con signo menos es la que me dice donde empuieza el nombre del file
            archivo = f[-n:-4]# El nombre del file solamente
            #histdist.to_csv("..//Complejidad - Datos/Distribuciones/"+archivo+'_t'+str(tiempo_total)+"_histogram_bins"+str(Nbins)+".csv",index = True)
            histdist.to_csv(root+archivo+"_histogram_bins"+str(Nbins)+".csv",index = True)
            
        except:
            print("Error en el archivo "+archivo+". No se pudo calcular los patrones por alguna razon")

def entropia(dist):
    ## Esta funcion calcula la entropia de Jensen de la distribucion "dist". dist es una serie de pandas o un array de numpy
    try:
        h = np.sum(-dist*np.log(dist))/np.log(len(dist))
    except:
        h = 0
    return (h)

def desequilibrio(dist,distn):
    '''
    dist: es la distribucion a analizar. 
    distn: es la distribucion normal de la misma cantidad de bines que dist
    '''
    # Calculo del Qo
    distpmax = pd.DataFrame()
    newdist = pd.DataFrame()
    distpmax["pe"] = dist['x'] # esto lo hago para que tenga el mismo largo que los datos a comparar
    #El maximo desequilibrio es cuando un bin de la distribucion contiene toda la probablidad y el resto es cero
    distpmax["pe"] = 0.0
    distpmax['pe'].iloc[1] = 1
    newdist['pe'] = (distpmax['pe'] + distn['pe'])/2.0
    qo = 1.0/(entropia(newdist["pe"])-0.5) # este es un parametro que sale de una ecuacion de algun paper citado por Rosso y en alguno de el

    # Calculo de las Q's
    qx = qo * (entropia((dist['x']+distn['pe'])/2)-entropia(dist['x'])/2-entropia(distn['pe'])/2)
    try: # este try es porque hay distribuciones que no tienen la columna y
        qy = qo * (entropia((dist['y']+distn['pe'])/2)-entropia(dist['y'])/2-entropia(distn['pe'])/2)
    except:
        qy = np.nan
    return qx,qy

def complex_entropy(dist):
    # Esta funcion tiene como argmento un dataframe que es una distribucion de probabilidades. Devuelve la entropia, complejidad, desequilibrio, 
    # entropia y fisher en ambos ejes. si la distribucion tiene solo un eje, como es el ejemplo de la trinaria o patterns direccional los resultados
    # van a estar en el eje x nada mas.
    
    # Genero la distribucion uniforme
    distn = pd.DataFrame()
    try:
        distn["pe"] = dist['x']
        distn["pe"] = 1.0/len(dist['x'])
    except:
        colnames = dist.columns
        dist['x'] = dist[colnames[1]]
        distn["pe"] = dist['x']
        distn["pe"] = 1.0/len(dist['x'])

    # Calculo la entropia para x y para y
    sx = entropia(dist['x'])
    try:
        sy = entropia(dist['y'])
    except:
        sy = np.nan
    sn = entropia(distn['pe'])

    # Medicion de desequilibrio
    qx , qy = desequilibrio(dist,distn) 

    # Calculo de complejidad
    cx = sx*qx
    try:
        cy = sy*qy
    except:
        cy = np.nan

    #calculo de Fisher
    dist['xnext'] = dist['x'].shift(-1)
    try:
        dist['ynext'] = dist['y'].shift(-1)
    except:
        dist['ynext'] = np.nan
        pass

    fx = 0.5*np.sum(np.power(np.sqrt(dist['xnext'])-np.sqrt(dist['x']),2))
    try:
        fy = 0.5*np.sum(np.power(np.sqrt(dist['ynext'])-np.sqrt(dist['y']),2))
    except:
        fy = np.nan
        
    # Resultado final
    result = pd.DataFrame({'Sx':[sx],
                          'Sy':[sy],
                          'Qx':[qx],
                          'Qy':[qy],
                          'Cx':[cx],
                          'Cy':[cy],
                          'Fx':[fx],
                          'Fy':[fy]})
    return(result)

def mapDist(nbines):
    # Esta funcion devuelve los puntos que van a definir la trayectoria de maxima y minima complejidad
    d = pd.DataFrame(np.triu(np.ones((nbines,nbines), dtype=np.bool_)).T).astype(int)
    for i in range(nbines):
        d.iloc[i] = d.iloc[i]/sum(d.iloc[i])
    d=pd.concat([d,pd.DataFrame(d.iloc[0]).T])
    l = len(d)
    d =d.T
    d.columns = range(l)
    return(d)

def curvas(nbines,precision):
    xo = 0.05 #Punto de quiebre de la pendiente t vs j
    yo = 0.95 # Punto de quiebre en Y de la pendiente t vs j
    cc = (1-yo)/(1-xo)
    dd = yo - cc*xo
    h = []
    p=[]
    q = []
    t = []
    tt = []
    qq = []

    d = mapDist(nbines)
    newdf = pd.DataFrame(d[0])
    for n in range(nbines-1):
        for j in range(precision):
            q = (1.0/(precision-j))-1.0/(precision) # meter el cambio de variable aca
            if q < xo:
                t = q*(yo/xo)
            if q >= xo:
                t = q*cc + dd
            a = d[n]
            b = d[n+1]
            tt = np.append(t,tt)
            qq = np.append(q,qq)
            p = (1-t)*a + t*b
            #p = (t-1)*(t-1)*a + (1-(t-1)*(t-1))*b
            #p = (1-np.power(t,0.5))*a + (np.power(t,0.5))*b
            #if t <= xo:
            #    p = (1-t*(yo/xo))*a + t*(yo/xo)*b
            #elif t>xo:
                # y = c*t + d
            #    p = (t*cc + dd )*a + (1-(t*cc + dd))*b
            newdf = pd.concat([newdf,p],axis=1)

    n = nbines-1
    precision = 100
    for j in range(precision):
        q = (1.0/(precision-j))-1.0/(precision) # meter el cambio de variable aca
        if q < xo:
            t = q*(yo/xo)
        if q >= xo:
            t = q*cc + dd
        a = d[n]
        b = d[n+1]
        tt = np.append(t,tt)
        qq = np.append(q,qq)
        p = (1-t)*a + t*b
        #p = (t-1)*(t-1)*a + (1-(t-1)*(t-1))*b
        #p = (1-np.power(t,0.5))*a + (np.power(t,0.5))*b
        #if t <= xo:
        #    p = (1-t*(yo/xo))*a + t*(yo/xo)*b
        #elif t>xo:
            # y = c*t + d
        #    p = (t*cc + dd )*a + (1-(t*cc + dd))*b
        newdf = pd.concat([newdf,p],axis=1)

            
            
        # Genero la distribucion uniforme
    newdf.columns = range(newdf.shape[1])
    distn = pd.DataFrame()
    distn["pe"] = newdf[0]
    distn["pe"] = 1.0/nbines

    # Calculo la entropia para x y para y
    curva = pd.DataFrame()
    curvaTemp = pd.DataFrame()
    h = []
    for p in newdf.columns:
        ht = entropia(newdf[p])
        h = np.append(h,ht)
        #curvaTemp[p] = p
        #curva = pd.concat([curva,curvaTemp])
    #sy = entropia(dist['y'])
    sn = entropia(distn['pe'])

    # Medicion de desequilibrio

        # Calculo del Qo
    distpmax = pd.DataFrame()
    newdist = pd.DataFrame()
    distpmax["pe"] = newdf[0] # esto lo hago para que tenga el mismo largo que los datos a comparar
    distpmax["pe"] = 0.0
    distpmax['pe'].iloc[1] = 1
    newdist['pe'] = (distpmax['pe'] + distn['pe'])/2.0
    qo = 1.0/(entropia(newdist["pe"])-0.5)

        #calculo de las Q's
    #qx = qo * (entropia((dist['x']+distn['pe'])/2)-entropia(dist['x'])/2-entropia(distn['pe'])/2)
    q = []
    for p in newdf.columns:
        qt = qo * (entropia((newdf[p]+distn['pe'])/2)-entropia(newdf[p])/2-entropia(distn['pe'])/2)
        q = np.append(q,qt)
    #qy = qo * (entropia((dist['y']+distn['pe'])/2)-entropia(dist['y'])/2-entropia(distn['pe'])/2)

    # Calculo de complejidad
    #cx = sx*qx
    c = h*q
    return(h,c)
