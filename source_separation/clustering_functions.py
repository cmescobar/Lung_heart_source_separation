import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from ss_utils.filter_and_sampling import resampling_by_points
from ss_utils.math_functions import _correlation, _correlations, cosine_similarities


def spectral_correlation_test(W, signal_in, samplerate, N_lax, interval_list, 
                              prom_spectra=False, measure='correlation', 
                              i_selection='max', threshold='mean'):
    '''Función que permite realizar un testeo de cada componente de la matriz
    W sobre el diccionario construido a partir de sonidos puramente respiratorios,
    obtenidos a partir de la misma señal. Retorna un arreglo de booleanos. 
    Si la entrada i es True, se trata de un sonido cardíaco. Si es False, 
    se trata de un sonido respiratorio.
    
    OJO: Se diferencia de la versión 1 en que este aplica zero padding en vez de
    resamplear.
    
    Parameters
    ----------
    W : ndarray
        Matriz W de la descomposición NMF.
    signal_in : ndarray
        Señal de entrada a comparar con las plantillas espectrales obtenidas.
    samplerate : float
        Tasa de muestreo de la base de datos utilizada en el diccionario. Esta debe coincidir
        con la tasa de muestreo de la señal a descomponer.
    N_lax : int
        Cantidad de puntos adicionales que se consideran para cada lado más allá de los
        intervalos dados.
    interval_list : list
        Lista de intervalos de las posiciones de sonidos cardiacos.
    prom_spectra : bool, optional
        Opción de resumir todos los espectros al promedio (estilo Welch PSD). Por defecto es
        False.
    measure : {'cosine', 'correlation'}, optional
        Medida utilizada para calcular la similaridad. 'cosine' usa similaridad de 
        coseno, mientras que 'correlation' calcula el coeficiente de Pearson. Por
        defecto es 'cosine'.
    i_selection : {'max', 'mean'}, optional
        Método de selección del componente que identifica la similarida. Con 'max' 
        se utiliza el máximo de todas las medidas, mientras que con 'mean' se utiliza
        el promedio. Por defecto es 'max'.
    threshold : float, optional
        Valor del umbral de decisión para clasificar una componente de la matriz
        W como sonido cardíaco. Por defecto es 0.5.
        
    Referencias
    -----------
    [1] Canadas-Quesada, et. al (2017). A non-negative matrix factorization 
        approach based on spectro-temporal clustering to extract heart sounds. 
        Applied Acoustics. Elsevier. Chapter 3.2.1.
    [2] Elaboración propia.
    '''        
    # Definición de la lista de segmentos de sonido respiratorio,
    # utilizados para compararse espectralmente
    resp_list = list()
    
    # Variable auxiliar que indica el inicio de una señal
    beg = 0
    
    for i in interval_list:
        # Definición de los límites a revisar
        lower = beg - N_lax if beg != 0 else 0
        upper = i[0] + N_lax
        
        # Segmento de señal respiratoria
        resp_signal = signal_in[lower:upper]
        
        # Agregando a las listas
        resp_list.append(resp_signal)
        
        # Se redefine la variable auxiliar
        beg = i[1]
    
    # Finalmente se agrega el segmento final
    resp_signal = signal_in[i[1] - N_lax:]
    resp_list.append(resp_signal)
    
    # Definición de la cantidad de puntos a utilizar en base al
    # largo de la matriz W, con la cual se compararán los
    # segmentos
    N = (W.shape[0] - 1) * 2
    
    # Definición de la matriz a rellenar
    resp_array = np.empty((len(resp_list), W.shape[0]))
    
    # Re acondicionando las señales
    for i in range(len(resp_list)):
        # Agregando ceros para la cantidad de puntos que se necesita. En caso de que los
        # segmentos sean más largos se cortan hasta N
        if len(resp_list[i]) < N:
            resp = np.concatenate((resp_list[i], [0] * (N - len(resp_list[i])) ))
        else:
            resp = resp_list[i]
        
        # Calculando el periodograma
        _, resp_to = welch(resp, fs=samplerate, nperseg=N, noverlap=int(0.75*N))
        
        # Agregando
        resp_array[i] = 20 * np.log10(resp_to + 1e-12)
        
    # Si es que se promedian los espectros
    if prom_spectra:
        resp_array = np.array([resp_array.mean(axis=0)])
    
    # Definición de la lista de booleanos de salida
    S_i_list = list()
    
    for i in range(W.shape[1]):
        # Se obtiene el bool correspondiente a la componente i. True si
        # es sonido cardíaco y False si es sonido respiratorio
        S_i = _spectral_correlation_criterion(20 * np.log10(W[:,i] + 1e-12), 
                                              resp_array, fcut_bin=-1, 
                                              i_selection=i_selection,
                                              measure=measure)
        # Agregando...
        S_i_list.append(S_i)
    
    # Transformando a array
    S_i_array = np.array(S_i_list)
        
    # Definición de umbral
    if threshold == 'mean':
        threshold = np.mean(S_i_array)
    elif threshold == 'median':
        threshold = np.median(S_i_array)
    
    # Aplicando umbral
    return S_i_array < threshold, S_i_array


def roll_off_test(X_list, f1, f2, samplerate, whole=False, percentage=0.85):
    '''Función que permite realizar un testeo del espectrograma de cada componente. 
    Si la entrada i es True, se trata de un sonido cardíaco. Si es False, se trata 
    de un sonido respiratorio.
    
    Parameters
    ----------
    X_list : list or array
        Lista de espectrogramas de las componentes obtenidas mediante NMF.
    f1 : float
        Valor de la frecuencia de corte inferior. Se recomienda usar en 20 Hz.
    f2 : float
        Valor de la frecuencia de corte superior. Se recomienda usar en 150 Hz.
    samplerate : float
        Tasa de muestreo de la señal a testear.
    whole : bool, optional
        Indica si los espectrogramas de X_list están hasta samplerate (True) o 
        hasta samplerate // 2 (False).
    percentage : float, optional
        Ponderación para la energía total del espectrograma en el criterio de
        selección. Por defecto es 0.85.
    
    Referencias
    -----------
    [1] Canadas-Quesada, et. al (2017). A non-negative matrix factorization 
        approach based on spectro-temporal clustering to extract heart sounds. 
        Applied Acoustics. Elsevier. Chapter 3.2.2.
    '''
    # Definición de la lista de booleanos de salida
    bool_list = list()
    
    # Definición frecuencia de corte
    if whole:
        f1_bin = int(f1 / samplerate * X_list[0].shape[0])
        f2_bin = int(f2 / samplerate * X_list[0].shape[0])
    else:
        f1_bin = int(f1 / (samplerate // 2) * X_list[0].shape[0])
        f2_bin = int(f2 / (samplerate // 2) * X_list[0].shape[0])
        
    for X_i in X_list:
        bool_list.append(_roll_off_criterion(X_i, f1=f1_bin, f2=f2_bin,
                                             percentage=percentage))
        
    return np.array(bool_list)


def energy_percentage_test(W, percentage=0.85):
    '''Criterio de "centroide" propuesto.
    
    Parameters
    ----------
    W : ndarray
        Matriz de plantillas espectrales W del NMF.
    percentage : float, optional
        Porcentaje de energía límite a evaluar. Por defecto es 85%.
    '''
    # Definición de la lista de puntos límite
    limit_point_list = list()
    
    # Agregando los valores para cada componente
    for i in range(W.shape[1]):
        limit_point_list.append(_p_percentage_energy(W[:,i], percentage=percentage))
    
    # Pasando a array
    limit_points = np.array(limit_point_list)
    
    # Cálculo de la media de los puntos
    mu_c = limit_points.mean()
    
    return limit_points >= mu_c, limit_points


def temporal_correlation_test(H, heart_rate_P, samplerate_signal, 
                              threshold='mean', measure='correlation', 
                              H_binary=True):
    '''Función que permite realizar un testeo de cada componente de la matriz
    H en comparación con el heart rate obtenido a partir de la señal original.
    Retorna un arreglo de booleanos. Si la entrada i es True, se trata de un
    sonido cardíaco. Si es False, se trata de un sonido respiratorio.
    
    Parameters
    ----------
    H : ndarray
        Matriz H de la descomposición NMF.
    heart_rate_P : str
        Señal binaria de la posición de los sonidos cardiacos.
    samplerate_signal : int
        Samplerate de la señal descompuesta en NMF. Puede ser distinta a 
        samplerate_original (por downsampling, por ejemplo).
    threshold : float or "mean" optional
        Valor del umbral de decisión para clasificar una componente de la matriz
        H como sonido cardíaco. Por defecto es "mean".
    measure : {'correlation', 'q_equal'}, optional
        Tipo de métrica para realizar la clasificación. 'correlation' calcula la
        correlación entre el heart rate y el H de cada componente. 'q_equal'
        calcula el porcentaje de puntos iguales en ambas representaciones.
        Por defect es 'correlation'. 
    H_binary : bool, optional
        Booleano que indica si es que el patrón temporal H se considera binario
        (True) o natural (False). Por defecto es True.
    
    Referencias
    -----------
    [1] Canadas-Quesada, et. al (2017). A non-negative matrix factorization 
        approach based on spectro-temporal clustering to extract heart sounds. 
        Applied Acoustics. Elsevier. Chapter 3.2.3.
    '''        
    # Definición de la lista de booleanos de salida
    TC_i_list = list()
    
    for i in range(H.shape[0]):
        # Si es que se quiere hacer en la dimension de P, se debe hacer 
        # un resample
        h_interest = resampling_by_points(H[i], samplerate_signal, 
                                          N_desired=len(heart_rate_P),
                                          normalize=True)
        
        # Aplicando el criterio
        TC_i = _temporal_correlation_criterion(h_interest, heart_rate_P, 
                                               measure=measure, 
                                               H_binary=H_binary)
        # Agregando
        TC_i_list.append(TC_i)
    
    # Pasando a array
    TC_i_array = np.array(TC_i_list)
    
    if threshold == 'mean':
        threshold = np.mean(TC_i_array)
    
    return TC_i_array >= threshold, TC_i_array


def temporal_correlation_test_segment(H, lower, upper, N_fade, N_lax, 
                                      samplerate_signal, threshold=0, 
                                      measure='correlation', H_binary=True):
    '''Función que permite realizar un testeo de cada componente de la matriz
    H en comparación con el heart rate obtenido a partir de la señal original.
    Retorna un arreglo de booleanos. Si la entrada i es True, se trata de un
    sonido cardíaco. Si es False, se trata de un sonido respiratorio.
    
    Parameters
    ----------
    H : ndarray
        Matriz H de la descomposición NMF.
    lower : int
        Índice inferior del segmento.
    upper : int
        Índice superior del segmento.
    N_fade : int
        Cantidad de puntos de fade.
    threshold : float or 'mean', optional
        Valor del umbral de decisión para clasificar una componente de la 
        matriz H como sonido cardíaco. Por defecto es 'mean'.
    measure : {'correlation', 'q_equal'}, optional
        Tipo de métrica para realizar la clasificación. 'correlation' calcula 
        la correlación entre el heart rate y el H de cada componente. 'q_equal'
        calcula el porcentaje de puntos iguales en ambas representaciones.
        Por defect es 'correlation'. 
    H_binary : bool, optional
        Valor del umbral de decisión de la para clasificar una componente como
        sonido cardíaco. Por defecto es 0.
    
    Referencias
    -----------
    [1] Canadas-Quesada, et. al (2017). A non-negative matrix factorization 
        approach based on spectro-temporal clustering to extract heart sounds. 
        Applied Acoustics. Elsevier. Chapter 3.2.3.
    '''
    # Definición del heart rate en ese segmento
    P = np.array([0] * (N_fade + N_lax) + 
                 [1] * abs(upper - lower - 2 * N_lax) + 
                 [0] * (N_fade + N_lax))
        
    # Definición de la lista de booleanos de salida
    TC_i_list = list()
    
    for i in range(H.shape[0]):
        h_interest = resampling_by_points(H[i], samplerate_signal, 
                                          N_desired=len(P),
                                          normalize=True)
        
        # Aplicando el criterio
        TC_i = _temporal_correlation_criterion(h_interest, P, 
                                               measure=measure, 
                                               H_binary=H_binary)
        # Agregando
        TC_i_list.append(TC_i)
    
    # Pasando a array
    TC_i_array = np.array(TC_i_list)
    
    if threshold == 'mean':
        threshold = np.mean(TC_i_array)
    
    return TC_i_array >= threshold, TC_i_array


def machine_learning_clustering(comps, signal_in, samplerate, N_lax, filepath_data, 
                                N=4096, classifier='svm', n_neighbors=1, pca_comps=30,
                                db_basys=1e-12):
    # Se obtienen los segmentos de sonido respiratorio
    with open(f'{filepath_data}', 'r', encoding='utf8') as file:
        intervals = literal_eval(file.readline())
        
    # Definición de la razón entre el samplerate original y el samplerate utilizado
    sr_ratio = 44100 // samplerate
    
    # Definición de la lista de segmentos de sonido respiratorio y cardiaco
    resp_list = list()
    heart_list = list()
    
    # Variable auxiliar que indica el inicio de una señal
    beg = 0
    
    for i in intervals:
        # Definición de los límites a revisar para el sonido respiratorio
        lower_resp = beg // sr_ratio - N_lax if beg != 0 else 0
        upper_resp = i[0] // sr_ratio + N_lax
        
        # Definición de los límites a revisar para el sonido cardiaco
        lower_heart = i[0] // sr_ratio - N_lax
        upper_heart = i[1] // sr_ratio + N_lax
        
        # Segmento de señal respiratoria
        resp_signal = signal_in[lower_resp:upper_resp]
        # Y cardiaca
        heart_signal = signal_in[lower_heart:upper_heart]
                
        # Agregando a las listas
        resp_list.append(resp_signal)
        heart_list.append(heart_signal)
        
        # Se redefine la variable auxiliar
        beg = i[1]
    
    # Finalmente se agrega el segmento final
    resp_signal = signal_in[i[1] // sr_ratio - N_lax:]
    resp_list.append(resp_signal)
    
    # Definición de la matriz a rellenar
    resp_array = np.empty((0, N//2+1))
    heart_array = np.empty((0, N//2+1))
    
    # Re acondicionando las señales
    for i in range(len(resp_list)):
        # Resampleando a 4 veces la cantidad de puntos que se necesita
        resp = resampling_by_points(resp_list[i], samplerate, N)
        resp_array = np.vstack((resp_array, 
                                20 * np.log10(1 / N * abs(np.fft.fft(resp)) 
                                              + db_basys)[:N//2+1]))
    
    for i in range(len(heart_list)):
        # Resampleando a 4 veces la cantidad de puntos que se necesita
        heart = resampling_by_points(heart_list[i], samplerate, N)
        heart_array = np.vstack((heart_array, 
                                 20 * np.log10(1 / N * abs(np.fft.fft(heart)) 
                                               + db_basys)[:N//2+1]))
    
    # Definición de la matriz de entrenamiento
    X_train = np.vstack((resp_array, heart_array))
    
    # Definición de la matriz de etiquetas de entrenamiento. Se define como
    # 0 para respiración y 1 para corazón
    Y_train =  np.array([0] * resp_array.shape[0] +
                        [1] * heart_array.shape[0])
    
    # Reducción de dimensiones
    pca = PCA(n_components=pca_comps)
    X_pca = pca.fit_transform(X_train)
    
    
    for num, i in enumerate(X_pca):
        if Y_train[num] == 0:
            color = 'r'
        elif Y_train[num] == 1:
            color = 'b'
        
        plt.scatter(i[0], i[1], color=color)
    
    
    if classifier == 'knn':
        clas = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='auto')
    elif classifier == 'svm':
        clas = svm.SVC(kernel='linear', degree=50, gamma='auto')
    else:
        raise Exception('Opción "classifier" no válida.')
    
    # Ajuste del clasificador
    clas.fit(X_pca, Y_train)
    
    # Definición de la variable de decisión
    decision = list()
    
    # Para cada componente
    for comp_sound in comps:
        # Para cada iteración, se define nuevamente esta lista auxiliar que sirve para
        # tomar la decisión de la componente i-ésima en base a la mayoría de votos de 
        # las características del sonido cardiorrespiratorio
        decision_comp = list()
        
        for i in intervals:
            # Definición de los límites a revisar para el sonido cardiaco
            lower = i[0] // sr_ratio - N_lax
            upper = i[1] // sr_ratio + N_lax
            
            # Definición del sonido cardiorrespiratorio i en el sonido completo
            hr_sound = resampling_by_points(comp_sound[lower:upper], samplerate, N)
            
            # Se calcula su magnitud
            feat = 20 * np.log10(1 / N * abs(np.fft.fft(hr_sound)) + db_basys)[:N//2+1]
            
            # Y se transforma
            feat = pca.transform(feat.reshape(1,-1))  
            # plt.scatter(feat[:,0], feat[:,1], color='cyan', marker='.')
            
            # Resultado clasificación
            Y_pred = clas.predict(feat)
            
            # Agregando a la lista
            decision_comp.append(Y_pred)
        
        # Una vez completada la lista de decisiones, se procede a contar sus propiedades
        q_hearts = np.sum(decision_comp)
        
        # Si la cantidad de componentes de corazón es mayor al 50%, entonces es corazón,
        # en caso contrario es respiración
        if q_hearts / len(decision_comp) >= 0.5:
            decision.append(True)
        else:
            decision.append(False)
    print(decision)
    # plt.show()
    return decision


def _p_percentage_energy(signal_in, percentage=0.85):
    '''Función que retorna el mínimo índice de un arreglo que cumple que 
    la energía a este ese índica sea mayor que el "x"% de su energía.
    
    Parameters
    ----------
    signal_in : ndarray
        Señal de entrada.
    percentage : float, optional
        Porcentaje de energía límite a evaluar. Por defecto es 85%.
    
    Returns
    -------
    index : int
        Primer índice que cumple el criterio.
    '''
    # Cálculo de la energía total
    total_energy = np.sum(abs(signal_in ** 2))
    
    # Si es que la suma hasta el punto i es mayor que el "x"% de la
    # energía total, se retorna ese primer punto que ya cumple el
    # criterio
    for i in range(len(signal_in)):
        if np.sum(abs(signal_in[:i] ** 2)) >= 0.85 * total_energy:
            return i


def _spectral_correlation_criterion(W_i, W_dic, fcut_bin, measure='cosine',
                                   i_selection='max'):
    '''Función que retorna el valor de verdad de la pertenencia de una componente
    X_i al cluster de sonidos cardíacos, utilizando el criterio de la correlación 
    espectral entre la información de la matriz W de la componente y un diccionario
    preconfigurado de sonido cardíacos. Se escoge el máximo y luego ese valor se 
    compara con un umbral. Retorna True si es que corresponde a sonido cardíaco y 
    False si es que no.
    
    Parameters
    ----------
    W_i : ndarray
        Información espectral de la componente i a partir de la matriz W.
    W_dic : array_like
        Arreglo con las componentes preconfiguradas a partir de sonidos puramente 
        cardíacos externos a esta base de datos.
    fcut_bin : int
        Límite de frecuencia para considerar la medida de relación (en bins). Se
        corta debido a que en general los valores para frecuencias altas son cero,
        se parecen mucho, generando distorsión en las muestras.
    measure : {'cosine', 'correlation'}, optional
        Medida utilizada para calcular la similaridad. 'cosine' usa similaridad de 
        coseno, mientras que 'correlation' calcula el coeficiente de Pearson. Por
        defecto es 'cosine'.
    i_selection : {'max', 'mean'}, optional
        Método de selección del componente que identifica la similarida. Con 'max' 
        se utiliza el máximo de todas las medidas, mientras que con 'mean' se utiliza
        el promedio. Por defecto es 'max'.
    
    Referencias
    -----------
    [1] Canadas-Quesada, et. al (2017). A non-negative matrix factorization 
        approach  based on spectro-temporal clustering to extract heart sounds. 
        Applied Acoustics. Elsevier. Chapter 3.2.1.
    '''    
    # Definición de la lista de valores a guardar
    if measure == 'cosine':
        SC_ij = cosine_similarities(W_dic[:,:fcut_bin], W_i[:fcut_bin])
    elif measure == 'correlation':
        SC_ij = _correlations(W_dic[:,:fcut_bin], W_i[:fcut_bin])
    else:
        raise Exception('Opción para "measure" no válida.')
    
    # Selección del índice
    if i_selection == 'max':
        S_i = max(SC_ij)
    elif i_selection == 'mean':
        S_i = np.mean(SC_ij)
    else:
        raise Exception('Opción para "i_selection" no válida.')
    
    return S_i


def _roll_off_criterion(X, f1, f2, percentage=0.85):
    '''Función que retorna el valor de verdad de la pertenencia de una componente
    X_i al cluster de sonidos cardíacos, utilizando el criterio de la comparación 
    de energía bajo una frecuencia f0 con respecto a la energía total de sí misma. 
    Retorna True si es que corresponde a sonido cardíaco y False si es que no.
    
    Parameters
    ----------
    X : ndarray
        Espectrograma de la componente a clasificar.
    f1 : float
        Valor de la frecuencia de corte inferior en bins.
    f2 : float
        Valor de la frecuencia de corte superior en bins.
    percentage : float, optional
        Ponderación para la energía total del espectrograma en el criterio de
        selección. Por defecto es 0.85.
    
    Referencias
    -----------
    [1] Canadas-Quesada, et. al (2017). A non-negative matrix factorization 
        approach  based on spectro-temporal clustering to extract heart sounds. 
        Applied Acoustics. Elsevier. Chapter 3.2.2.
    '''
    # Propiedad del porcentaje
    percentage = percentage if percentage <= 1 else 1
    
    # Definición del Roll-Off
    ro = np.sum(abs(X[f1:f2,:]) ** 1)
    
    # Definición de la energía del espectrograma
    er = np.sum(abs(X) ** 1)
    
    # Finalmente, se retorna la cualidad de verdad del criterio
    return ro >= percentage * er


def _temporal_correlation_criterion(H_i, P, measure='correlation', H_binary=True, 
                                   show_plot=False):
    '''Función que retorna el valor de verdad de la pertenencia de una componente
    X_i al cluster de sonidos cardíacos, utilizando el criterio de la correlación
    temporal entre la información de la matriz H de la componente i, y el heart 
    rate del sonido cardiorespiratorio original. Retorna True si es que corresponde 
    a sonido cardíaco y False si es que no. 
    
    H_i : ndarray
        Información temporal de la componente i a partir de la matriz H.
    P : ndarray
        Heart rate de la señal cardiorespiratoria.
    measure : {'correlation', 'q_equal'}, optional
        Tipo de métrica para realizar la clasificación. 'correlation' calcula la
        correlación entre el heart rate y el H de cada componente. 'q_equal'
        calcula el porcentaje de puntos iguales en ambas representaciones.
        Por defect es 'correlation'. 
    H_binary : bool, optional
        Booleano que indica si es que el patrón temporal H se considera binario
        (True) o natural (False). Por defecto es True.
    
    Referencias
    -----------
    [1] Canadas-Quesada, et. al (2017). A non-negative matrix factorization 
        approach  based on spectro-temporal clustering to extract heart sounds. 
        Applied Acoustics. Elsevier. Chapter 3.2.3.
    '''
    # Selección del tipo de H a procesar
    if H_binary:
        # Obtener el promedio de la señal
        H_i_mean =  np.mean(H_i)
        # Preprocesando la matriz H_i
        H_in = np.where(H_i >= H_i_mean, 1, 0)
    else:
        H_in = H_i
    
    # Gráfico de P y H_in
    if show_plot:
        plt.plot(P)
        plt.plot(H_in)
        plt.show()
    
    # Selección medida de desempeño
    if measure == 'correlation':
        # Calculando la correlación
        TC_i = _correlation(P, H_in)
    elif measure == 'q_equal':
        TC_i = sum(np.equal(P, H_in)) / len(H_in)
    
    return TC_i


# Módulo de testeo
if __name__ == '__main__':
    pass
