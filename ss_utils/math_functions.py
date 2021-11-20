import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance


def hamming_window(N):
    # Definición de la ventana hamming de modo que se pueda generar para un
    # largo de ventana definido
    return np.asarray([0.53836 - 0.46164 * np.cos((2 * np.pi * i)/N)
                       for i in range(int(N))])


def hann_window(N):
    # Definición de la ventana hamming de modo que se pueda generar para un
    # largo de ventana definido
    return np.asarray([0.5 - 0.5 * np.cos((2 * np.pi * i)/N)
                       for i in range(int(N))])


def recognize_peaks_by_derivates(x, signal, peak_type='min', tol_dx=0.01,
    tol_d2x=1e-2, lookup=1500, plot=False):
    '''Función que permite detectar peaks de una señal.

    Parameters
    ----------
    x : ndarray or list
        Unidad en el eje independiente.
    signal : ndarray or list
        Señal de entrada.
    peak_type : {'min', 'max', 'all'}, optional
        Definición del evento a detectar: mínimos ('min'), máximos ('max'),
        o ambos ('all'). Por defecto es 'min'.
    tol_dx : float, optional
        Umbral de tolerancia para definir un peak en base a la derivada.
        Por defecto es 0.01.
    tol_d2x : float, optional
        Umbral de tolerancia para definir la naturaleza de un peak en base a 
        la segunda derivada. Por defecto es 0.01.
    lookup : int, optional
        Cantidad de puntos a revisar en el entorno para asegurar el peak. Por
        defecto es 1500.
    plot : bool, optional
        Booleano que indica si se plotean los peaks. Por defecto es False.
    
    Returns
    -------
    out_indexes: list
        Posiciones de los peaks.
    '''
    # Se definen las derivadas 
    dx = np.gradient(signal, x)
    d2x = np.gradient(dx, x)
    
    # Buscando los puntos donde la derivada se vuelve cero
    der_vect_0 = [i for i in range(len(dx)) if abs(dx[i]) <= tol_dx]
    
    # Y definiendo si estos puntos corresponden a mínimos o máximos se realiza
    if peak_type == 'min':
        sel_indexes = [i for i in der_vect_0 if d2x[i] >= tol_d2x]
    elif peak_type == 'max':
        sel_indexes = [i for i in der_vect_0 if d2x[i] <= - tol_d2x]
    elif peak_type == 'all':
        sel_indexes = der_vect_0
    else:
        raise ValueError('La opcion de eleccion de peak utilizada no es valida.')
    
    # Seleccionando un punto característico de la región (ya que
    # muchos de los "puntos" aparecen agrupados en más puntos). En primer lugar,
    # se obtiene un vector de diferencias para conocer los puntos en los que se
    # pasa de un cluster a otro
    dif_indexes = [i + 1 for i in range(len(sel_indexes) - 1)
                   if sel_indexes[i + 1] - sel_indexes[i] > 1] + \
                  [len(sel_indexes) + 1]

    # Separando los clusters de puntos y encontrando el índice representativo de
    # cada uno
    begin = 0
    out_indexes = []
    for i in dif_indexes:
        # Definición del punto posible. Se hace round en caso de que sea un
        # decimal, e int para pasarlo si o si a un elemento tipo "int" para
        # indexar 
        possible_point = int(round(np.mean(sel_indexes[begin:i])))
        
        # Finalmente, se debe reconocer si este punto es realmente un mínimo o
        # un  máximo y no un punto de inflexión. Para ello se revisará en un
        # rango de 'lookup' alrededor de este punto. Definiendo los puntos a
        # revisar 
        look_before = signal[possible_point - lookup] \
            if possible_point - lookup >= 0 else signal[0]
        look_after  = signal[possible_point + lookup] \
            if possible_point + lookup <= len(signal) else signal[len(signal)-1]

        # Luego, realizando la comparación
        if peak_type == 'min':
            # Corroborando que alrededor de este punto se forma un "valle"
            if (look_after > signal[possible_point] and 
                look_before > signal[possible_point]):
                out_indexes.append(possible_point)

        elif peak_type == 'max':
            # Corroborando que alrededor de este punto se forma una "cueva"
            if (look_after < signal[possible_point] and 
                look_before < signal[possible_point]):
                out_indexes.append(possible_point)
        
        elif peak_type == 'all':
            # Corroborando alguno de los 2 casos anteriores
            if (look_after > signal[possible_point] and 
                look_before > signal[possible_point]) or \
               (look_after < signal[possible_point] and 
                look_before < signal[possible_point]):
                out_indexes.append(possible_point)

        # Redefiniendo el comienzo del análisis
        begin = i
    
    # Graficando para corroborar visualmente
    if plot:
        plt.subplot(3,1,1)
        plt.plot(signal)
        plt.plot(out_indexes, [signal[i] for i in out_indexes], 'rx')

        plt.subplot(3,1,2)
        plt.plot(dx)

        plt.subplot(3,1,3)
        plt.plot(d2x)

        plt.show()

    return out_indexes


def wiener_filter(X, WiHi, W, H, alpha=1, div_basys=1e-15):
    '''Aplicación de filtro de Wiener para las componentes obtenidas a 
    partir de la descomposición NMF. Está dada por:
    M_i = (WiHi) ** a / (sum_{i} (WiHi) ** a)
    
    Parameters
    ----------
    X : ndarray
        Señal a descomponer mediante NMF.
    WiHi : ndarray
        Componente i de la descomposición NMF.
    W : ndarray
        Matriz que contiene la información espectral de las componentes.
    H : ndarray
        Matriz que contiene la información temporal de las componentes.
    alpha : int, optional
        Exponente utilizado para cada componente. Por defecto es 1.
    div_basys : float, optional
        Valor base utilizado en la división (para evitar división por cero).
        Por defecto es 1e-15.
    '''
    # Definición del WH
    WH_alpha = np.zeros(X.shape)
    
    for i in range(W.shape[1]):
        WH_alpha += np.outer(W[:,i], H[i]) ** alpha
        
    # Obteniendo la máscara
    mask = np.divide(WiHi ** alpha, WH_alpha + div_basys)
    
    # Aplicando la máscara al espectrograma original, se obtiene el resultado
    # final del proceso de separación de fuentes
    return mask * X


def SNR(signal_in, signal_denoised, snr_type='db'):
    '''Función que retorna el SNR de una señal de entrada en comparación con
    la señal limpia.

    Paramters
    ---------
    signal_in : ndarray
        Señal de entrada.
    signal_denoised : ndarray
        Señal base.
    snr_type : 'db'
        Unidad en la que se define el SNR. POR COMPLETAR MÁS OPCIONES.

    Returns
    -------
    snr : float
        SNR de la señal de entrada.
    '''
    if snr_type == 'db':
        return 10 * np.log10(sum(signal_in ** 2) / 
                             sum((signal_in - signal_denoised) ** 2))


def moving_average(signal_in, Lf):
    '''Función que permite hacer una media móvil de una señal.
    
    Parameters
    ----------
    signal_in : ndarray or list
        Señal de entrada.
    Lf : int
        Largo de la ventana a considerar.
        
    Returns
    -------
    result : ndarray
        Señal de salida.
    '''
    # Definición de N
    N = len(signal_in)
    # Creación del vector del resultado
    result = np.zeros(N)
    
    # Se hace el promedio para cada segmento
    for n in range(N):
        if 0 <= n <= Lf - 1:
            result[n] = np.divide(sum(signal_in[:n+Lf+1]), Lf + n + 1)
        elif Lf <= n <= N - Lf - 1:
            result[n] = np.divide(sum(signal_in[n-Lf:n+Lf+1]), 2*Lf + 1)
        elif N - Lf <= n <= N - 1:
            result[n] = np.divide(sum(signal_in[n-Lf:N]), Lf + N - 1)
            
    return result


def raised_cosine_modified(N, beta):
    '''Creación de una ventana tipo pulso coseno elevado.
    
    Parameters
    ----------
    N : int
        Cantidad de puntos de la ventana.
    beta : float
        Parámetro de la función coseno elevado para la apertura de la ventana.
        
    Returns
    -------
    rc_out : ndarray
        Ventana pulso coseno elevado de N puntos con el valor de beta ingresado
    '''
    # Definición de la frecuencia f
    f = np.linspace(-1/2, 1/2, N)
    
    # Control de parámetro para beta
    if beta <= 0:
        beta = 0
    elif beta >= 1:
        beta = 1
    
    # Definición del vector de salida
    rc_out = np.array([])
    
    # Para punto f
    for i in f:
        if abs(i) <= (1 - beta)/2:
            rc_out = np.concatenate((rc_out, [1]))
        elif (1 - beta)/2 < abs(i) <= (1 + beta)/2:
            to_append =  np.cos(np.pi / beta * (abs(i) - (1 - beta)/2))
            rc_out = np.concatenate((rc_out, [to_append]))
        else:
            rc_out = np.concatenate((rc_out, [0]))
            
    return rc_out


def raised_cosine_fading(N, beta, side='right'):
    ''' Creacion de una ventana de desvanecimiento basada en coseno elevado.
    
    Parameters
    ----------
    N : int
        Cantidad de puntos de la ventana.
    beta : float
        Parámetro de la función coseno elevado para la apertura de la ventana.
    side : {'left', 'right'}, optional
        Dirección en la cual se puede usará la ventana. Se recomienda 'right' para
        el final de la señal y 'left' para el comienzo. Por defecto es 'right'.
    
    Returns
    -------
    vanish_window : ndarray
        Ventana de desvanecimiento de N puntos.
    '''    
    # Definición de la frecuencia f
    f = np.linspace(-1, 1, 2*N)
    
    # Control de parámetro para beta
    if beta <= 0:
        beta = 0
    elif beta >= 1:
        beta = 1
    
    # Definición del vector de salida
    rc_out = np.array([])
    
    # Para punto f
    for i in f:
        if abs(i) <= (1 - beta)/2:
            rc_out = np.concatenate((rc_out, [1]))
        elif (1 - beta)/2 < abs(i) <= (1 + beta)/2:
            to_append =  1/2 * (1 + np.cos(np.pi / beta * (abs(i) - (1 - beta)/2)))
            rc_out = np.concatenate((rc_out, [to_append]))
        else:
            rc_out = np.concatenate((rc_out, [0]))
    
    # Selección del lado
    if side == 'right':
        vanish_window = rc_out[N:]
    elif side == 'left':
        vanish_window = 1 - rc_out[N:]
    
    return vanish_window


def db_coef(db):
    '''Función que obitene el coeficiente por el cual se debe multiplicar un arreglo
    para obtener el valor de decibel deseado (relativo).

    Parameters
    ----------
    db : float
        Valor de dB deseado para realizar una transformación.

    Returns
    -------
    db_value : float
        Valor por el que se debe multiplicar un arreglo para obtener el decibel 
        deseado.
    '''
    return 10 ** (db/20)


def db_attenuation(signal_in, db):
    '''Función que permite atenuar una señal a partir de su valor en dB

    Parameters
    ----------
    signal_in : ndarray
        Señal a atenuar.
    dB : float
        Valor de atenuación en dB (positivo para atenuar).

    Returns
    -------
    signal_attenuated : ndarray
        Señal atenuada en db dB.
    '''
    return signal_in * db_coef(-db)
    

def _correlation(a, b):
    '''Función de correlación entre 2 series temporales.
    
    Parameters
    ----------
    a , b : ndarray
        Series de entrada.
    
    Returns
    -------
    r : float
        Correlación entre las 2 entradas, dadas por:
        1 / (N - 1) * np.sum((a - mu_a) * (b - mu_b)) / (sig_a * sig_b)
        
    Referencias
    -----------
    [1] https://en.wikipedia.org/wiki/Correlation_and_dependence
    '''
    # Definición de la cantidad de puntos
    N = len(a)
    
    # Cálculo de la media de ambas series
    mu_a = np.mean(a)
    mu_b = np.mean(b)
    
    # Cálculo de la desviación estándar de ambas series
    sig_a = np.std(a)
    sig_b = np.std(b)
    
    # Definición de correlación
    r =  1 / (N - 1) * np.sum((a - mu_a) * (b - mu_b)) / (sig_a * sig_b)
    
    # Propiedad de límite para r    
    r = r if r <= 1.0 else 1.0
    r = r if r >= -1.0 else -1.0

    return r


def _correlations(A, b):
    '''Función de correlación entre 2 series temporales, en donde A es una
    matriz de series temporales.
    
    Parameters
    ----------
    A : ndarray
        Matriz de series.
    b : ndarray
        Serie de entrada.
    
    Returns
    -------
    r : float
        N correlaciones entre las 2 entradas, dadas por:
        1 / (N - 1) * np.sum((a - mu_a) * (b - mu_b)) / (sig_a * sig_b)
        
        En donde a corresponde a cada fila de A.
        
    Referencias
    -----------
    [1] https://en.wikipedia.org/wiki/Correlation_and_dependence
    '''
    # Definición de la cantidad de puntos
    if A.shape[1] == len(b):
        N = A.shape[1]
    else:
        raise Exception(f'Dimensiones entre A ({A.shape}) y b ({b.shape}) no '
                        f'coinciden.')
    
    # Cálculo de la media de ambas series
    mu_a = A.mean(axis=1)
    mu_b = b.mean()
    
    # Cálculo de la desviación estándar de ambas series
    sig_a = A.std(axis=1)
    sig_b = b.std()
    
    # Definición de correlación
    r =  1 / (N - 1) * np.sum((A.T - mu_a).T * (b - mu_b), axis=1) / (sig_a * sig_b)
    
    # Propiedad de límite para r    
    r = np.where(r >= 1, 1, r)
    r = np.where(r <= -1, -1, r)

    return r


def cosine_similarity(a, b):
    '''Similitud coseno entre un vector a y b.
    
    Parameters
    ----------
    a, b : array_shape
        Entradas a comparar.
    
    Returns
    -------
    cos_sim : float
        Similitud de coseno.
    
    References
    ----------
    [1] https://en.wikipedia.org/wiki/Cosine_similarity
    [2] https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/
        scipy.spatial.distance.cosine.html
    '''
    return 1 - distance.cosine(a, b)


def cosine_similarities(A, b):
    '''Similitud coseno entre un vector A y b. En este caso A puede
    ser una matriz de dimensión (n x m) y b siempre es de dimensión
    m. Si A es una matriz, se retorna un arreglo de dimensión n 
    (similitud de coseno entre b y cada una de las filas de A).
    
    Parameters
    ----------
    A : ndarray
        Matriz de series.
    b : ndarray
        Serie de entrada.
    
    Returns
    -------
    cos_sim : ndarray or float
        Similitud de coseno.
    
    References
    ----------
    [1] https://en.wikipedia.org/wiki/Cosine_similarity
    '''
    norm_a = np.sqrt(np.sum(A ** 2, axis=1))
    norm_b = np.sqrt(np.sum(b ** 2))
    
    # Definición del numerador
    num = np.sum((A * b), axis=1)
    den = norm_a * norm_b
    
    return num / den 
