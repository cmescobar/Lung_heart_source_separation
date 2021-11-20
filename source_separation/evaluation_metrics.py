import numpy as np
from scipy import signal
from ss_utils.math_functions import _correlation


def get_PSD(signal_in, samplerate, window='hann', N=2048, noverlap=1024):
    '''Función que obtiene el PSD de una señal a partir del método de Welch.
    
    Parameters
    ----------
    signal_in : ndarray
        Señal a transformar a PSD.
    samplerate : float
        Tasa de muestreo de la señal de entrada.
    window : str or tuple or array, optional
        Revisar función get_window de scipy para más detalles. Por defecto es 'hann'.
    N : float, optional
        Largo de la ventana que se utiliza para calcular la PSD. Por defecto es 2048.
    noverlap : float, optional
        Traslape de las ventanas para calcular la PSD. Por defecto es 1024.
    
    Returns
    -------
    f : ndarray
        Arreglo de frecuencias de la señal (sirve para graficar).
    psd: 
        PSD de la señal.
    
    References
    ----------
    [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
    '''
    return signal.welch(x=signal_in, fs=samplerate, window=window, nperseg=N,
                        noverlap=noverlap)


def SNR(signal_in, noise_in):
    '''Cálculo del SNR de la señal. Puede ser utilizar para calcular la SIR,
    SAR o SDR de una señal.
    
    Parameters
    ----------
    signal_in : ndarray
        Señal de interés.
    noise_in : ndarray
        Señal de ruido o interferencia.
    
    References
    ----------
    [1] E. Vincent, R. Gribonval and C. Fevotte, "Performance measurement in 
        blind audio source separation," in IEEE Transactions on Audio, Speech, 
        and Language Processing, vol. 14, no. 4, pp. 1462-1469, July 2006, 
        doi: 10.1109/TSA.2005.858005.
    '''
    return 10 * np.log10(np.sum(signal_in ** 2) / np.sum(noise_in ** 2))


def PSNR(signal_original, signal_obtained, eps=1):
    '''Obtención del valor de PSNR basado en [1].
    
    Parameters
    ----------
    signal_original : ndarray
        Señal original.
    signal_obtained : ndarray
        Señal recuperada mediante algún método.
        
    References
    ----------
    [1] Al-Naggar, N. Q., & Al-Udyni, M. H. (2019). Performance of Adaptive 
    Noise Cancellation with Normalized Last-Mean-Square Based on the Signal-
    to-Noise Ratio of Lung and Heart Sound Separation. 2019. Journal of 
    Healthcare Engineering.
    '''
    # Definición de la rmse
    rmse = MSE(signal_original, signal_obtained, options='RMSE', eps=eps)
    return 20 * np.log10(max(signal_original) / rmse)


def SIR(X_list, S, interest_index):
    '''Cálculo de la razón señal a interferencia. Función adaptada para
    la descomposición NMF.
    
    Parameters
    ----------
    X_list : list
        Lista de espectrogramas para cada componente de interés.
    S : ndarray
        Espectrograma de la señal a descompuesta.
    interest_index : int
        Índice de la señal de interés para el cálculo de la SIR.
        
    References
    ----------
    [1] E. Vincent, R. Gribonval and C. Fevotte, "Performance measurement in 
        blind audio source separation," in IEEE Transactions on Audio, Speech, 
        and Language Processing, vol. 14, no. 4, pp. 1462-1469, July 2006, 
        doi: 10.1109/TSA.2005.858005.
    [2] G. Shah and C. B. Papadias, "Blind recovery of cardiac and respiratory 
        sounds using non-negative matrix factorization & time-frequency masking”. 
        13th IEEE International Conference on BioInformatics and BioEngineering, 
        Chania, 2013.
    [3] G. Shah and C. Papadias, “Separation of cardiorespiratory sounds using 
        time-frequency masking and sparsity”. 2013 18th International Conference 
        on Digital Signal Processing (DSP).
    '''
    # Máscara de la señal de interés
    mask_i = np.divide(abs(X_list[interest_index]), abs(S) + 1e-12)
    
    # Componentes enmascaradas por la máscara de la señal de interés
    masked_list = np.array([(mask_i * abs(X_i)) 
                            for i, X_i in enumerate(X_list) 
                            if i != interest_index])
    
    return 10 * np.log10(np.sum(abs(X_list[interest_index]) ** 2) / 
                         np.sum(masked_list ** 2))
 

def SDR(signal_original, signal_obtained):
    '''Función que calcula la razón señal a distorsión de una señal.
    
    Parameters
    ----------
    signal_original : ndarray
        Señal original.
    signal_obtained : ndarray
        Señal recuperada mediante algún método.
    
    References
    ----------
    [1] E. Vincent, R. Gribonval and C. Fevotte, "Performance measurement in 
        blind audio source separation," in IEEE Transactions on Audio, Speech, 
        and Language Processing, vol. 14, no. 4, pp. 1462-1469, July 2006, 
        doi: 10.1109/TSA.2005.858005.
    '''
    return 10 * np.log10(np.sum(signal_original ** 2) / \
                         np.sum((signal_obtained - signal_original) ** 2))


def MSE(signal_original, signal_obtained, options='MSE', scale='abs', eps=1):
    '''Cálculo de el error cuadrático medio entre 2 señales.
    
    Parameters
    ----------
    signal_original : ndarray
        Señal original.
    signal_obtained : ndarray
        Señal recuperada mediante algún método.
    options : {'MSE', 'RMSE', 'NMSE'}, optional
        Tipo de MSE calculado. "MSE" es el clásico cálculo de MSE. "RMSE" es
        el cálculo de MSE con raíz cuadrada. "NMSE" es el cálculo de MSE normalizado
        por la señal de entrada. Por defecto es "MSE".
    scale : {'abs', 'dB'}, optional
        Escala de retorno del valor. "abs" retorna el valor directo. "dB" retorna
        el valor en decibeles. Por defecto es "abs".
    eps : float, optional
        Valor de epsilon utilizado en la diferencia de cálculo de MSE. Por defecto 
        es 1.
    
    References
    ----------
    [1] Al-Naggar, N. Q., & Al-Udyni, M. H. (2019). Performance of Adaptive 
        Noise Cancellation with Normalized Last-Mean-Square Based on the Signal-
        to-Noise Ratio of Lung and Heart Sound Separation. 2019. Journal of 
        Healthcare Engineering.
    [2] https://en.wikipedia.org/wiki/Root-mean-square_deviation
    '''
    # Propiedad de eps
    if eps not in [-1, 1]:
        raise Exception('"eps" solo puede tomar valores 1 y -1.')
    
    # Definición de la escala
    if scale == 'abs':
        f = lambda x: x
        
    elif scale == 'dB':
        f = lambda x: 20 * np.log10(x)
    
    else:
        raise Exception('Parámetro "scale" no está bien definido.')
    
    # Definición del MSE de salida
    if options == 'MSE':
        return f(np.mean(abs(signal_original - eps * signal_obtained) ** 2))
    
    elif options == 'RMSE':
        return f(np.sqrt(np.mean(abs(signal_original - eps * signal_obtained) ** 2)))
    
    elif options == 'NMSE':
        return f(np.sum(abs(signal_original - eps * signal_obtained) ** 2) / \
                 np.sum(signal_original ** 2))
    
    else:
        raise Exception('Parámetro "options" no está bien definido.')
    

def HNRP(signal_original, signal_obtained):
    '''Cálculo del "Heart Noise Reduction Percentage". 
    
    Parameters
    ----------
    signal_original : ndarray
        Señal original (sin separación o denosing).
    signal_obtained : ndarray
        Señal reconstruida u obtenida mediante algún algoritmo.
    
    References
    ----------
    [1] Mondal, A., Bhattacharya, P. S., & Saha, G. (2011). Reduction of heart 
        sound interference from lung sound signals using empirical mode 
        decomposition technique. Journal of Medical Engineering & Technology.
    [2] C. Lin and E. Hasting, "Blind source separation of heart and lung sounds 
        based on nonnegative matrix factorization" 2013 International Symposium 
        on Intelligent Signal Processing and Communication Systems.
    [3] Canadas-Quesada, F. J., Ruiz-Reyes, N., Carabias-Orti, J., Vera-Candeas, 
        P., & Fuertes-Garcia, J. (2017). A non-negative matrix factorization 
        approach based on spectro-temporal clustering to extract heart sounds. 
        Applied Acoustics. Elsevier.
    '''
    mixed = np.mean(signal_original ** 2)
    free = np.mean(signal_obtained ** 2)
    return abs(mixed - free) / mixed


def performance_HNRP(hnrp, signal_hr, signal_heart):
    '''Medida de desempeño de una atenuación de sonido cardíaco en base al valor
    de HNRP.
    
    Parameters
    ----------
    hnrp : float
        Valor de HNRP calculado.
    signal_hr: ndarray
        Señal cardiorespiratoria correspondiente (original).
    signal_heart : ndarray
        Señal cardiaca correspondiente (original).
        
    References
    ----------
    [1] C. Lin and E. Hasting, "Blind source separation of heart and lung sounds 
        based on nonnegative matrix factorization" 2013 International Symposium 
        on Intelligent Signal Processing and Communication Systems.
    [2] Canadas-Quesada, F. J., Ruiz-Reyes, N., Carabias-Orti, J., Vera-Candeas, 
        P., & Fuertes-Garcia, J. (2017). A non-negative matrix factorization 
        approach based on spectro-temporal clustering to extract heart sounds. 
        Applied Acoustics. Elsevier.
    '''
    # Porcentaje de presencia del sonido cardíaco en el sonido cardiorespiratorio.
    b = np.sum(signal_heart ** 2) / np.sum(signal_hr ** 2)
    
    return 1 - abs(b - hnrp) / b


def get_correlation(signal_original, signal_obtained):
    return _correlation(signal_original, signal_obtained)


def psd_correlation(signal_original, signal_obtained, samplerate, window='hann', 
                    N=2048, noverlap=1024):
    # Cálculo de las PSD
    _, psd_ori = get_PSD(signal_original, samplerate, window=window, N=N,
                         noverlap=noverlap)
    _, psd_obt = get_PSD(signal_obtained, samplerate, window=window, N=N,
                         noverlap=noverlap)
    
    # Calcular la correlación entre las psd en dB
    return _correlation(20 * np.log10(psd_ori), 20 * np.log10(psd_obt))