import pywt
import numpy as np
from PyEMD import EMD, EEMD
from scipy import signal, stats
from ss_utils.descriptor_functions import get_spectrogram, get_inverse_windowed_signal
from ss_utils.filter_and_sampling import lowpass_filter, highpass_filter


def homomorphic_filter(signal_in, samplerate, cutoff_freq=10, delta_band=5, 
                       filter_type='lowpass', epsilon=1e-10):
    '''Función que retorna la salida de un filtro homomórfico para una señal de entrada. 
    Esta representación busca modelar la envolvente de la señal de interés.
    
    Parameters
    ----------
    signal_in : ndarray or list
        Señal de entrada.
    samplerate : int
        Tasa de muestreo de la señal de entrada.
    cutoff_freq : int, optional
        Frecuencia de corte del filtro. Por defecto es 100.
    delta_band : int, optional
        Ancho de banda de transición entre los puntos del filtro. Por defecto es 50.
    filter_type : {"lowpass", "highpass"}, optional
        Tipo de filtro a aplicar. "lowpass" para pasa bajos y "highpass" para pasa 
        altos. Por defecto es "lowpass".
    epsilon: float, optional
        Valor que se suma al cálculo de logaritmo para evitar problemas de indefinición.
        Por defecto es 1e-10.
        
    Returns
    -------
    signal_out : ndarray
        Señal de salida del filtro homomórfico, correspondiente a la envolvente (en caso 
        de elegir filter_type="lowpass") o a la componente de alta frecuencia (en caso 
        de elegir filter_type="highpass")
    
    References
    ----------
    [1] Gill, D., Gavrieli, N., & Intrator, N. (2005, September). Detection and 
        identification of heart sounds using homomorphic envelogram and 
        self-organizing probabilistic model. In Computers in Cardiology, 
        2005 (pp. 957-960). IEEE.
    [2] Schmidt, S. E., Holst-Hansen, C., Graff, C., Toft, E., & Struijk, J. J. (2010). 
        Segmentation of heart sound recordings by a duration-dependent hidden Markov 
        model. Physiological measurement, 31(4), 513.
    
    '''
    # Se toma el logaritmo de la señal de entrada para separar la modulación en suma
    log_signal = np.log(abs(signal_in) + epsilon)
    
    # Se aplica el filtro
    if filter_type == 'lowpass':
        _, log_filt = lowpass_filter(log_signal, samplerate, freq_pass=cutoff_freq, 
                                     freq_stop=cutoff_freq + delta_band)
    elif filter_type == 'highpass':
        _, log_filt = highpass_filter(log_signal, samplerate, freq_pass=cutoff_freq, 
                                      freq_stop=cutoff_freq + delta_band)
    else:
        raise Exception('Opción filter_type inválida. Use "lowpass" o "highpass".')
    
    # Y se retorna desde el logaritmo
    return np.exp(log_filt)


def shannon_envolve(signal_in, alpha=2):
    '''Función que calcula la envolvente dada por la envolvente de Shannon.
    
    Parameters
    ----------
    signal_in : ndarray or list
        Señal de entrada.
    alpha : float, optional
        Exponente al que se elevan los argumentos de la expresión de Shannon. 
        Si es 1, se calcula la entropía de Shannon. Si es 2 se calcula la energía 
        de Shannon. Si es 1.5 se calcula una expresión empírica planteada en [2]. 
        Por defecto es 2.
    
    Returns
    -------
    signal_out : ndarray
        Señal escalada por la envolvente de energía de Shannon.
    
    References
    ----------
    [1] Gill, D., Gavrieli, N., & Intrator, N. (2005, September). Detection and 
        identification of heart sounds using homomorphic envelogram and 
        self-organizing probabilistic model. In Computers in Cardiology, 
        2005 (pp. 957-960). IEEE.
    [2] Moukadem, A., Schmidt, S., & Dieterlen, A. (2015). High order statistics 
        and time-frequency domain to classify heart sounds for subjects under 
        cardiac stress test. Computational and mathematical methods in medicine, 
        2015.
    '''
    return - (signal_in ** alpha) * np.log(signal_in ** alpha)


def hilbert_representation(signal_in, samplerate):
    '''Obtención de la transformada de Hilbert de la señal, a través de la cual 
    es posible representar la "señal analítica". Retorna la señal analítica,
    la fase instantánea y la frecuencia instantánea.
    
    Parameters
    ----------
    signal_in : ndarray or list
        Señal de entrada.
    samplerate : int
        Tasa de muestreo de la señal de entrada.
    
    Returns
    -------
    analytic_function : ndarray
        Función analítica obtenida a partir de la suma de la señal original en el 
        eje real y la transformada de Hilbert en el eje imaginario.
    instantaneous_phase : ndarray
        Fase instantánea obtenida a partir del arctan(.) de la razón entre la 
        transformada de Hilbert y la señal original, la cual está relacionada por
        la función analítica.
    instantaneous_frequency : ndarray
        Frecuencia instantánea obtenida a partir de la fase instantánea, la cual
        a su vez se calcula como la fase de la señal analítica.
    
    References
    ----------
    [1] Varghees, V. N., & Ramachandran, K. I. (2017). Effective heart sound 
        segmentation and murmur classification using empirical wavelet transform 
        and instantaneous phase for electronic stethoscope. IEEE Sensors Journal, 
        17(12), 3861-3872.
    [2] Choi, S., & Jiang, Z. (2008). Comparison of envelope extraction algorithms 
        for cardiac sound signal segmentation. Expert Systems with Applications, 
        34(2), 1056-1069.
    [3] Varghees, V. N., & Ramachandran, K. I. (2014). A novel heart sound activity 
        detection framework for automated heart sound analysis. Biomedical Signal 
        Processing and Control, 13, 174-188.
    '''
    # Obtener la transformada de hilbert de la señal
    analytic_function = signal.hilbert(signal_in)
    
    # Definición de la fase instantánea
    instantaneous_phase = np.unwrap(np.angle(analytic_function))
    
    # Definición de la frecuencia instantánea
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)
    
    return analytic_function, instantaneous_phase, instantaneous_frequency


def emd_decomposition(signal_in, samplerate, max_imf=-1, decomposition_type='EMD'):
    '''Función que permite descomponer en modos la señal mediante el algoritmo de 
    Empirical Mode Decomposition (EMD). Cada una de las funciones de salida se 
    denominan funciones de modo intrínseco (IMFs).
    
    Parameters
    ----------
    signal_in : ndarray or list
        Señal de entrada.
    samplerate : int
        Tasa de muestreo de la señal de entrada.
    max_imf : int
        Cantidad máxima de niveles a descomponer. Por defecto es -1 (todos los que 
        se puedan). 
    decomposition_type : {"EMD", "EEMD"}, optional
        Método de descomposición. Por defecto es "EMD".
    
    Returns
    -------
    imfs : ndarray
        Arreglo de IMF's, ordenados desde el primer nivel (índice 0) hasta el 
        último nivel posible (índice -1).
    
    References
    ----------
    [1] Tseng, Y. L., Ko, P. Y., & Jaw, F. S. (2012). Detection of the third 
        and fourth heart sounds using Hilbert-Huang transform. Biomedical 
        engineering online, 11(1), 8.
    '''
    if decomposition_type == 'EMD':
        # Definición del objeto EMD
        emd_machine = EMD()
        
        # Cálculo de la EMD
        imfs = emd_machine.emd(signal_in, max_imf=max_imf)
    
    elif decomposition_type == 'EEMD':
        # Definición del objeto EMD
        emd_machine = EEMD()
        
        # Cálculo de la EMD
        imfs = emd_machine.eemd(signal_in, max_imf=max_imf)
    
    else:
        raise Exception('Opción "decomposition_type" inválida.')
    
    return imfs


def simplicity_based_envelope(signal_in, N=64, noverlap=32, m=10, tau=2000):
    '''Función que calcula la envolvente de simplicidad de la señal, basado en
    teoría de sistemas. El método consiste en la obtención de un vector X de m 
    delays (tau), al cual se le calcula la matriz de correlación C = X^T.X. 
    
    Esta matriz de correlación es descompuesta en valores singulares (SVD), los
    cuales son indicadores de "regularidad" de la señal. Si es que pocos valores 
    singulares son altos, entonces la señal es regular. En cambio si todos tienen
    valores similares, la señal será caótica. 
    
    Por ende, se calcula la entropía de los valores propios para expresar esta 
    noción en un único indicador.
    
    Finalmente el término de simplicidad está dado por la expresión:
    simplicity = 1 / (2 ** H)
    
    Donde H corresponde a la entropía de los valores propios.
    
    Parameters
    ----------
    signal_in : ndarray or list
        Señal de entrada.
    N : int, optional
        Tamaño de la ventana de análisis de la señal. Por defecto es 64.
    noverlap : int, optional
        Cantidad de puntos de traslape que se utiliza para calcular la STFT. Por 
        defecto es 32.
    m : int, optional
        Cantidad de términos de delay a considerar para la construcción de la 
        matriz X. Por defecto es 10.
    tau : int, optional
        Delay de entre cada uno de los puntos para la costrucción de la matriz 
        X. Por defecto es 2000.
    
    Returns
    -------
    simplicity_out : ndarray
        Vector de simplicidad calculado para cada una de las ventanas de la 
        señal original.
    
    References
    ----------
    [1] Nigam, V., & Priemer, R. (2005). Accessing heart dynamics to estimate 
        durations of heart sounds. Physiological measurement, 26(6), 1005.
    [2] Kumar, D., Carvalho, P. D., Antunes, M., Henriques, J., Maldonado, M., 
        Schmidt, R., & Habetha, J. (2006, September). Wavelet transform and 
        simplicity based heart murmur segmentation. In 2006 Computers in 
        Cardiology (pp. 173-176). IEEE.
    [3] Vepa, J., Tolay, P., & Jain, A. (2008, March). Segmentation of heart 
        sounds using simplicity features and timing information. In 2008 IEEE 
        International Conference on Acoustics, Speech and Signal Processing 
        (pp. 469-472). IEEE.
    '''
    # Definición del vector de salida
    simplicity_out = list()
    
    while signal_in.any():
        # Se corta la cantidad de muestras que se necesite, o bien, las que se
        # puedan cortar
        if len(signal_in) >= N:
            q_samples = N
            hop = int(N - noverlap)
        else:
            q_samples = hop = len(signal_in)

        # Recorte en la cantidad de muestras
        signal_frame = signal_in[:q_samples]
        signal_in = signal_in[hop:]
    
        # Definición de la cantidad de vectores P a utilizar
        P = q_samples - (m - 1) * tau
        
        # Definición de la matriz X
        X = np.empty((P, m))
        
        # Obtención de cada fila de la matriz X
        for i in range(P):
            X[i] = signal_frame[i:(i + m * tau):tau]
        
        # Ponderación por P
        X *= 1 / np.sqrt(P)
        
        # Multiplicación de X para la obtención de la matriz de correlación
        C = np.matmul(X.T, X)
        
        # Descomposición SVD de la matriz de correlación
        lambd = np.linalg.svd(C, compute_uv=False)
        
        # Normalización de los lambda
        lambd = lambd / sum(lambd)
        
        # Cálculo de la entropía
        H_i = - sum(lambd * np.log(lambd))
        
        # Calculando la simplicidad 
        simplicity = 1 / (2 ** H_i)
        
        # Y agregando a la lista de salida
        simplicity_out.append(simplicity)
        
    return np.array(simplicity_out)


def variance_fractal_dimension(signal_in, samplerate, NT=1024, noverlap=512, 
                               kmin=4, kmax=4, step_size_method='unit'):
    '''Variance fractal dimension está dada por la expresión:
    D_o = D_E + 1 - H

    Donde D_E corresponde a la dimensión de la señal a revisar (por
    ejemplo, en el caso de una curva D_E = 1, para un plano D_E = 2 y 
    para el espacio D_E = 3) y donde:
        H = lim_{dt -> 0} 1/2 * log(var(ds)) / log(dt)
    
    En el que 's' es la señal muestreada y 'ds' la variación entre 2 
    puntos. Asi mismo, 'dt' es la diferencia entre 2 puntos.
    
    Parameters
    ----------
    signal_in : ndarray or list
        Señal de entrada.
    samplerate : int
        Tasa de muestreo de la señal de entrada.
    NT : int
        Tamaño de la ventana de análisis de la señal.
    noverlap : int
        Cantidad de puntos de traslape entre ventanas sucesivas.
    kmin : int, optional
        Cantidad mínima de iteraciones k a considerar para la función de escalas 
        por sub-ventanas, n_k. Por defecto es 4.
    kmax : int, optional
        Cantidad máxima de iteraciones k a considerar para la función de escalas 
        por sub-ventanas, n_k. Por defecto es 4.
    step_size_method : {"unit", "dyadic"}, optional
        Definición del tipo de función de escalas n_k. "unit" para n_k = k y 
        "dyadic" para n_k = k ** 2. Por defecto es "unit".
    
    Returns
    -------
    vfdt : ndarray
        Arreglo que contiene la Variance Fractal Dimension (VFD) a lo 
        largo del tiempo.
    
    References
    ----------
    [1] Phinyomark, A., Phukpattaranont, P., & Limsakul, C. (2014). 
        Applications of variance fractal dimension: A survey. Fractals, 
        22(01n02), 1450003.
    [2] Gnitecki, J., & Moussavi, Z. (2003, September). Variance fractal 
        dimension trajectory as a tool for hear sound localization in lung 
        sounds recordings. In Proceedings of the 25th Annual International 
        Conference of the IEEE Engineering in Medicine and Biology Society 
        (IEEE Cat. No. 03CH37439) (Vol. 3, pp. 2420-2423). IEEE.
    [3] Carvalho, P., Gilt, P., Henriques, J., Eugénio, L., & Antunes, M. 
        (2005, September). Low complexity algorithm for heart sound 
        segmentation using the variance fractal dimension. In IEEE 
        International Workshop on Intelligent Signal Processing, 2005. 
        (pp. 194-199). IEEE.
    '''
    # Definición del vector d_sigma
    d_sigma = []
    
    # Definición de función de step
    if step_size_method == 'unit':
        step_f = lambda k: k
    elif step_size_method == 'dyadic':
        step_f = lambda k: 2 ** k
    else:
        raise Exception('Opción "step_size_method" no valida.')

    while signal_in.any():
        # Se corta la cantidad de muestras que se necesite, o bien, las que se
        # puedan cortar
        if len(signal_in) >= NT:
            q_samples = NT
            hop = int(NT - noverlap)
        else:
            q_samples = hop = len(signal_in)

        # Recorte en la cantidad de muestras
        signal_frame = signal_in[:q_samples]
        signal_in = signal_in[hop:]
                
        # Definición de los arreglos para el cálculo de la pendiente
        xaxis = np.array([])     # Correspondiente a los valores dx
        yaxis = np.array([])     # Correspondiente a los valores var_dx
        
        for k in range(kmin, kmax + 1):
            # Definición de la cantidad de ventanas nk
            nk = step_f(k)
            
            # Definición de la cantidad del tamaño de las sub-ventanas. Se
            # reemplaza NT por q_samples ya que es lo mismo y permite considerar
            # los últimos segmentos de audio
            Nk = int(q_samples / nk)
            
            # Definición de los j * n_k 
            j_nk_arrays = np.array([[j * nk, (j - 1) * nk] 
                                    for j in range(1, Nk)]).T
            
            # Calculo del delta_x
            delta_x = signal_frame[j_nk_arrays[0]] - signal_frame[j_nk_arrays[1]]

            # Calculo de var_dx
            var_dx_k = 1 / (Nk - 1) * (sum(delta_x ** 2) - 
                                       1 / Nk * (sum(delta_x)) ** 2)

            # Definición de delta_t
            delta_t = nk # / samplerate

            # Agregando a las listas
            xaxis = np.concatenate((xaxis, [np.log(delta_t)]))
            yaxis = np.concatenate((yaxis, [np.log(var_dx_k)]))
        
        # Estimación de la pendiente s
        if xaxis.shape[0] == 1:
            s = yaxis[0] / xaxis[0]
        else:
            s = stats.linregress(xaxis, yaxis)[0]

        # Con lo cual es posible obtener d_sigma
        d_sigma.append(2 - s / 2)
        
    return np.array(d_sigma)


def stationary_multiscale_wavelets(signal_in, wavelet='db4', levels=[2,3,4], 
                                   start_level=1, end_level=6, erase_pad=True):
    '''Función que permite calcular la multplicación en distintas escalas 
    de una descomposición en Wavelets estacionarias. La SWT (o Stationary 
    Wavelet Decomposition) corresponde a la clásica DWT (Discrete Wavelets
    descomposition), pero sin el paso utilizado para decimar la señal. Por 
    lo tanto, las señales mantienen su largo a través de las escalas.
    
    Parameters
    ----------
    signal_in : ndarray or list
        Señal de entrada a analizar mediante multiscale SWT.
    wavelet : {pywt.families(kind='discrete')} type, optional
        Wavelet utilizado para el proceso de dwt. Revisar en la 
        documentación de pywt. Por defecto es "db4".
    levels : ndarray or list
        Niveles a multiplicar mediante multiscale product. Asegurarse de que  
        estén entre "start_level" y "end_level". Por defecto es [2,3,4].
    start_level : int, optional
        Nivel en el que comienza la descomposición. Por defecto es 1.
    end_level : int, optional
        Nivel en el que termina la descomposición. Por defecto es 6.
    erase_pad : bool, optional
        Booleano que indica si es que se elimina el pad utilizado para 
        calcular el SWT. Por defecto es True. 
    
    Returns
    -------
    wav_mult : ndarray
        Resultado de la multiplicación multiescala de los coeficientes de
        detalle, obtenidas a partir de la descomposición SWT.
    coeffs : list 
        Lista que contiene todos los coeficientes de la descomposición SWT 
        entre los niveles establecidos. Los primeros índices corresponden 
        a coeficientes de aproximación, mientras que los segundos a 
        coeficientes de detalle.
    
    References
    ----------
    [1] Flores-Tapia, D., Moussavi, Z. M., & Thomas, G. (2007). Heart 
        sound cancellation based on multiscale products and linear 
        prediction. IEEE transactions on biomedical engineering, 54(2), 
        234-243.
    [2] Yadollahi, A., & Moussavi, Z. M. (2006). A robust method for 
        heart sounds localization using lung sounds entropy. IEEE 
        transactions on biomedical engineering, 53(3), 497-502.
    '''
    # Definición de la cantidad de puntos de la señal
    N = signal_in.shape[0]
    
    # Cantidad de puntos deseados
    points_desired = 2 ** int(np.ceil(np.log2(N)))
    
    # Definición de la cantidad de puntos de padding
    pad_points = (points_desired-N) // 2
    
    # Paddeando para lograr el largo potencia de 2 que se necesita
    audio_pad = np.pad(signal_in, pad_width=pad_points, 
                       constant_values=0)
    
    # Descomposición en Wavelets estacionarias
    coeffs = pywt.swt(audio_pad, wavelet=wavelet, level=end_level, 
                      start_level=start_level)
    
    # Definición del arreglo de multiplicación multiescala
    wav_mult = np.ones(len(coeffs[0][0]))
    
    # Realizando la multiplicación entre los distintos niveles
    for level in levels:
        # Se utilizan estos índices debido a cómo se ordena la 
        # salida de la función pywt.swt(.)
        wav_mult *= coeffs[-level + start_level][1]
    
    # Eliminar puntos de padding
    if erase_pad:
        wav_mult_out = wav_mult[pad_points:-pad_points]
        
        # Definición de la lista de coeficientes
        coeffs_out = list()
        
        for coef in coeffs:
            coeffs_out.append((coef[0][pad_points:-pad_points],
                               coef[1][pad_points:-pad_points]))
    
    else:
        wav_mult_out = wav_mult
        coeffs_out = coeffs
    
    return wav_mult_out, coeffs_out


def stationary_wavelets_decomposition(signal_in, wavelet='db4', levels=[2,3,4],
                                      start_level=1, end_level=6, erase_pad=True):
    '''Función que permite obtener las distintas escalas de una descomposición 
    en Wavelets estacionarias. La SWT (o Stationary Wavelet Decomposition) 
    corresponde a la clásica DWT (Discrete Wavelets descomposition), 
    pero sin el paso utilizado para decimar la señal. Por lo tanto, las señales 
    mantienen su largo a través de las escalas.
    
    Parameters
    ----------
    signal_in : ndarray or list
        Señal de entrada a analizar mediante multiscale SWT.
    wavelet : {pywt.families(kind='discrete')} type, optional
        Wavelet utilizado para el proceso de dwt. Revisar en la 
        documentación de pywt. Por defecto es "db4".
    levels : ndarray or list
        Niveles de la descomposición a obtener. Asegurarse de que  
        estén entre "start_level" y "end_level". Por defecto es [2,3,4].
    start_level : int, optional
        Nivel en el que comienza la descomposición. Por defecto es 1.
    end_level : int, optional
        Nivel en el que termina la descomposición. Por defecto es 6.
    erase_pad : bool, optional
        Booleano que indica si es que se elimina el pad utilizado para 
        calcular el SWT. Por defecto es True. 
    
    Returns
    -------
    wav_coeffs : ndarray
        Coeficientes de detalle, obtenidas a partir de la descomposición SWT.
    
    References
    ----------
    [1] Springer, D. B., Tarassenko, L., & Clifford, G. D. (2015). 
        Logistic regression-HSMM-based heart sound segmentation. 
        IEEE Transactions on Biomedical Engineering, 63(4), 822-832.
    '''
    # Definición de la cantidad de puntos de la señal
    N = signal_in.shape[0]
    
    # Cantidad de puntos deseados
    points_desired = 2 ** int(np.ceil(np.log2(N)))
    
    # Definición de la cantidad de puntos de padding
    pad_points = (points_desired-N) // 2
    
    # Paddeando para lograr el largo potencia de 2 que se necesita
    audio_pad = np.pad(signal_in, pad_width=pad_points, 
                       constant_values=0)
    
    # Descomposición en Wavelets estacionarias
    coeffs = pywt.swt(audio_pad, wavelet=wavelet, level=end_level, 
                      start_level=start_level)
    
    # Definición del arreglo donde se almacenarán los distintos niveles 
    # de descomposición
    wav_coeffs = np.zeros((len(coeffs[0][1]), 0))
    
    # Realizando la multiplicación entre los distintos niveles
    for level in levels:
        # Se utilizan estos índices debido a cómo se ordena la 
        # salida de la función pywt.swt(.)
        coef_i =  np.expand_dims(coeffs[-level + start_level][1], -1)
        
        # Concatenando los coeficientes
        wav_coeffs = np.concatenate((wav_coeffs, coef_i), axis=1)
        
    # Eliminar puntos de padding
    if erase_pad:
        wav_coeffs_out = wav_coeffs[pad_points:-pad_points]
    else:
        wav_coeffs_out = wav_coeffs
    
    return wav_coeffs_out


def modified_spectral_tracking(signal_in, samplerate, freq_obj=[150, 200], N=512, 
                               noverlap=256, padding=0, repeat=0, window='tukey'):
    '''Función que permite realizar spectral tracking a través del tiempo para 
    ciertas frecuencias.
    
    Parameters
    ----------
    signal_in : ndarray or list
        Señal de entrada.
    samplerate : int
        Tasa de muestreo de la señal de entrada.
    freq_obj : list, optional
        Frecuencias a analizar para el spectral tracking. Por defecto es 
        [150,200]. 
    **kwargs : Revisar parámetros de get_spectrogram.
    
    Returns
    -------
    spectral_trackings : list
        Lista de trackeos espectrales en base a las frecuencias entregadas en 
        "freq_obj".
    
    References
    ----------
    [1] Iwata, A., Ishii, N., Suzumura, N., & Ikegaya, K. (1980). Algorithm for 
        detecting the first and the second heart sounds by spectral tracking. 
        Medical and Biological Engineering and Computing, 18(1), 19-26.
    '''
    # Definición de la lista de trackings espectrales
    spectral_trackings = list()
    
    # Se obtiene el espectrograma
    _, f, S = get_spectrogram(signal_in, samplerate, N=N, padding=padding, 
                              repeat=repeat, noverlap=noverlap, window=window, 
                              whole=False)
    
    # Para cada frecuencia de interés
    for freq in freq_obj:
        # Se obtiene la frecuencia más cercana en base a la FFT
        freq_ind = np.argmin(abs(f - freq))
        
        # Y se guarda el tracking de esta frecuencia
        spectral_trackings.append(abs(S[freq_ind]))
        
    return spectral_trackings


def spectral_energy_bands(signal_in, samplerate, band_limits=[150, 200], alpha=2,
                          N=512,  noverlap=256, padding=0, repeat=0, 
                          window='hann'):
    '''Función que permite realizar spectral tracking a través del tiempo para 
    ciertas frecuencias.
    
    Parameters
    ----------
    signal_in : ndarray or list
        Señal de entrada.
    samplerate : int
        Tasa de muestreo de la señal de entrada.
    band_limits : list, optional
        Límite de las bandas de frecuencia a analizar para el cálculo de la 
        energía. Por defecto es [150,200]. 
    alpha : float, optional
        Exponente usado para el cálculo de la energía. Por defecto es 2.
    **kwargs : Revisar parámetros de get_spectrogram.
    
    Returns
    -------
    spectral_energy : ndarray
        Energía por ventanas en la banda de frecuencia especificada.
    
    References
    ----------
    [1] Schmidt, S. E., Holst-Hansen, C., Graff, C., Toft, E., & Struijk, J. J. 
        (2010). Segmentation of heart sound recordings by a duration-dependent 
        hidden Markov model. Physiological measurement, 31(4), 513.
    '''
    # Se obtiene el espectrograma
    _, f, S = get_spectrogram(signal_in, samplerate, N=N, padding=padding, 
                              repeat=repeat, noverlap=noverlap, window=window, 
                              whole=False)
    
    # Se obtienen los índices de interés dentro de la banda de frecuencia
    # especificada
    band_indexes = np.where((f > band_limits[0]) & (f < band_limits[1]))[0]
    
    # Obtención de la energía espectral en la banda de interés
    spectral_energy = abs(S[band_indexes]) ** alpha
    
    # Finalmente se suma para cada instante de tiempo
    return spectral_energy.sum(axis=0)


def get_spectral_info(signal_windowed, N=128, ind_audio=0, normalize=True):
    '''Función que permite obtener la fft sobre una señal ventaneada.
    
    Parameters
    ----------
    signal_windowed : ndarray
        Señal ventaneada.
    N : int, optional
        Cantidad de puntos de padding. Por defecto es 128.
    normalize : bool, optional
        Normalización de la señal. Por defecto es True.
        
    Returns
    -------
    fft_windowed : ndarray
        FFT unilateral de la señal ventaneada.
    '''
    # Definición de una matriz de padding del doble de largo
    to_pad = np.zeros((signal_windowed.shape[0], N))    
    
    # Aplicando el padding
    if signal_windowed.ndim > 2:
        signal_to = np.concatenate((signal_windowed[:, :, ind_audio], to_pad), 
                                   axis=1)
    else:
        signal_to = np.concatenate((signal_windowed, to_pad), axis=1)
    
    # Cálculo de la magnitud de la FFT
    fft_windowed = abs(np.fft.fft(signal_to))
    
    # Normalizando
    if normalize:
        # Se resta el mínimo y se normaliza (para dejarlo entre 0 y 1)
        fft_windowed = (fft_windowed.T - fft_windowed.min(axis=1)).T
        fft_windowed = (fft_windowed.T / fft_windowed.max(axis=1)).T
    
    return np.expand_dims(fft_windowed[:,:N], -1)


def get_envelope_pack(signal_in, samplerate, homomorphic_dict=None, 
                      hilbert_dict=None, simplicity_dict=None, 
                      vfd_dict=None, multiscale_wavelet_dict=None, 
                      spec_track_dict=None, spec_energy_dict=None, 
                      wavelet_dict=None, norm_type='minmax'):
    '''Función que permite obtener un set de envolventes concatenadas 
    en un arreglo en base a una señal de entrada. Es posible obtener 
    envolventes a partir de filtros homomórficos, envolventes basados 
    en la transformada de Hilbert, envolvente de simplicidad, envolvente 
    de dimensión fractal, envolventes de wavelet multiescala o tracking 
    espectrales.
    
    Parameters
    ----------
    signal_in : ndarray or list
        Señal de entrada.
    samplerate : float
        Tasa de muestreo de la señal de entrada.
    homomorphic_dict : dict or None
        Diccionario con información sobre los parámetros de la función 
        "homomorphic_filter". Por defecto es None. Si es None, no se 
        incluye como envolvente.
    homomorphic_dict : dict or None, optional
        Diccionario con información sobre los parámetros de la función 
        "homomorphic_filter". Por defecto es None. Si es None, no se 
        incluye como envolvente.
    hilbert_dict : dict or None, optional
        Diccionario con booleanos de inclusión de ciertas envolventes.
        'analytic_env' es el booleano para agregar la envolvente 
        analítica obtenida de la magntitud de la señal analítica.
        'inst_phase' es el booleano para agregar la fase instantánea
        obtenida como la fase de la señal analítica. 'inst_freq' es el
        booleano para agregar la frecuencia instantánea obtenida como 
        la derivada de la fase de la señal analítica. Por defecto es 
        None. Si es None, no se incluye como envolvente.
    vfd_dict : dict or None, optional
        Diccionario con información sobre los parámetros de la función 
        "variance_fractal_dimension". Por defecto es None. Si es None, 
        no se incluye como envolvente.
    multiscale_wavelet_dict : dict or None, optional
        Diccionario con información sobre los parámetros de la función 
        "stationary_multiscale_wavelets". Por defecto es None. Si es 
        None, no se incluye como envolvente.
    spec_track_dict : dict or None, optional
        Diccionario con información sobre los parámetros de la función 
        "spectral_energy_bands". Por defecto es None. Si es None, no 
        se incluye como envolvente.
    spec_energy_dict : dict or None, optional
        Diccionario con información sobre los parámetros de la función 
        "modified_spectral_tracking". Por defecto es None. Si es 
        None, no se incluye como envolvente.
    wavelet_dict : dict or None, optional
        Diccionario con información sobre los parámetros de la función 
        "stationary_wavelet_decomposition". Por defecto es None. Si es 
        None, no se incluye como envolvente.
    norm_type : {'minmax', 'mu-sigma'}, optional
        Tipo de normalización a aplicar en la señal. 'minmax' normaliza
        la señal entre 0 y 1. 'mu-sigma' normaliza la señal 
        estadísticamente restando la media y dividiendo por la desviación
        estándar. Por defecto es 'minmax'.
    
    Returns
    -------
    envelope_out : ndarray
        Arreglo con las distintas envolventes.
    '''
    # Función auxiliar para calcular las normalizaciones entre 0 y 1
    def _norm_01(x, resample=False):
        if resample:
            x = signal.resample(x, len(signal_in))
        x = x - min(x)
        return x / max(abs(x))
    
    # Función auxiliar para calcular las normalizaciones en base a
    # la media y la desviación estándar
    def _norm_mu_sigma(x, resample=False):
        if resample:
            x = signal.resample(x, len(signal_in))
        x = x - np.mean(x)
        return x / np.std(x)
    
    # Función auxiliar para definir la función de normalización
    def _norm_func(option):
        if option == 'minmax':
            return _norm_01
        elif option == 'mu-sigma':
            return _norm_mu_sigma
        else:
            raise Exception('Error en la selección de normalización.')
    
    # Definición de la función de normalización
    norm_func = _norm_func(norm_type)
    
    # Creación del vector de envolventes
    envelope_out = np.zeros((len(signal_in), 0))
    
    ### Filtro homomórfico ###
    if homomorphic_dict is not None:
        hom_out = homomorphic_filter(signal_in, samplerate, 
                                     cutoff_freq=homomorphic_dict['cutoff_freq'], 
                                     delta_band=homomorphic_dict['delta_band'], 
                                     filter_type='lowpass', epsilon=1e-10)
        
        # Normalizando
        hom_out = norm_func(hom_out)
        
        # Concatenando
        hom_out = np.expand_dims(hom_out, -1)
        envelope_out = np.concatenate((envelope_out, hom_out), axis=1)
    
    
    ### Transformada de Hilbert ###
    if hilbert_dict is not None:
        ### Pasos para transformada de Hilbert Modificada ###
        # Aplicando un threshold
        abs_signal = np.where(abs(signal_in) < 0.1 * max(abs(signal_in)),
                              0, abs(signal_in))

        # Cálculo de la envolvente de energía de Shannon
        shannon_hilb = shannon_envolve(abs_signal + 1e-10, alpha=2)

        # Cálculo de las envovlentes de hilbert (frecuencia instantánea)
        signal_hilb = homomorphic_filter(shannon_hilb, samplerate, 
                                         cutoff_freq=10, delta_band=5)

        # Se asegura que esté entre 0 y 1
        signal_hilb = _norm_01(signal_hilb)

        # Y se calcula su representación
        analytic_env_mod, inst_phase, inst_freq = \
                            hilbert_representation(signal_hilb, samplerate)
        
        if hilbert_dict['analytic_env']:
            # Obtener la envolvente analítica básica
            analytic_env, _, _ = hilbert_representation(signal_in, samplerate)
            
            # Normalizando
            analytic_env = norm_func(abs(analytic_env))
            analytic_env = np.expand_dims(analytic_env, -1)
            
            # Concatenando
            envelope_out = np.concatenate((envelope_out, analytic_env), axis=1)
        
        if hilbert_dict['analytic_env_mod']:
            # Normalizando
            analytic_env_mod = norm_func(abs(analytic_env_mod))
            analytic_env_mod = np.expand_dims(analytic_env_mod, -1)
            
            # Concatenando
            envelope_out = np.concatenate((envelope_out, analytic_env_mod), axis=1)
            
        if hilbert_dict['inst_phase']:
            # Normalizando
            inst_phase = norm_func(inst_phase)
            inst_phase = np.expand_dims(inst_phase, -1)
            
            # Concatenando
            envelope_out = np.concatenate((envelope_out, inst_phase), axis=1)    
        
        
        if hilbert_dict['inst_freq']:
            # Normalizando
            inst_freq = norm_func(inst_freq)
            inst_freq = np.expand_dims(inst_freq, -1)
            
            # Concatenando
            envelope_out = np.concatenate((envelope_out, inst_freq), axis=1)  
    
    
    ### Simplicity based envelope ###
    if simplicity_dict is not None:
        simp_env = simplicity_based_envelope(signal_in, N=simplicity_dict['N'], 
                                             noverlap=simplicity_dict['noverlap'], 
                                             m=simplicity_dict['m'], 
                                             tau=simplicity_dict['tau'])    
        
        # Normalizando
        simp_env = norm_func(simp_env, resample=True)
        
        # Concatenando
        simp_env = np.expand_dims(simp_env, -1)
        envelope_out = np.concatenate((envelope_out, simp_env), axis=1)
    
        
    ### Variance fractal dimension ###
    if vfd_dict is not None:
        vfd_env = variance_fractal_dimension(signal_in, samplerate, NT=vfd_dict['N'], 
                                             noverlap=vfd_dict['noverlap'], 
                                             kmin=vfd_dict['kmin'], kmax=vfd_dict['kmax'], 
                                             step_size_method=vfd_dict['step_size_method'])
        
        
        # Resampleando
        vfd_env_res = get_inverse_windowed_signal(vfd_env, N=vfd_dict['N'], 
                                                  noverlap=vfd_dict['noverlap'])
        
        # Recortando para el ajuste con la señal
        N_cut = vfd_dict['N'] // 2
        
        # Normalización
        vfd_env_norm = norm_func(vfd_env_res[:len(signal_in)], resample=False)
        
        # Hacerlo inverso
        if vfd_dict['inverse']:
            vfd_env_norm = 1 - vfd_env_norm
        
        # Concatenando
        vfd_env_norm = np.expand_dims(vfd_env_norm, -1)
        envelope_out = np.concatenate((envelope_out, vfd_env_norm), axis=1)
    
    
    ### Stationary Multiscale Wavelets ###
    if multiscale_wavelet_dict is not None:
        wav_mult, _ = \
            stationary_multiscale_wavelets(signal_in, wavelet=multiscale_wavelet_dict['wavelet'], 
                                           levels=multiscale_wavelet_dict['levels'], 
                                           start_level=multiscale_wavelet_dict['start_level'], 
                                           end_level=multiscale_wavelet_dict['end_level'])
        
        # Normalizando
        wav_mult = norm_func(abs(wav_mult))
        
        # Concatenando
        wav_mult = np.expand_dims(wav_mult, -1)
        envelope_out = np.concatenate((envelope_out, wav_mult), axis=1)
    
        
    ### Spectral tracking ###
    if spec_track_dict is not None:
        track_list = modified_spectral_tracking(signal_in, samplerate, 
                                                freq_obj=spec_track_dict['freq_obj'], 
                                                N=spec_track_dict['N'], 
                                                noverlap=spec_track_dict['noverlap'], 
                                                padding=spec_track_dict['padding'], 
                                                repeat=spec_track_dict['repeat'], 
                                                window=spec_track_dict['window'])
        
        # Normalizando y concatenando
        for track in track_list:
            # Resampleando
            track_res = get_inverse_windowed_signal(track, N=spec_track_dict['N'], 
                                                    noverlap=spec_track_dict['noverlap'])
            
            # Recortando para el ajuste con la señal
            N_cut = spec_track_dict['N'] // 2
            
            # Normalización
            track_norm = norm_func(track_res[N_cut:N_cut + len(signal_in)], 
                                   resample=False)
            
            # Concatenando
            track_norm = np.expand_dims(track_norm, -1)
            envelope_out = np.concatenate((envelope_out, track_norm), axis=1)
    
    
    ### Spectral Energy Bands ###
    if spec_energy_dict is not None:
        energy_env = spectral_energy_bands(signal_in, samplerate, 
                                           band_limits=spec_energy_dict['band_limits'], 
                                           alpha=spec_energy_dict['alpha'],
                                           N=spec_energy_dict['N'],  
                                           noverlap=spec_energy_dict['noverlap'], 
                                           padding=spec_energy_dict['padding'],
                                           repeat=spec_energy_dict['repeat'], 
                                           window=spec_energy_dict['window'])
        
        # Resampleando
        energy_env_res = \
                    get_inverse_windowed_signal(energy_env, N=spec_energy_dict['N'], 
                                                noverlap=spec_energy_dict['noverlap'])
        
        # Recortando para el ajuste con la señal
        N_cut = spec_energy_dict['N'] // 2
        
        # Normalización
        energy_env_norm = norm_func(energy_env_res[N_cut:N_cut + len(signal_in)], 
                                    resample=False)
        
        # Concatenando
        energy_env_norm = np.expand_dims(energy_env_norm, -1)
        envelope_out = np.concatenate((envelope_out, energy_env_norm), axis=1)
    
    
    ### Wavelet decomposition ###
    if wavelet_dict is not None:
        wav_coeffs = \
            stationary_wavelets_decomposition(signal_in, wavelet=wavelet_dict['wavelet'], 
                                              levels=wavelet_dict['levels'],
                                              start_level=wavelet_dict['start_level'], 
                                              end_level=wavelet_dict['end_level'], 
                                              erase_pad=True)

        # Normalizando y concatenando
        for i in range(wav_coeffs.shape[1]):
            # Normalización
            wavelet_norm = norm_func(abs(wav_coeffs[:,i]), resample=False)

            # Concatenando
            wavelet_norm = np.expand_dims(wavelet_norm, -1)
            envelope_out = np.concatenate((envelope_out, wavelet_norm), axis=1)  
    
    return envelope_out
