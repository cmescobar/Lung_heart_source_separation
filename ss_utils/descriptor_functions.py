import numpy as np
import matplotlib.pyplot as plt
from ss_utils.math_functions import hamming_window, hann_window
from scipy.signal.windows import tukey, nuttall


# Descriptores temporales
def centroide(signal_in):
    '''Función que calcula el centroide de una señal de entrada.

    Parameters
    ----------
    signal_in: ndarray or list
        Señal de entrada.

    Returns
    -------
    signal_out : ndarray or list
        Centroide en cada instante de tiempo de la señal.
    '''
    return sum([i*abs(signal_in[i]) for i in range(len(signal_in))]) / sum(abs(signal_in))


def centroide_normalizado(signal_in):
    '''Función que calcula el centroide normalizado de una señal de entrada.

    Parameters
    ----------
    signal_in: ndarray or list
        Señal de entrada.

    Returns
    -------
    signal_out : ndarray or list
        Centroide normalizado en cada instante de tiempo de la señal.
    '''
    return sum([i*abs(signal_in[i]) for i in range(len(signal_in))]) / len(signal_in)


def promedio_aritmetico(signal_in):
    return sum(signal_in) / len(signal_in)


def varianza(signal_in):
    return sum((signal_in - promedio_aritmetico(signal_in)) ** 2) / len(signal_in)


def skewness(signal_in):
    return sum((signal_in - promedio_aritmetico(signal_in)) ** 3) / \
           (len(signal_in) * varianza(signal_in) ** (3 / 2))


def kurtosis(signal_in):
    return sum((signal_in - promedio_aritmetico(signal_in)) ** 4) / \
           (len(signal_in) * varianza(signal_in) ** 2) - 3


def rms(signal_in):
    return np.sqrt(sum(signal_in ** 2) / len(signal_in))


def max_amp(signal_in):
    return max(abs(signal_in))


def zero_cross_rate(signal_in):
    return sum([abs(np.sign(signal_in[i]) - np.sign(signal_in[i + 1]))
                for i in range(len(signal_in) - 1)]) / (2 * len(signal_in))


# Descriptores frecuenciales
def pendiente_espectral(signal_in):
    # Aplicando fft
    fft_signal_in = np.fft.fft(signal_in)
    # Definición de kappa
    K = len(fft_signal_in)
    # Luego realizando el cálculo del flujo espectral
    a = sum([i * abs(fft_signal_in[i]) for i in range(round(K / 2))])
    b = sum([i for i in range(round(K / 2))]) * \
        sum([abs(fft_signal_in[i]) for i in range(round(K / 2))])
    c = sum([i ** 2 for i in range(round(K / 2))])
    d = sum([i for i in range(round(K / 2))])
    return (K/2*a - b) / (K/2*c - d**2)


def centroide_espectral(signal_in):
    # Aplicando fft
    fft_signal_in = np.fft.fft(signal_in)
    # Luego se hace el calculo del centroide
    return sum([i * abs(fft_signal_in[i])**2
                for i in range(round(len(fft_signal_in) / 2))]) / \
           sum(abs(fft_signal_in) ** 2)


def flujo_espectral(signal_in, signal_in_before):
    # Aplicando fft
    fft_1 = np.fft.fft(signal_in)
    fft_2 = np.fft.fft(signal_in_before)
    # Definición de kappa
    K = len(fft_1)
    # Luego realizando el cálculo del flujo espectral
    return np.sqrt(sum([(abs(fft_1[i]) - abs(fft_2[i])) ** 2
                        for i in range(round(K / 2))])) / (K / 2)


def spectral_flatness(signal_in):
    # Aplicando fft
    fft_signal_in = np.fft.fft(signal_in)
    # Definición de kappa
    K = round(len(fft_signal_in)/2)
    # Luego realizando el cálculo de la planitud espectral
    return np.exp(1 / K * sum([np.log(abs(fft_signal_in[i])) for i in range(K)])) /\
           (1 / K * sum([abs(fft_signal_in[i]) for i in range(K)]))


def abs_fourier_shift(signal_in, samplerate, N_rep):
    signal_in_rep = np.tile(signal_in, N_rep)
    fourier = np.fft.fft(signal_in_rep)
    frec = np.fft.fftfreq(len(signal_in_rep), 1/samplerate)
    fourier_shift = abs(fourier)
    return frec, fourier_shift


def abs_fourier_db_half(signal_in, samplerate, N_rep):
    signal_in_rep = np.tile(signal_in, N_rep)
    fourier = 20 * np.log10(abs(np.fft.fft(signal_in_rep)) + 1e-12)
    frec = np.fft.fftfreq(len(signal_in_rep), 1 / samplerate)
    return frec[:len(signal_in) // 2], fourier[:len(signal_in)//2] 


def get_spectrogram(signal_in, samplerate, N=512, padding=0, repeat=0, noverlap=0, 
                    window='tukey', whole=False):
    '''Función que permite obtener la STFT de una señal.
    
    Parameters
    ----------
    signal_in : ndarray
        Señal de entrada a transformar.
    samplerate : int
        Tasa de muestreo de la señal de entrada.
    N : int, optional
        Cantidad de puntos a utilizar por ventana. Por defecto es 512.
    padding : int, optional
        Cantidad de puntos de zero padding al final de la señal. Por defecto es 0.
    repeat : int, optional
        Cantidad de veces que se repite la señal en el cálculo de la STFT. Por defecto es 0.
    noverlap : int, optional
        Cantidad de puntos de traslape que se utiliza para calcular la STFT. Por defecto
        es 0.
    window : {'tukey', 'hamming', 'hann', 'nuttall'}, None, optional
        Ventana a utilizar para el cálculo de la STFT. Por defecto es 'tukey'. Con None se
        aplica ventana rectangular.
    whole : bool, optional
        Indica si se calcula la STFT hasta samplerate (True) o hasta samplerate // 2 (False).
        Por defecto es False.
        
    Returns
    -------
    t : ndarray
        Arreglo que indica las etiquetas temporales de la matriz que representa la STFT.
    f : ndarray
        Arreglo que indica las etiquetas frecuenciales de la matriz que representa la STFT.
    S : ndarray
        Espectrograma calculado a partir de la STFT de la señal de entrada.
    '''
    
    # Corroboración de criterios: noverlap <= N - 1
    if N <= noverlap:
        raise Exception('noverlap debe ser menor que N.')
    elif noverlap < 0:
        raise Exception('noverlap no puede ser negativo')
    else:
        noverlap = int(noverlap)
        
    # Propiedad de repeat
    repeat = int(repeat) if repeat >= 0 else 0
    
    # Lista donde se almacenará los valores del espectrograma
    to_fft = []
    # Lista de tiempo
    times = []
    
    # Variables auxiliares
    t = 0   # Tiempo
    
    # Definición del paso de avance
    step = N - noverlap
    
    # Si el norverlap es 0, se hacen ventanas 2 muestras más grandes 
    # para no considerar los bordes izquierdo y derecho (que son 0)
    if noverlap == 0:
        N_window = N + 2
    else:
        N_window = N
    
    # Seleccionar ventana.
    if window == 'tukey':
        wind_mask = tukey(N_window)
    elif window == 'hamming':
        wind_mask = hamming_window(N_window)
    elif window == 'hann':
        wind_mask = hann_window(N_window)
    elif window == 'nuttall':
        wind_mask = nuttall(N_window)
    elif window is None:
        wind_mask = np.array([1] * N_window)
    
    # Y se recorta en caso de noverlap cero
    wind_mask = wind_mask[1:-1] if noverlap == 0 else wind_mask
    
    # Definición de bordes de signal_in
    signal_in = np.concatenate((np.zeros(N//2), signal_in, np.zeros(N//2)))
    
    # Iteración sobre el audio
    while signal_in.size != 0:
        # Se corta la cantidad de muestras que se necesite, o bien, las que se
        # puedan cortar
        if len(signal_in) >= N:
            # Se obtienen las N muestras de interés
            signal_frame = signal_in[:N]
            
            # Y se corta la señal para la siguiente iteración
            signal_in = signal_in[step:]
            
        # En la última iteración se añaden ceros para lograr el largo N
        else:
            # Definición del último frame
            last_frame = signal_in[:]
            
            # Se rellena con ceros hasta lograr el largo            
            signal_frame = np.append(last_frame, [0] * (N - len(last_frame)))
            
            # Y se corta la señal para la siguiente iteración
            signal_in = signal_in[:0]
    
        # Agregando a los vectores del espectro
        to_fft.append(signal_frame)
        
        # Agregando al vector de tiempo
        times.append(t)
        t += step/samplerate
    
    # Ventaneando
    signal_wind = np.array(to_fft) * wind_mask

    # Repetición de la señal
    if repeat > 0:
        signal_wind = np.pad(signal_wind, pad_width=((0,0), (repeat * N // 2, repeat * N // 2)),
                             mode='reflect')
    
    # Aplicando padding
    zeros = np.zeros((signal_wind.shape[0], padding), dtype=signal_wind.dtype)
    signal_padded = np.concatenate((signal_wind, zeros), axis=1)

    # Aplicando transformada de fourier
    spect = np.fft.fft(signal_padded)
    
    # Preguntar si se quiere el espectro completo, o solo la mitad (debido a
    # que está reflejado hermitianamente)
    if whole:
        # Generar el vector de frecuencias para cada ventana
        freqs = np.linspace(0, samplerate, N + padding + repeat * 2 * (N // 2))

        # Una vez obtenido el spect_mag y spect_pha, se pasa a matriz
        spect = np.array(spect, dtype=np.complex128)
    else:
        # Generar el vector de frecuencias para cada ventana
        freqs = np.linspace(0, samplerate//2, ( N + padding + repeat * 2 * (N // 2))//2 + 1)

        # Una vez obtenido el spect_mag y spect_pha, se pasa a matriz
        spect = np.array(spect, 
                         dtype=np.complex128)[:, :(N + padding + repeat * 2 * (N // 2))//2 + 1]

    # Escalando
    spect *= np.sqrt(1 / (N * np.sum(wind_mask ** 2)))
    
    # Se retornan los valores que permiten construir el espectrograma 
    # correspondiente
    return times, freqs, spect.T


def get_inverse_spectrogram(X, N=None, padding=0, repeat=0, noverlap=0, window='tukey', 
                            whole=False):
    '''Función que permite obtener la ISTFT de un espectrograma.
    
    Parameters
    ----------
    X : ndarray
        Espectrograma a aplicar la ISTFT
    N : int, optional
        Cantidad de puntos a utilizar por ventana. Por defecto es None, con lo cual
        se asume que el N es la dimensión fila de X.
    padding : int, optional
        Cantidad de puntos de padding al final de la señal. Por defecto es 0.
    repeat : int, optional
        Cantidad de veces que se repite la señal en el cálculo de la STFT. Por defecto es 0.
    noverlap : int, optional
        Cantidad de puntos de traslape que se utiliza para calcular la STFT. Por defecto
        es 0.
    window : {'tukey', 'hamming', 'hann', 'nuttall'}, None, optional
        Ventana a utilizar para el cálculo de la STFT. Por defecto es 'tukey'. Con None se
        aplica ventana rectangular.
    whole : bool, optional
        Indica si se calcula la STFT hasta samplerate (True) o hasta samplerate // 2 (False).
        Por defecto es False.
        
    Returns
    -------
    s_out : ndarray
        Señal en el tiempo de correspondiente a la ISTFT de X.
    '''
    # Preguntar si es que la señal está en el rango 0-samplerate. En caso de 
    # que no sea así, se debe concatenar el conjugado de la señal para 
    # recuperar el espectro. Esto se hace así debido a la propiedad de las 
    # señales reales que dice que la FT de una señal real entrega una señal 
    # hermitiana (parte real par, parte imaginaria impar). Luego, como solo 
    # tenemos la mitad de la señal, la otra parte correspondiente a la señal 
    # debiera ser la misma pero conjugada, para que al transformar esta señal 
    # hermitiana mediante la IFT, se recupere una señal real (correspondiente a 
    # la señal de audio).
    
    # Propiedad de padding y repeat
    padding = int(padding) if padding > 0 else 0
    repeat = int(repeat) if repeat > 0 else 0
    
    if not whole:
        # Se refleja lo existente utilizando el conjugado
        X = np.concatenate((X, np.flip(np.conj(X[1:-1, :]), axis=0)))
            
    # Obtener la dimensión de la matriz
    rows, cols = X.shape
        
    # Corroboración de criterios: noverlap <= N - 1
    if rows <= noverlap:
        raise Exception('noverlap debe ser menor que la dimensión fila.')
    else:
        noverlap = int(noverlap)
    
    # Definición de N dependiendo de la naturaleza de la situación
    if N is None or N > rows:
        N = rows
    
    # Si el norverlap es 0, se hacen ventanas 2 muestras más grandes 
    # para no considerar los bordes izquierdo y derecho (que son 0)
    if noverlap == 0:
        N_window = N + 2
    else:
        N_window = N
    
    # Seleccionar ventana
    if window == 'tukey':
        wind_mask = tukey(N_window)
    elif window == 'hamming':
        wind_mask = hamming_window(N_window)
    elif window == 'hann':
        wind_mask = hann_window(N_window)
    elif window == 'nuttall':
        wind_mask = nuttall(N_window)
    elif window is None:
        wind_mask = np.array([1] * N_window)
        
    # Y se recorta en caso de noverlap cero
    wind_mask = wind_mask[1:-1] if noverlap == 0 else wind_mask
    
    # Destransformando y re escalando se obtiene
    ifft_scaled = np.fft.ifft(X, axis=0) * np.sqrt(N * np.sum(wind_mask ** 2))
    
    # Retirando el padding
    if padding > 0:
        # Se corta el padding
        ifft_scaled = ifft_scaled[:-padding,:]
    
    if repeat > 0:
        ifft_scaled = ifft_scaled[repeat*N//2:-repeat*N//2,:] 
    
    # A partir del overlap, el tamaño de cada ventana de la fft (dimensión fila)
    # y la cantidad de frames a las que se les aplicó la transformación 
    # (dimensión columna), se define la cantidad de muestras que representa la
    # señal original
    step = N - noverlap                     # Tamaño del paso
    total_samples = step * (cols - 1) + N   # Tamaño total del arreglo
    
    # Definición de una lista en la que se almacena la transformada inversa
    inv_spect = np.zeros(total_samples, dtype=np.complex128)
    # Definición de una lista de suma de ventanas cuadráticas en el tiempo
    sum_wind = np.zeros(total_samples, dtype=np.complex128)
    
    # Transformando columna a columna (nótese la división en tiempo por una 
    # ventana definida)
    for i in range(cols):
        beg = i * step
        # Se multiplica por el kernel para la reconstrucción a partir de la
        # ventana aplicada inicialmente. Fuente:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.istft.html        
        # Agregando al arreglo
        inv_spect[beg:beg+N] += ifft_scaled[:, i]
        
        # Se suma la ventana (que sirve como ponderador)
        sum_wind[beg:beg+N] += wind_mask
    
    # Se corta el padding agregado
    inv_spect = inv_spect[N//2:-1*N//2]
    sum_wind = sum_wind[N//2:-1*N//2]
    
    # Finalmente se aplica la normalización por la cantidad de veces que se
    # suma cada muestra en el proceso anterior producto del traslape,
    # utilizando las ventanas correspondientes
    return np.real(np.divide(inv_spect, np.where(sum_wind > 1e-10, sum_wind, 1)))


def get_windowed_signal(signal_in, samplerate, N=512, noverlap=0, 
                        padding_value=2):
    '''Función que permite obtener la representación ventaneada en matriz 
    de una señal. Se diferencia de la original en que es utilizada para 
    una matriz de dimensiones (len(signal_in), 1).
    
    Parameters
    ----------
    signal_in : ndarray
        Señal de entrada a transformar. Puede tener más de un canal (por 
        ejemplo: audio, wavelets, shannon, etc.)
    samplerate : int
        Tasa de muestreo de la señal de entrada.
    N : int, optional
        Cantidad de puntos a utilizar por ventana. Por defecto es 512.
    noverlap : int, optional
        Cantidad de puntos de traslape que se utiliza para calcular la 
        matriz. Por defecto es 0.
    padding_value : float, optional
        Valor que se utiliza para hacer padding de la señal cuando se 
        encuentra en la última ventana (que generalmente tiene menos) 
        puntos que las anteriores. Por defecto es 2.
        
    Returns
    -------
    signal_out : ndarray
        Arreglo de ventanas desplazadas (delay) de la señal.
        
    References
    ----------
    [1] Renna, F., Oliveira, J., & Coimbra, M. T. (2019). Deep convolutional 
        neural networks for heart sound segmentation. IEEE journal of 
        biomedical and health informatics, 23(6), 2435-2445.
    '''
    # Corroboración de criterios: noverlap <= N - 1
    if N <= noverlap:
        raise Exception('noverlap debe ser menor que N.')
    elif noverlap < 0:
        raise Exception('noverlap no puede ser negativo')
    else:
        noverlap = int(noverlap)
    
    # Lista donde se almacenará los valores del espectrograma
    signal_out = list()
    
    # Definición del paso de avance
    step = N - noverlap
        
    # Iteración sobre el audio
    while signal_in.shape[0] != 0:
        # Se corta la cantidad de muestras que se necesite, o bien, las 
        # que se puedan cortar
        if signal_in.shape[0] >= N:
            # Se obtienen las N muestras de interés
            signal_frame = signal_in[:N]
            
            # Y se corta la señal para la siguiente iteración
            signal_in = signal_in[step:]
            
        # En la última iteración se añaden ceros para lograr el largo N
        else:
            # Definición del último frame
            last_frame = signal_in[:]
            
            # Se rellena con ceros hasta lograr el largo
            if signal_in.ndim == 1:
                signal_frame = np.zeros(N) + padding_value
                signal_frame[:last_frame.shape[0]] = last_frame
            
            elif signal_in.ndim == 2:
                signal_frame = np.zeros((N, last_frame.shape[1])) + \
                               padding_value
                signal_frame[:last_frame.shape[0], 
                             :last_frame.shape[1]] = last_frame

            # Y se corta la señal para la siguiente iteración
            signal_in = signal_in[:0]
        
        # Agregando a los vectores del espectro
        signal_out.append(signal_frame)
    
    return np.array(signal_out)


def get_inverse_windowed_signal(signal_in, N, noverlap):
    '''Función que permite obtener la representación original de una 
    señal a partir de una matriz de señal ventaneada.
    
    Parameters
    ----------
    signal_in : ndarray
        Señal de entrada a transformar. Puede tener más de un canal (por 
        ejemplo: audio, wavelets, shannon, etc.)
    N : int, optional
        Cantidad de puntos a utilizar por ventana. Por defecto es 512.
    noverlap : int, optional
        Cantidad de puntos de traslape que se utiliza para calcular la 
        matriz. Por defecto es 0.
        
    Returns
    -------
    signal_out : ndarray
        Reconstrucción a partir de una señal ventaneada.
    '''
    # A partir del overlap, el tamaño de cada ventana y la cantidad de frames 
    # a las que se les ventanea, se define la cantidad de muestras que 
    # representa la señal original
    step = N - noverlap                               # Tamaño del paso
    total_samples = step * (len(signal_in) - 1) + N   # Tamaño total del arreglo
    
    # Definición de una lista en la que se almacena la transformada inversa
    inv_wind = np.zeros(total_samples, dtype=np.float)
    
    # Definición de una lista de suma de ventanas cuadráticas en el tiempo
    sum_wind = np.zeros(total_samples, dtype=np.float)
    
    # Transformando punto a punto (nótese la división en tiempo por una 
    # ventana definida)
    for i, sample in enumerate(signal_in):
        # Definición del punto inicial
        beg = i * step
        # Se agrega una ventana de "N" puntos con valor "sample"
        inv_wind[beg:beg+N] += sample
        
        # Se agrega una ventana de "N" puntos con valor 1 que permitirá 
        # corregir por los valores de traslape
        sum_wind[beg:beg+N] += 1
        
    return np.divide(inv_wind, sum_wind)


def get_noised_signal(signal_in, snr_expected, seed=None, plot_signals=False,
                      normalize=True):
    '''Función que permite agregar ruido blanco gaussiano a una señal de 
    entrada, utilizando una especificación SNR en decibeles.
    
    Parameters
    ----------
    signal_in : ndarray or list
        Señal de entrada.
    snr_expected : float
        Relación SNR deseada para la señal de salida.
    seed : int or None, optional
        Semilla a utilizar para la creación del ruido blanco gaussiano. Por
        defect es None.
    plot_signal : bool, optional
        Booleano para preguntar si es que se grafica la señal original en 
        conjunto con el ruido blanco generado. Por defecto es False.
    normalize : bool, optional
        Booleano para normalizar la señal de salida. Por defecto es True.
        
    Returns
    -------
    signal_out : ndarray
        Señal con ruido blanco según la relación "snr_expected".
    '''
    # Calcular la energía de la señal de entrada
    e_signal = np.sum(signal_in ** 2)
    
    # Aplicación de la semilla a utilizar para la creación del ruido blanco
    if seed is not None:
        np.random.seed(seed)
    
    # Creación del ruido blanco gaussiano
    signal_noise_01 = np.random.normal(0, 1, size=len(signal_in))
    
    # Calcular la energía de la señal de ruido a añadir
    e_noise_01 = np.sum(signal_noise_01 ** 2)
    
    # Calculando el coeficiente necesario para que la energía del ruido
    # cumpla con la SNR especificada
    e_noise_desired = e_signal / (10 ** (snr_expected / 10))
    
    # Definición del coeficiente
    k = e_noise_desired / e_noise_01
    
    # Se define el ruido deseado a agregar
    signal_noise = np.sqrt(k) * signal_noise_01
        
    # Finalmente se agrega la señal de entrada
    signal_out = signal_in + signal_noise
    
    # Normalizando
    if normalize:
        signal_out = signal_out / max(abs(signal_out))
        
    # Graficando
    if plot_signals:
        plt.plot(signal_in)
        plt.plot(signal_noise)
        plt.show()
        
    return signal_out


def normalize(X):
    mf = X.mean(0)
    sf = X.std(0)
    a = 1 / sf
    b = - mf / sf
    return X * a + b, a, b


# Módulo de testeo
if __name__ == '__main__':
    normal = lambda mu, sigma, x:  1/(sigma * np.sqrt(2 * np.pi)) * \
        np.exp( - (x - mu)**2 / (2 * sigma**2))

    s1 = 10
    u1 = 7

    s2 = 15
    u2 = 15

    x = np.linspace(-35,25,1000)

    n1 = normal(u1, s1, x)
    n2 = normal(u2, s2, x)

    print(centroide(n1))
    print(centroide(n2))

    plt.plot(n1)
    plt.plot(n2)
    plt.axvline(centroide(n1), ymin=-1, color='b')
    plt.axvline(centroide(n2), ymin=-1, color='r')
    plt.show()