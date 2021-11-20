import numpy as np
import matplotlib.pyplot as plt
from ss_utils.math_functions import raised_cosine_fading


def fading_signal(signal_in, N, beta=1, side='both'):
    '''Función que permite aplicar un desvanecimiento de entrada, salida o ambos
    en una señal. 
    
    Parameters
    ----------
    signal_in : list or ndarray
        Señal a desvanecer en los bordes.
    N : int
        Cantidad de puntos de la ventana.
    beta : float
        Parámetro de la función coseno elevado para la apertura de la ventana.
    side : {'both', 'left', 'right'}, optional
        Dirección en la cual se puede usará la ventana. Se recomienda 'right' para
        el final de la señal y 'left' para el comienzo. Para 'both' se aplica tanto
        al inicio como al final. Por defecto es 'both'.
        
    Returns
    -------
    signal_faded : ndarray
        Señal con desvanecimiento en los bordes indicados.
    '''
    # Opciones de fading
    if side == 'both':
        # Segmentos a los que se le aplica la ventana fade
        faded_left = raised_cosine_fading(N, beta, side='left') * signal_in[:N]
        faded_right = raised_cosine_fading(N, beta, side='right') * signal_in[-N:]
        
        # Reconstruyendo
        signal_faded = np.concatenate((faded_left, signal_in[N:-N], faded_right))
        
    elif side == 'left':
        # Segmento al cual se le aplica la ventana fade
        faded_seg = raised_cosine_fading(N, beta, side=side) * signal_in[:N]
        
        # Reconstruyendo
        signal_faded = np.concatenate((faded_seg, signal_in[N:]))
    
    elif side == 'right':
        # Segmento al cual se le aplica la ventana fade
        faded_seg = raised_cosine_fading(N, beta, side=side) * signal_in[-N:]
        
        # Reconstruyendo
        signal_faded = np.concatenate((signal_in[:-N], faded_seg))
    else:
        raise Exception('Opción side no escogida correctamente. Intente nuevamente.')
    
    return signal_faded


def fade_connect_signals(signal_list, N, beta=1):
    '''Función que permite conectar una lista de señales mediante un fade de N
    puntos, basado en una transición coseno elevado.
    
    Parameters
    ----------
    signal_list : list
        Lista de señales a mezclar.
    N : int
        Cantidad de puntos de la ventana.
    beta : float
        Parámetro de la función coseno elevado para la apertura de la ventana.
    
    Returns
    -------
    signal_faded : ndarray
        Señal conectada mediante una transición coseno elevada de N puntos para 
        cada segmento.
    '''
    # Definición del primer signal_faded
    signal_faded = signal_list[0]
    
    for i in range(1, len(signal_list)):
        # Definición de la cantidad de puntos de fading
        N_to = min(N, len(signal_faded), len(signal_list[i]))
        
        # Aplicando fading de manera correspondiente a las señales
        faded_left = fading_signal(signal_faded, N_to, beta, side='right')
        faded_right = fading_signal(signal_list[i], N_to, beta, side='left')

        # Rellenando con ceros
        to_sum_left = np.concatenate((faded_left, [0] * (len(signal_list[i]) - N_to)))
        to_sum_right = np.concatenate(([0] * (len(signal_faded) - N_to), faded_right))

        # Se redefine signal_faded para la siguiente iteración
        signal_faded = to_sum_left + to_sum_right
    
    return signal_faded




# Módulo de testeo
if __name__ == '__main__':
    import soundfile as sf
    import matplotlib.pyplot as plt
    filename = 'Interest_Audios/Heart_sound_files/Level 4/136_1b1_Ar_sc_Meditron'
    audio, samplerate = sf.read(f'{filename}.wav')
    import time

    N = 50
    audio_left = audio[1000:1500]
    audio_center = audio[1500-N:2000+N]
    audio_right = audio[2000:2500]

    a = fade_connect_signals([audio_left, audio_center, audio_right], N, beta=1)

    plt.plot(range(1000, 2500), a, linewidth=3, label='Original')
    plt.plot(range(1000, 1500), audio_left, label='left')
    plt.plot(range(1500-N, 2000+N), audio_center, label='center')
    plt.plot(range(2000, 2500), audio_right, label='right')
    plt.legend()
    plt.show()
