from copy import Error
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ss_utils.filter_and_sampling import bandpass_filter, \
    downsampling_signal, upsampling_signal, lowpass_filter
from ss_utils.envelope_functions import get_envelope_pack


def get_signal_in_eval(signal_in, samplerate, append_audio=True, append_envelopes=False, 
                       apply_bpfilter=False, bp_parameters=None, 
                       homomorphic_dict=None, hilbert_dict=None, 
                       simplicity_dict=None, vfd_dict=None, 
                       multiscale_wavelet_dict=None, spec_track_dict=None,
                       spec_energy_dict=None, wavelet_dict=None, 
                       norm_type='minmax', append_fft=False):
    '''Función que obtiene las envolventes a partir de un archivo de audio en la 
    base de datos.
    
    Parameters
    ----------
    signal_in : str
        Señal de entrada a procesar a fs = 1000 Hz.
    append_audio : bool, optional
        Booleano que indica si se agrega el archivo de audio raw. Por defecto 
        es True.
    append_envelopes : bool, optional
        Booleano que indica si se agregan las envolventes de los archivos de
        audio. Por defecto es False.
    apply_bpfilter : bool, optional
        Aplicar un filtro pasa banda de manera previa sobre el audio.
        Por defecto es False.
    bp_parameters : list or ndarray, optional
        Arreglo de largo 4 indicando las frecuencias de corte en el orden:
        [freq_stop_1, freq_pass_1, freq_pass_2, freq_stop_2]. Por defecto 
        es None.
    homomorphic_dict : dict, optional
        Diccionario con los parámetros de la función 'homomorphic_filter'. 
        Por defecto es None.
    hilbert_dict : bool, optional
        hilbert_dict : dict or None, optional
        Diccionario con booleanos de inclusión de ciertas envolventes.
        'analytic_env' es el booleano para agregar la envolvente 
        analítica obtenida de la magntitud de la señal analítica.
        'inst_phase' es el booleano para agregar la fase instantánea
        obtenida como la fase de la señal analítica. 'inst_freq' es el
        booleano para agregar la frecuencia instantánea obtenida como 
        la derivada de la fase de la señal analítica. Por defecto es 
        None. Si es None, no se incluye como envolvente.
    simplicity_dict : dict, optional
        Diccionario con los parámetros de la función 
        'simplicity_based_envelope'. Por defecto es None.
    vfd_dict : dict, optional
        Diccionario con los parámetros de la función 
        'variance_fractal_dimension'. Por defecto es None.
    multiscale_wavelet_dict : dict, optional
        Diccionario con los parámetros de la función 
        'stationary_multiscale_wavelets'. Por defecto es None.
    spec_track_dict : dict, optional
        Diccionario con los parámetros de la función 
        'modified_spectral_tracking'. Por defecto es None.
    spec_energy_dict : dict or None, optional
        Diccionario con los parámetros de la función 
        "modified_spectral_tracking". Por defecto es None.
    wavelet_dict : dict, optional
        Diccionario con los parámetros de la función 
        'stationary_wavelets_decomposition'. Por defecto es None.
    append_fft : bool, optional
        Booleano que indica si se agregan la FFT unilateral de audio. Por 
        defecto es False.
        
    Returns
    -------
    audio_info_matrix : ndarray
        Matriz que contiene todas las ventanas de largo N de todos los archivos 
        de audio de la base de datos escogida.
    labels_adj : ndarray
        Matriz que contiene todas las etiquetas de todos los archivos 
        de audio de la base de datos escogida.
    '''        
    # Normalizando el audio
    audio = signal_in / max(abs(signal_in))
    
    # Aplicación de filtro pasa banda si es que se especifica
    if apply_bpfilter:
        audio = bandpass_filter(audio, samplerate, bp_method='scipy_fir',
                                freq_stop_1=bp_parameters[0], 
                                freq_pass_1=bp_parameters[1], 
                                freq_pass_2=bp_parameters[2], 
                                freq_stop_2=bp_parameters[3],
                                normalize=True)    
    
    # Definición de la variable en la que se almacenará la información
    audio_info = np.zeros((len(audio), 0))
    
    # Preguntar si se agrega el archivo de audio
    if append_audio:
        # Y agregando una dimensión para dejarlo en formato matriz
        audio_mat = np.expand_dims(audio, -1)
        
        # Concatenando
        audio_info = np.concatenate((audio_info, audio_mat), axis=1)
    
    
    # Preguntar si se agrega el pack de envolventes
    if append_envelopes:
        # Calculando las envolventes
        envelopes = get_envelope_pack(audio, samplerate, 
                                      homomorphic_dict=homomorphic_dict, 
                                      hilbert_dict=hilbert_dict,
                                      simplicity_dict=simplicity_dict, 
                                      vfd_dict=vfd_dict, 
                                      multiscale_wavelet_dict=multiscale_wavelet_dict,
                                      spec_track_dict=spec_track_dict,
                                      spec_energy_dict=spec_energy_dict, 
                                      wavelet_dict=wavelet_dict, 
                                      norm_type=norm_type)
        # Concatenando
        audio_info = np.concatenate((audio_info, envelopes), axis=1)
    
    # Retornando
    return audio_info
  

def transform_3_to_4_classes(y_hat):
    '''Función que permite transformar la salida de una red CNN encoder-decoder
    con 3 clases para los sonidos cardiacos (S0, S1 y S2) a una salida con 4
    clases (S1, Sis, S2 y Dia).
    
    Parameters
    ----------
    y_hat : ndarray or list
        Señales de salida de la red, indicando las probabilidades de cada clase.
        
    Returns
    -------
    y_out : ndarray
        Salida con las 4 clases definidas.
    s0_segments: list
        Lista con los intervalos donde se encuentran los sonidos no cardiacos S0.
    s0_class: list
        Lista con las etiquetas para cada uno de los intervalos donde se 
        encuentran los sonidos no cardiacos S0 (relativo a s0_segments). 
    '''
    # Definición de cada clase
    y_pred = np.argmax(y_hat[0], axis=-1)

    # Encontrando los puntos de cada sonido
    s0_pos = np.where(y_pred == 0)[0]

    # Definiendo los límites de cada sonido
    s0_segments = list()
    beg_seg = s0_pos[0]

    for i in range(len(s0_pos) - 1):
        if s0_pos[i + 1] - s0_pos[i] != 1:
            s0_segments.append([beg_seg, s0_pos[i]])
            beg_seg = s0_pos[i + 1]

    if s0_pos[-1] > beg_seg:
        s0_segments.append([beg_seg, s0_pos[-1]])

    # Clase del segmento s0
    s0_class = list()

    # Revisando segmento a segmento
    for seg in s0_segments:
        # Si es el segmento límite de la izquierda
        if seg[0] == 0:
            if y_pred[seg[1] + 1] == 2:
                s0_class.append('sys')
            elif y_pred[seg[1] + 1] == 1:
                s0_class.append('dia')
        
        # Si es el segmento límite de la derecha
        elif seg[1] == len(y_pred) - 1:
            if y_pred[seg[0] - 1] == 1:
                s0_class.append('sys')
            elif y_pred[seg[0] - 1] == 2:
                s0_class.append('dia')
        
        # Si es segmento intermedio
        else:
            if (y_pred[seg[0] - 1] == 1) and (y_pred[seg[1] + 1] == 2):
                s0_class.append('sys')
            elif (y_pred[seg[0] - 1] == 2) and (y_pred[seg[1] + 1] == 1):
                s0_class.append('dia')
            else:
                s0_class.append('Unidentified')

    # Finalmente, definiendo las clases
    y_out = np.zeros(y_pred.shape[0])

    # S1 : 1 & S2 : 3
    y_out += 1 * (y_pred == 1) + 3 * (y_pred == 2)

    # Sys: 2 & Dia: 4
    for i in range(len(s0_segments)):
        if s0_class[i] == 'sys':
            y_out[s0_segments[i][0]:s0_segments[i][1]+1] = 2
        elif s0_class[i] == 'dia':
            y_out[s0_segments[i][0]:s0_segments[i][1]+1] = 4
        elif s0_class[i] == 'Unidentified':
            y_out[s0_segments[i][0]:s0_segments[i][1]+1] = -1
        else:
            raise Exception('Error en el algoritmo de traspaso de clases. '
                            'Presencia de un segmento no identificado.')
            
    return y_out, s0_segments, s0_class


def eval_sound_model(signal_in, samplerate, model_name, lowpass_params=None):
    '''Rutina que permite evaluar un PCG de la base de datos utilizando una de las
    redes neuronales entrenadas.
    
    Parameters
    ----------
    filepath : str
        Dirección de la señal de entrada a evaluar por la red neuronal.
    model_name : str
        Nombre de la red a utilizar entre las disponibles en 
        "heart_sound_segmentation/models".
    lowpass_params : dict or None
        Diccionario que contiene la información del filtro pasa bajos en la 
        salida de la red. Si es None, no se utiliza. Por defecto es None.
    
    Returns
    -------
    y_hat : ndarray
        Salidas de la red indicando la probabilidad de ocurrencia de cada clase.
    '''
    # Función auxiliar para abrir el archivo y acondicionarlo a 1000 Hz.
    def _conditioning_signal(signal_in, samplerate):        
        # Acondicionando en caso de que no tenga samplerate de 1000 Hz.
        if samplerate < 1000:
            print(f'Upsampling de la señal de fs = {samplerate} Hz a fs = 1000 Hz.') 
            new_rate = 1000           
            audio_to = upsampling_signal(signal_in, samplerate, new_samplerate=new_rate)
            print("Por testear (!)")

        elif samplerate > 1000:
            print(f'Downsampling de la señal de fs = {samplerate} Hz a fs = 1000 Hz.')
            new_rate, audio_to = downsampling_signal(signal_in, samplerate, 
                                                     freq_pass=450, 
                                                     freq_stop=500)
        
        else:
            print(f'Samplerate adecuado a fs = {samplerate} Hz.')
            audio_to = signal_in
            new_rate = 1000
        
        # Mensaje para asegurar
        print(f'Señal acondicionada a {new_rate} Hz para la segmentación.')
        
        # Asegurándose de que el largo de la señal sea par
        if len(audio_to) % 2 != 0:
            audio_to = np.concatenate((audio_to, [0]))
        
        return audio_to, new_rate

    
    # Definición de los parámetros originales utilizados en el entrenamiento
    # de la red
    N_env_vfd = 64
    step_env_vfd = 8
    N_env_spec = 64
    step_env_spec = 8
    N_env_energy = 128
    step_env_energy = 16

    # Parámetros filtro pasabanda
    apply_bpfilter = True
    bp_parameters = [20, 30, 180, 190]

    # Parámetros de envolvente
    append_audio = True
    append_envelopes = True
    homomorphic_dict = {'cutoff_freq': 10, 'delta_band': 5}
    hilbert_dict = {'analytic_env': True, 'analytic_env_mod': True, 
                    'inst_phase': False, 'inst_freq': False}
    simplicity_dict = None
    vfd_dict = {'N': N_env_vfd, 'noverlap': N_env_vfd - step_env_vfd, 
                'kmin': 4, 'kmax': 4, 'step_size_method': 'unit', 
                'inverse': True}
    multiscale_wavelet_dict = {'wavelet': 'db6', 'levels': [3,4], 
                               'start_level': 0, 'end_level': 4}
    spec_track_dict =  {'freq_obj': [40, 60], 'N': N_env_spec, 
                        'noverlap': N_env_spec - step_env_spec, 
                        'padding': 0, 'repeat': 0, 'window': 'hann'}
    spec_energy_dict = {'band_limits': [30, 120], 'alpha': 1, 'N': N_env_energy, 
                        'noverlap': N_env_energy - step_env_energy, 
                        'padding': 0, 'repeat': 0 , 'window': 'hann'}
    wavelet_dict = {'wavelet': 'db6', 'levels': [4], 'start_level': 0, 
                    'end_level': 4}
    append_fft = False
    
    
    # Lectura del archivo
    audio, samplerate = _conditioning_signal(signal_in, samplerate)
    
    # Definición de los datos de entrenamiento y testeo
    audio_data = \
            get_signal_in_eval(audio, samplerate, append_audio=append_audio, 
                               append_envelopes=append_envelopes, 
                               apply_bpfilter=apply_bpfilter, 
                               bp_parameters=bp_parameters, 
                               homomorphic_dict=homomorphic_dict, 
                               hilbert_dict=hilbert_dict, 
                               simplicity_dict=simplicity_dict, 
                               vfd_dict=vfd_dict, 
                               multiscale_wavelet_dict=multiscale_wavelet_dict, 
                               spec_track_dict=spec_track_dict,
                               spec_energy_dict=spec_energy_dict, 
                               wavelet_dict=wavelet_dict, 
                               norm_type='minmax', append_fft=append_fft)
        
    # Ajustando los datos de entrada a la red
    audio_data = np.expand_dims(audio_data, 0)

    # Cargando el modelo
    try:
        model = tf.keras.models.load_model(f'models/{model_name}.h5')
    except:
        raise Error(f'Modelo "{model_name}" no existe en la carpeta "models".')
    
    # Realizando la predicción
    y_hat = model.predict(audio_data, verbose=1)
    
    # Opción de suavizado de la salida de la red
    if lowpass_params is not None:
        # Definición de la salida de este proceso
        y_hat_to = np.zeros((1, y_hat.shape[1], y_hat.shape[2]))

        for i in range(y_hat.shape[2]):
            # Aplicando el filtro pasa bajos para cada salida de la red
            _, y_hat_to[0, :, i] = \
                    lowpass_filter(y_hat[0, :, i], samplerate, 
                                   freq_pass=lowpass_params['freq_pass'], 
                                   freq_stop=lowpass_params['freq_stop'])
    else:
        y_hat_to = y_hat

    
    return audio, y_hat_to


def class_representations(y_hat, plot_outputs=False, audio_data=None):
    '''Representación en clases a partir de las probabilidades de ocurrencia de
    la salida de la red.    
    
    Parameters
    ----------
    y_hat : ndarray
        Salidas de la red indicando la probabilidad de ocurrencia de cada clase.
    plot_outputs : bool
        Booleano para indicar si se realizan gráficos. Por defecto es False.
    audio_data : ndarray or None
        Señal correspondiente al y_hat, usado para graficar.Por defecto 
        es None.
        
    Returns
    -------
    y_out2 : ndarray
        Salida de la red indicando las 2 posibles clases.
    y_out3 : ndarray
        Salida de la red indicando las 3 posibles clases.
    y_out4 : ndarray
        Salida de la red indicando las 4 posibles clases.
    '''
    # Obteniendo la señal de clases
    y_out3 = np.argmax(y_hat[0], axis=-1)
    
    # Representación en 2 etiquetas
    y_out2 = 1 - (y_out3 == 0)
    
    # Pasando a obtener 4 clases
    y_out4, _, _ = transform_3_to_4_classes(y_hat)
    
    # Graficar
    if plot_outputs:
        # Obteniendo los gráficos
        fig, axs = plt.subplots(2, 1, figsize=(7,4), sharex=True, frameon=True)
        
        audio_data_plot = 0.5 * audio_data / max(abs(audio_data))
        axs[0].plot(audio_data_plot - min(audio_data_plot), label=r'$s(n)$', 
                    color='silver', zorder=0)
        axs[0].plot(y_hat[0,:,0], label=r'$S_0$', color='limegreen', zorder=2)
        axs[0].plot(y_hat[0,:,1], label=r'$S_1$', color='red', zorder=1)
        axs[0].plot(y_hat[0,:,2], label= r'$S_2$', color='blue', zorder=1)
        axs[0].legend(loc='lower right')
        axs[0].set_yticks([0, 0.5, 1])
        axs[0].set_ylabel(r'$P(y(n) = k | X)$')

        axs[1].plot(y_out3)
        axs[1].set_ylabel(r'$y(n)$')
        axs[1].set_yticks([0,1,2])
        axs[1].set_yticklabels([r'$S_0$', r'$S_1$', r'$S_2$'])
        axs[1].set_ylim([-0.3, 2.3])

        # Alineando las etiquetas del eje vertical
        fig.align_ylabels(axs[:])
        # Remove horizontal space between axes
        fig.subplots_adjust(wspace=0.1, hspace=0)
        plt.show(block=False)
    
    return y_out2, y_out3, y_out4


# Módulo de testeo
if __name__ == '__main__':
    pass
