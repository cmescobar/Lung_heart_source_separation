import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.io import wavfile
from source_separation.nmf_decompositions import nmf_process
from heart_prediction import hss_segmentation, find_segments_limits
from ss_utils.filter_and_sampling import downsampling_signal, upsampling_signal


def find_and_open_audio(db_folder):
    '''Función que permite la apertura de archivos de audio en la base
    de datos de la carpeta especificada.
    
    Parameters
    ----------
    db_folder : str
        Carpeta de la base de datos.
        
    Returns
    -------
    audio : ndarray
        Señal de audio de interés.
    samplerate : int or float
        Tasa de muestreo de la señal.    
    '''
    def _file_selection(filenames):
        print('Seleccione el archivo que desea descomponer:')
        for num, name in enumerate(filenames):
            print(f'[{num + 1}] {name}')
            
        # Definición de la selección
        selection = int(input('Selección: '))
        
        # Se retorna
        try:
            return filenames[selection-1].strip('.wav')
        except:
            raise Exception('No ha seleccionado un archivo válido.')
    
    
    def _open_file(filename):
        # Obtención del archivo de audio .wav
        try:
            samplerate, audio = wavfile.read(f'{filename}.wav')
        except:
            audio, samplerate = sf.read(f'{filename}.wav')
            
        return audio, samplerate
    
    
    # Definición del archivo a revisar
    filenames = [i for i in os.listdir(db_folder) if i.endswith('.wav')]

    # Definición de la ubicación del archivo
    filename = f'{db_folder}/{_file_selection(filenames)}'
    
    # Retornando
    return _open_file(filename)


def nmf_lung_heart_separation(signal_in, samplerate, model_name,
                              samplerate_nmf=11025,
                              filter_parameters={'bool':False},
                              nmf_method='replace_segments',
                              plot_segmentation=False,
                              plot_separation=False):
    '''Función que permite hacer un preprocesamiento de la señal
    auscultada de entrada en la función.
    
    Parameters
    ----------
    signal_in : ndarrray
        Señal de entrada.
    samplerate : float
        Tasa de muestreo de la señal de entrada.
    model_name : str
        Nombre del modelo de la red en la dirección 
        "heart_sound_segmentation/models".
    samplerate_nmf : float, optional
        Frecuencia de muestreo deseada para la separación de 
        fuentes. Por defecto es 11025 Hz.
    nmf_method : {'to_all', 'on_segments', 'masked_segments', 
                  'replace_segments'}, optional
        Método de descomposición NMF a aplicar en la separación
        de fuentes. Por defecto es "replace_segments".
    plot_segmentation : bool, optional
        Booleano que indica si es que se grafica el proceso de 
        segmentación. Por defecto es False.
    plot_separation : bool, optional
        Booleano que indica si es que se grafica el proceso de 
        separación de fuentes. Por defecto es False.

    Returns
    -------
    resp_signal : ndarray
        Señal respiratoria obtenida mediante la descomposición.
    heart_signal : ndarray
        Señal cardíaca obtenida mediante la descomposición.
    '''
    def _conditioning_signal(signal_in, samplerate, samplerate_to):
        # Acondicionando en caso de que no tenga samplerate de 1000 Hz.
        if samplerate < samplerate_to:
            print(f'Upsampling de la señal de fs = {samplerate} Hz '
                  f'a fs = {samplerate_to} Hz.') 
            new_rate = samplerate_to           
            audio_to = upsampling_signal(signal_in, samplerate, new_samplerate=new_rate)

        elif samplerate > samplerate_to:
            print(f'Downsampling de la señal de fs = {samplerate} Hz '
                  f'a fs = {samplerate_to} Hz.')
            new_rate, audio_to = downsampling_signal(signal_in, samplerate, 
                                                     freq_pass=samplerate_to//2-100, 
                                                     freq_stop=samplerate_to//2)
        
        else:
            print(f'Samplerate adecuado a fs = {samplerate} Hz.')
            audio_to = signal_in
            new_rate = samplerate_to
        
        # Mensaje para asegurar
        print(f'Señal acondicionada a {new_rate} Hz para la separación de fuentes.')
        
        # Asegurándose de que el largo de la señal sea par
        if len(audio_to) % 2 != 0:
            audio_to = np.concatenate((audio_to, [0]))
        
        return audio_to, new_rate


    # Definición de los parámetros de filtros pasa bajos de la salida de la red
    lowpass_params = {'freq_pass': 140, 'freq_stop': 150}

    # Definición de los parámetros NMF
    nmf_parameters = {'n_components': 2, 'N': 1024, 'N_lax': 100, 
                      'N_fade': 100, 'noverlap': int(0.9 * 1024), 'repeat': 0, 
                      'padding': 0, 'window': 'hamming', 'init': 'random',
                      'solver': 'mu', 'beta': 2, 'tol': 1e-4, 
                      'max_iter': 500, 'alpha_nmf': 0, 'l1_ratio': 0, 
                      'random_state': 0, 'dec_criteria': 'temp_criterion'}
    
    
    # Realizando un downsampling para obtener la tasa de muestreo
    # fs = 11025 Hz utilizada en la separación de fuentes
    audio_to, _ = _conditioning_signal(signal_in, samplerate, 
                                       samplerate_nmf)
    
    # Obteniendo la salida de la red
    _, y_hat_to, (y_out2, _, _) = \
            hss_segmentation(signal_in, samplerate, model_name,
                             length_desired=len(audio_to),
                             lowpass_params=lowpass_params,
                             plot_outputs=False)

    # Definiendo los intervalos para realizar la separación de fuentes
    interval_list = find_segments_limits(y_out2, segments_return='Heart')
    
    # Print de sanidad
    print(f'Aplicando separación de fuentes {nmf_method}...')
    
    # Aplicando la separación de fuentes
    resp_signal, heart_signal = \
            nmf_process(audio_to, samplerate_nmf, hs_pos=y_out2, 
                        interval_list=interval_list, 
                        nmf_parameters=nmf_parameters,
                        filter_parameters=filter_parameters, 
                        nmf_method=nmf_method)
    
    print('Separación de fuentes completada')
    
    # Graficando la segmentación
    if plot_segmentation:
        audio_data_plot = 0.5 * audio_to / max(abs(audio_to))
        plt.plot(audio_data_plot - min(audio_data_plot), label=r'$s(n)$', 
                 color='silver', zorder=0)
        plt.plot(y_hat_to[0,:,0], label=r'$S_0$', color='limegreen', zorder=2)
        plt.plot(y_hat_to[0,:,1], label=r'$S_1$', color='red', zorder=1)
        plt.plot(y_hat_to[0,:,2], label=r'$S_2$', color='blue', zorder=1)
        for i in interval_list:
            plt.axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)
        plt.legend(loc='lower right')
        plt.yticks([0, 0.5, 1])
        plt.ylabel(r'$P(y(n) = k | X)$')
        plt.show()
    
    
    # Graficando la separación de fuentes
    if plot_separation:
        fig, ax = plt.subplots(3, 1, figsize=(15,7), sharex=True)
        
        ax[0].plot(audio_to)
        ax[0].set_ylabel('Señal\noriginal')
        for i in interval_list:
            ax[0].axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)
        
        ax[1].plot(resp_signal)
        ax[1].set_ylabel('Señal\nRespiratoria')
        for i in interval_list:
            ax[1].axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)
        
        ax[2].plot(heart_signal)
        ax[2].set_ylabel('Señal\nCardiaca')
        for i in interval_list:
            ax[2].axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)
        
        # Ajustando las etiquetas del eje
        fig.align_ylabels(ax[:])
        # Quitando el espacio entre gráficos
        fig.subplots_adjust(wspace=0.1, hspace=0)

        plt.suptitle('Separación de fuentes')
        plt.show()
        
    return resp_signal, heart_signal


def nmf_lung_heart_separation_params(signal_in, samplerate,*,model_name, 
                                     lowpass_params, nmf_parameters, 
                                     samplerate_nmf=11025,
                                     nmf_method='replace_segments',
                                     filter_parameters={'bool': False},
                                     plot_segmentation=False,
                                     plot_separation=False):
    '''Función que permite hacer un preprocesamiento de la señal
    auscultada de entrada en la función. A diferencia de la función
    principal, en esta es posible definir los parámetros de filtro pasabajo 
    y de descomposición NMF.
    
    Parameters
    ----------
    signal_in : ndarrray
        Señal de entrada.
    samplerate : float
        Tasa de muestreo de la señal de entrada.
    model_name : str
        Nombre del modelo de la red en la dirección 
        "heart_sound_segmentation/models".
    lowpass_params : dict or None
        Diccionario que contiene la información del filtro pasa 
        bajos en la salida de la red. Si es None, no se utiliza. 
        Por defecto es None.
    nmf_parameters : dict
        Diccionario que contiene los parámetros de interés a definir
        para la descomposición NMF de la señal. Se recomienda usar:
        {'n_components': 2, 'N': 1024, 'N_lax': 100, 
        'N_fade': 100, 'noverlap': int(0.9 * 1024), 'repeat': 0, 
        'padding': 0, 'window': 'hamming', 'init': 'random',
        'solver': 'mu', 'beta': 2, 'tol': 1e-4, 
        'max_iter': 500, 'alpha_nmf': 0, 'l1_ratio': 0, 
        'random_state': 0, 'dec_criteria': 'temp_criterion'}.
    samplerate_nmf : float, optional
        Frecuencia de muestreo deseada para la separación de 
        fuentes. Por defecto es 11025 Hz.
    nmf_method : {'to_all', 'on_segments', 'masked_segments', 
                  'replace_segments'}, optional
        Método de descomposición NMF a aplicar en la separación
        de fuentes. Por defecto es "replace_segments".
    filter_parameters : dict, optional
        Diccionario que mediante el key "bool" permite controlar
        si se aplica un filtro pasa bajo sobre la señal respiratoria
        obtenida a la salida. Por defecto es {'bool': False}.
    plot_segmentation : bool, optional
        Booleano que indica si es que se grafica el proceso de 
        segmentación. Por defecto es False.
    plot_separation : bool, optional
        Booleano que indica si es que se grafica el proceso de 
        separación de fuentes. Por defecto es False.

    Returns
    -------
    resp_signal : ndarray
        Señal respiratoria obtenida mediante la descomposición.
    heart_signal : ndarray
        Señal cardíaca obtenida mediante la descomposición.
    '''
    def _conditioning_signal(signal_in, samplerate, samplerate_to):
        # Acondicionando en caso de que no tenga samplerate de 1000 Hz.
        if samplerate < samplerate_to:
            print(f'Upsampling de la señal de fs = {samplerate} Hz '
                  f'a fs = {samplerate_to} Hz.') 
            new_rate = samplerate_to           
            audio_to = upsampling_signal(signal_in, samplerate, new_samplerate=new_rate)

        elif samplerate > samplerate_to:
            print(f'Downsampling de la señal de fs = {samplerate} Hz '
                  f'a fs = {samplerate_to} Hz.')
            new_rate, audio_to = downsampling_signal(signal_in, samplerate, 
                                                     freq_pass=samplerate_to//2-100, 
                                                     freq_stop=samplerate_to//2)
        
        else:
            print(f'Samplerate adecuado a fs = {samplerate} Hz.')
            audio_to = signal_in
            new_rate = samplerate_to
        
        # Mensaje para asegurar
        print(f'Señal acondicionada a {new_rate} Hz para la separación de fuentes.')
        
        # Asegurándose de que el largo de la señal sea par
        if len(audio_to) % 2 != 0:
            audio_to = np.concatenate((audio_to, [0]))
        
        return audio_to, new_rate

    
    # Realizando un downsampling para obtener la tasa de muestreo
    # fs = 11025 Hz utilizada en la separación de fuentes
    audio_to, _ = _conditioning_signal(signal_in, samplerate, 
                                       samplerate_nmf)
    
    # Obteniendo la salida de la red
    _, y_hat_to, (y_out2, _, _) = \
            hss_segmentation(signal_in, samplerate, model_name,
                             length_desired=len(audio_to),
                             lowpass_params=lowpass_params,
                             plot_outputs=False)

    # Definiendo los intervalos para realizar la separación de fuentes
    interval_list = find_segments_limits(y_out2, segments_return='Heart')
    
    # Print de sanidad
    print(f'Aplicando separación de fuentes {nmf_method}...')
    
    # Aplicando la separación de fuentes
    resp_signal, heart_signal = \
            nmf_process(audio_to, samplerate_nmf, hs_pos=y_out2, 
                        interval_list=interval_list, 
                        nmf_parameters=nmf_parameters,
                        filter_parameters=filter_parameters, 
                        nmf_method=nmf_method)
    
    
    print('Separación de fuentes completada')
    
    # Graficando la segmentación
    if plot_segmentation:
        audio_data_plot = 0.5 * audio_to / max(abs(audio_to))
        plt.plot(audio_data_plot - min(audio_data_plot), label=r'$s(n)$', 
                 color='silver', zorder=0)
        plt.plot(y_hat_to[0,:,0], label=r'$S_0$', color='limegreen', zorder=2)
        plt.plot(y_hat_to[0,:,1], label=r'$S_1$', color='red', zorder=1)
        plt.plot(y_hat_to[0,:,2], label=r'$S_2$', color='blue', zorder=1)
        for i in interval_list:
            plt.axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)
        plt.legend(loc='lower right')
        plt.yticks([0, 0.5, 1])
        plt.ylabel(r'$P(y(n) = k | X)$')
        plt.show()
    
    
    # Graficando la separación de fuentes
    if plot_separation:
        fig, ax = plt.subplots(3, 1, figsize=(15,7), sharex=True)
        
        ax[0].plot(audio_to)
        ax[0].set_ylabel('Señal\noriginal')
        for i in interval_list:
            ax[0].axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)
        
        ax[1].plot(resp_signal)
        ax[1].set_ylabel('Señal\nRespiratoria')
        for i in interval_list:
            ax[1].axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)
        
        ax[2].plot(heart_signal)
        ax[2].set_ylabel('Señal\nCardiaca')
        for i in interval_list:
            ax[2].axvspan(xmin=i[0], xmax=i[1], facecolor='purple', alpha=0.1)
        
        # Ajustando las etiquetas del eje
        fig.align_ylabels(ax[:])
        # Quitando el espacio entre gráficos
        fig.subplots_adjust(wspace=0.1, hspace=0)

        plt.suptitle('Separación de fuentes')
        plt.show()
        
    return resp_signal, heart_signal


# Módulo de testeo
if __name__ == '__main__':
    # Definición de la función a testear
    test_func = 'resample_example'
    
    # Aplicación de la función 
    if test_func == 'resample_example':
        # Archivo de audio
        db_folder = 'samples_test'
        audio, samplerate = find_and_open_audio(db_folder)
        
        # Definición de los parámetros de filtros pasa bajos de la salida de la red
        lowpass_params = {'freq_pass': 140, 'freq_stop': 150}

        # Definición de los parámetros NMF
        nmf_parameters = {'n_components': 2, 'N': 1024, 'N_lax': 100, 
                          'N_fade': 100, 'noverlap': int(0.9 * 1024), 'repeat': 0, 
                          'padding': 0, 'window': 'hamming', 'init': 'random',
                          'solver': 'mu', 'beta': 2, 'tol': 1e-4, 
                          'max_iter': 500, 'alpha_nmf': 0, 'l1_ratio': 0, 
                          'random_state': 0, 'dec_criteria': 'temp_criterion'}

        # Obteniendo la señal 
        resp_signal, heart_signal = \
            nmf_lung_heart_separation_params(audio, samplerate, 
                                             model_name='definitive_segnet_based', 
                                             lowpass_params=lowpass_params, 
                                             nmf_parameters=nmf_parameters,
                                             samplerate_nmf=11025)
        
        # Creaciónde la figura        
        fig, axs = plt.subplots(3, 1, figsize=(15,8), sharex=True)

        # Aplicando downsampling
        new_rate, audio_dwns = \
                    downsampling_signal(audio, samplerate, 
                                        freq_pass=11025//2-100, 
                                        freq_stop=11025//2)
        print('Nueva tasa de muestreo para plot:', new_rate)
        
        axs[0].plot(audio_dwns)
        axs[0].set_ylabel('Señal\noriginal')
        axs[0].set_xticks([])
        axs[0].set_ylim([-1.3, 1.3])
        axs[0].set_title('Señal original & componentes obtenidas')

        axs[1].plot(resp_signal)
        axs[1].set_xticks([])
        axs[1].set_ylabel('Señales\nrespiratorias')
        axs[1].set_ylim([-1.3, 1.3])

        axs[2].plot(heart_signal)
        axs[2].set_xlabel('Muestras')
        axs[2].set_ylabel('Señales\ncardiacas')
        axs[2].set_ylim([-1.3, 1.3])
        
        # Alineando los labels del eje y
        fig.align_ylabels(axs[:])
        
        # Remover espacio horizontal entre plots
        fig.subplots_adjust(hspace=0)
        
        plt.show()
