import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from ss_utils.evaluation_functions import eval_sound_model, class_representations


def hss_segmentation(signal_in, samplerate, model_name,
                     length_desired, lowpass_params=None, 
                     plot_outputs=False):
    '''Función que segmenta un sonido auscultado de entrada utilizando
    uno de los modelos disponibles en la carpeta "models".
    
    Parameters
    ----------
    signal_in : ndarray
        Señal de entrada.
    samplerate : int or float
        Tasa de muestreo de la señal de entrada.
    model_name : str
        Nombre del modelo de la red en la dirección 
        "heart_sound_segmentation/models".
    length_desired : int or float
        Largo deseado de la señal.
    lowpass_params : dict or None
        Diccionario que contiene la información del filtro pasa 
        bajos en la salida de la red. Si es None, no se utiliza. 
        Por defecto es None.
    plot_outputs : bool
        Booleano para indicar si se realizan gráficos. Por defecto 
        es False.
        
    Returns
    -------
    y_hat : ndarray
        Salidas de la red indicando la probabilidad de ocurrencia 
        de cada clase.
    y_hat_to : ndarray
        Salidas de la red indicando la probabilidad de ocurrencia 
        de cada clase, pero con la corrección de la cantidad de 
        puntos.
    (y_out2, y_out3, y_out4) : list of ndarray
        Salida de la red indicando las 2, 3 y 4 posibles clases.
    '''
    # Salida de la red
    _audio, y_hat = eval_sound_model(signal_in, samplerate, model_name,
                                     lowpass_params=lowpass_params)
    
    # Definición del largo deseado ajustado a y_hat
    length_desired_to = round(len(y_hat[0,:,0]) / len(_audio) * \
                              length_desired)
    
    # Definición de las probabilidades resampleadas
    y_hat_to = np.zeros((1, length_desired_to, 3))
    
    # Para cada una de las salidas, se aplica un resample
    for i in range(3):
        y_hat_to[0, :, i] = \
            segments_redimension(y_hat[0, :, i], 
                                 length_desired=length_desired_to,
                                 kind='cubic')
        
    # Definiendo la cantidad de puntos finales a añadir
    q_times = length_desired - y_hat_to.shape[1]
    
    # Generando los puntos a añadir
    points_to_add = np.tile(y_hat_to[:,-1,:], (1, q_times, 1))
    
    # Agregando los puntos a la señal
    y_hat_to = np.concatenate((y_hat_to, points_to_add), axis=1)
        
    # Representación en clases
    y_out2, y_out3, y_out4 = \
        class_representations(y_hat_to, plot_outputs=plot_outputs,
                              audio_data=None)
    
    return y_hat, y_hat_to, (y_out2, y_out3, y_out4)


def segments_redimension(signal_in, length_desired, kind='linear'):
    '''Función que redimensiona la salida y_hat de las redes para
    dejarlo en función de un largo deseado.
    
    Parameters
    ----------
    signal_in : ndarray
        Señal de entrada.
    length_desired : int
        Largo deseado de la señal.
    kind : str
        Opción kind de la función "scipy.interpolate.interp1d".
        Por defecto es "linear".
    
    Returns 
    -------
    signal_out : ndarray
        Señal resampleada.
    '''
    # Definición del eje temporal de la señal
    x = np.linspace(0, length_desired - 1, len(signal_in)) 
    
    # Función de interpolación en base a los datos de la señal
    f = interp1d(x, signal_in, kind=kind)
    
    # Definición de la nueva cantidad de puntos
    x_new = np.arange(length_desired)
        
    # Interpolando finalmente
    return f(x_new)


def find_segments_limits(y_hat, segments_return='Non-Heart'):
    '''Función que obtiene los límites de las posiciones de los sonidos
    cardiacos a partir de la señal binaria indica su presencia.
    
    Parameters
    ----------
    y_hat : ndarray
        Señal binaria que indica la presencia de sonidos cardiacos.
    segments_return : {'Heart', 'Non-Heart'}, optional
        Opción que decide si es que se retornan los intervalos de sonido 
        cardiaco o los intervalos libres de sonido cardiaco. Por defecto
        es 'Non-Heart'.
    
    Returns
    -------
    interval_list : list
        Lista de intervalos en los que se encuentra el sonido cardiaco.
    '''
    # Encontrando los puntos de cada sonido
    if segments_return == 'Non-Heart':
        hss_pos = np.where(y_hat == 0)[0]
    
    elif segments_return == 'Heart':
        hss_pos = np.where(y_hat == 1)[0]
        
    else:
        raise Exception('Opción no válida para "segments_return".')
    
    # Definición de la lista de intervalos
    interval_list = list()
    
    # Inicio del intervalo
    beg_seg = hss_pos[0]
    
    # Definiendo 
    for i in range(len(hss_pos) - 1):
        if hss_pos[i + 1] - hss_pos[i] != 1:
            interval_list.append([beg_seg, hss_pos[i]])
            beg_seg = hss_pos[i + 1]

    if hss_pos[-1] > beg_seg:
        interval_list.append([beg_seg, hss_pos[-1]])
        
    return interval_list


# Módulo de testeo
if __name__ == '__main__':
    print("Testeo de función en utils\n")
    
    # Definición de la función a testear
    test_func = 'signal_segmentation'
    
    if test_func == 'signal_segmentation':
        # Definición de la frecuencia de muestreo deseada para separación de fuentes
        samplerate_des = 11025  # Hz
        
        # Cargando el archivo de audio 
        db_folder = 'samples_test'
        audio, samplerate = sf.read(f'{db_folder}/123_1b1_Al_sc_Meditron.wav')
                                
        # Parámetros del filtro pasa bajos a la salida de la red
        lowpass_params = {'freq_pass': 140, 'freq_stop': 150}
        # Definición del modelo a utilizar
        model_name = 'definitive_segnet_based'
        
        # Obteniendo la salida de la red
        y_hat, y_hat_to, (y_out2, y_out3, y_out4) = \
                hss_segmentation(audio, samplerate, model_name,
                                 length_desired=len(audio),
                                 lowpass_params=lowpass_params,
                                 plot_outputs=False)
        
        plt.figure(figsize=(9,5))
        audio_data_plot = 0.5 * audio / max(abs(audio))
        plt.plot(audio_data_plot + 0.5, label=r'$s(n)$', 
                color='silver', zorder=0)
        plt.plot(y_hat_to[0,:,0], label=r'$S_0$', color='limegreen', zorder=2)
        plt.plot(y_hat_to[0,:,1], label=r'$S_1$', color='red', zorder=1)
        plt.plot(y_hat_to[0,:,2], label=r'$S_2$', color='blue', zorder=1)
        plt.legend(loc='lower right')
        plt.yticks([0, 0.5, 1])
        plt.xlabel('Muestras')
        plt.ylabel(r'$P(y(n) = k | X)$')
        plt.title('Predicción de sonidos cardiacos')
        plt.show()
