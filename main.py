import soundfile as sf
import matplotlib.pyplot as plt
from ss_utils.filter_and_sampling import downsampling_signal
from heart_lung_separation import nmf_lung_heart_separation


# Módulo de testeo
if __name__ == '__main__':
    # Abriendo audio de ejemplo
    filename = 'samples_test/123_1b1_Al_sc_Meditron.wav'
    audio, samplerate = sf.read(filename)
    
    # Obteniendo las señales
    lung_signal, heart_signal = \
        nmf_lung_heart_separation(audio, samplerate, 
                                  model_name='definitive_segnet_based')
    
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

    axs[1].plot(lung_signal)
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
