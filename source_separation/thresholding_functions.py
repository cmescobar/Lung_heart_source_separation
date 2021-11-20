import numpy as np
import cvxpy as cp
import pywt


def Sure_Shrink(signal_in, solve_method='iterations', step=0.001):
    ''' Aplicación del problema de optimización mediante SURE para
    la definición de un valor umbral que permita hacer denoising de
    la señal descompuesta en wavelets.
    
    Parámetros
    - signal_in: Señal de entrada x_i
    - solve_method: Método de resolución del valor "t"
        - ["iterations"]: Se resuelve viendo todos los posibles valores
                          entre cero y el máximo de la señal
        - ["optimization"]: Se resuelve mediante el problema de optimización.
                            TODAVIA NO FUNCIONA
    - step: Valor del salto aplicado al vector que se crea en el
            método de resolución 'iterations'. Se crea un vector
            que avanza en este monto hasta el máximo del valor
            absoluto de la señal
    
    Referencias:
    (1) Cai, C., & Harrington, P. de B. (1998). Different Discrete 
        Wavelet Transforms Applied to Denoising Analytical Data
    (2) G.P. Nason (2008). Wavelet Methods in Statistics with R.
    (3) Donoho, D. L., & Johnstone, I. M. (1995). Adapting to 
        Unknown Smoothness via Wavelet Shrinkage. Journal of the 
        American Statistical Association
    '''
    # Definición de N
    N = len(signal_in)
    
    # Función auxiliar que cuenta la cantidad de puntos en que la magnitud
    # de la señal es menor a un parámetro t
    count = lambda t: sum(abs(signal_in) <= t)
    
    # Función auxiliar que compara la magnitud de la señal con un parámetro 
    # t, escoge el mínimo y lo eleva al cuadrado generando un arreglo de 
    # valores de este tipo. Luego se suma cada uno
    #min_comp = lambda t: sum([min(abs(x_i), t) ** 2 for x_i in signal_in])
    min_comp = lambda t: sum(np.where(abs(signal_in) < t, 
                                      abs(signal_in) ** 2, 
                                      t ** 2))
    
    # Definición de la función/expresión SURE (función objetivo)
    sure_fo = lambda t: N - 2 * count(t) + min_comp(t)
    
    if solve_method == 'optimization': #### POR COMPLETAR
        # Definición de la variable a optimizar -> delta es el t óptimo
        delta = cp.Variable()

        # Definición de las restricciones
        constraints = [delta >= 0,
                       delta <= np.sqrt(2 * np.log(N))]

        # Definición de la función objetivo
        obj = cp.Minimize(sure_fo(delta))

        # Definición del problema de optimización
        prob = cp.Problem(obj, constraints)

        # Resolviendo el problema
        prob.solve(solver=cp.GUROBI, verbose=True)
    
    elif solve_method == 'iterations':
        # Definición de los posibles valores de t
        possible_t = np.arange(0, max(abs(signal_in)), step=step)
        
        # Definición de los valores de SURE a revisar
        sure_to_review = list()
        
        for i in possible_t:
            sure_to_review.append(abs(sure_fo(i)))
        
        # Una vez calculadas todas las SURE, se busca el índice 
        # del mínimo
        index_optimum = sure_to_review.index(min(sure_to_review))
        
        # Finalmente, se obtiene el valor del delta
        delta = possible_t[index_optimum]
    
        return delta
    

def wavelet_thresholding(signal_in, delta=None, threshold_criteria='soft',
                         threshold_delta='mad', min_percentage=None,
                         print_delta=False, log_base='e'):
    '''Definición de los tipos de thresholding aplicados a una función transformada
    al dominio wavelet.
    
    Parámetros
    - signal_in: Señal de entrada
    - delta: Valor del umbral manual. En caso de no querer ingresar este valor
             por defecto se mantiene como "None"
    - threshold_criteria: Criterio de aplicación de umbral, entre "hard" y "soft"
    - threshold_delta: Selección del criterio de cálculo de umbral. Opciones:
        - ["mad"]: Median Absolute Deviation
        - ["universal"]: universal (4)
        - ["sureshrink"]: Aplicando SURE (4)
        - ["percentage"]: Aplicación del porcentage en relación al máximo
    - min_percentage: Valor del porcentaje con respecto al máximo en la opción
                      "percentage" de la variable "threshold_delta"
    - print_delta: Booleano para indicar si se imprime el valor de delta
                      
    Referencias: 
    (1) http://www.numerical-tours.com/matlab/denoisingwav_1_wavelet_1d/
    (2) https://dsp.stackexchange.com/questions/15464/wavelet-thresholding
    (3) Valencia, D., Orejuela, D., Salazar, J., & Valencia, J. (2016). 
        Comparison analysis between rigrsure, sqtwolog, heursure and 
        minimaxi techniques using hard and soft thresholding methods.
    (4) Cai, C., & Harrington, P. de B. (1998). Different Discrete Wavelet
        Transforms Applied to Denoising Analytical Data
    '''
    # Definición del umbral de corte
    if delta is None:
        if threshold_delta == 'mad':
            delta = mad_thresholding(signal_in, log_base=log_base)
        
        elif threshold_delta == 'universal':
            delta = universal_thresholding(signal_in, log_base=log_base)
        
        elif threshold_delta == 'sureshrink':
            delta = Sure_Shrink(signal_in, solve_method='iterations', 
                                step=0.001)
        
        elif threshold_delta == 'percentage':
            delta = min_percentage * max(abs(signal_in))
            
    if print_delta:
        print(delta)

    return pywt.threshold(signal_in, value=delta, 
                          mode=threshold_criteria, 
                          substitute=0)


def mad_thresholding(signal_in, log_base='e'):
    # Se calcula la mediana
    med = np.median(signal_in)
    # Y se obtiene el sigma usando la median absolute deviation (MAD)
    sigma = np.median(abs(signal_in - med)) / 0.6745
    
    # Luego delta está dado por
    if log_base == 'e':
        delta = sigma * np.sqrt(2 * np.log(len(signal_in)))
    elif log_base == 2:
        delta = sigma * np.sqrt(2 * np.log2(len(signal_in)))
    elif log_base == 10:
        delta = sigma * np.sqrt(2 * np.log10(len(signal_in)))
    else:
        raise Exception('log_base no especificado correctamente.')
        
    return delta


def universal_thresholding(signal_in, log_base='e'):
    # Se calcula la mediana de la magnitud
    med = np.median(abs(signal_in))
    # Estimación del sigma
    sigma = med / 0.6745
    
    # Luego delta está dado por
    if log_base == 'e':
        delta = sigma * np.sqrt(2 * np.log(len(signal_in)))
    elif log_base == 2:
        delta = sigma * np.sqrt(2 * np.log2(len(signal_in)))
    elif log_base == 10:
        delta = sigma * np.sqrt(2 * np.log10(len(signal_in)))
    else:
        raise Exception('log_base no especificado correctamente.')
        
    return delta


def thresholding_processing(signal_in):
    '''Proceso que permite separar las envolventes de los murmullos del primer
    sonido cardíaco (S1) que no fueron totalmente removidos por ALPF
    
    Referencias: 
    - Qingshu Liu, et.al. An automatic segmentation method for heart sounds.
      2018. Biomedical Engineering.
    '''
    
    # Definición de los parámetros según el paper
    lamb = 0.8
    theta_c = 0.025
    
    # Definición del factor dependiente de la naturaleza de la señal
    theta_a = lamb * np.std(signal_in)
    
    # Definición del umbral de corte
    theta = min(theta_a, theta_c)
    
    return np.array([i if abs(i) >= theta else 0 for i in signal_in])
