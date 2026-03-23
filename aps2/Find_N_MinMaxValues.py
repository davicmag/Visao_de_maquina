import numpy as np

def Find_N_MaxValues(array_in, N):
	
	"""
	#busca pelos indices dos N maiores valores do array_IN

    Keyword arguments:
    array_in -- nparray 2D de entrada
    N = nro de indexes maximos a serem localizados

    Returns:
    location -- lista dos indexes dos N maiores valores do array_IN

    """

	array_dump = array_in.copy()
	locations = []
	minimum = array_dump.min() - 1
	for i in range(N):
		maxIndex = np.unravel_index(array_dump.argmax(), array_dump.shape)
		locations.append(maxIndex)
		array_dump[maxIndex] = minimum
	    
	return(locations)


def Find_N_MinValues(array_in, N):
	
	"""
	#busca pelos indices dos N menores valores do array_IN

    Keyword arguments:
    array_in -- nparray 2D de entrada
    N = nro de indexes mínimos a serem localizados

    Returns:
    location -- lista dos indexes dos N menores valores do array_IN

    """

	array_dump = array_in.copy()
	locations = []
	maximum = array_dump.max() + 1
	for i in range(N):
		minIndex = np.unravel_index(array_dump.argmin(), array_dump.shape)
		locations.append(minIndex)
		array_dump[minIndex] = maximum
	    
	return(locations)