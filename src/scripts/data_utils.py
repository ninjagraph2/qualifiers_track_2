import numpy as np
from nilearn.connectome import ConnectivityMeasure

def get_connectome(timeseries: np.ndarray, conn_type: str = 'corr') -> np.ndarray:

    indicies_to_extract = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                          19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                          30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                          45, 46, 47, 48, 49, 50, 51, 52, 215, 216, 219, 220, 253, 254, 261, 262]
    
    # Извлечение нужных элементов по третьему измерению
    timeseries_selected = timeseries[:, :, indicies_to_extract]

    if conn_type == 'corr':
        # Создание матрицы соединений
        conn = ConnectivityMeasure(kind='correlation', standardize=False).fit_transform(timeseries_selected)

        # Замена значений на близкие к единице
        conn[conn == 1] = 0.999999

        # Обнуление диагонали
        for i in conn:
            np.fill_diagonal(i, 0)

        # Применение обратной гиперболической тангенс-функции
        conn = np.arctanh(conn)

    else:
        raise NotImplementedError("Тип соединения не реализован.")

    return conn