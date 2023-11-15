**Calificador Volátil**

Declarar una variable en la memoria global o compartida usando el *calificador volátil* evita la optimización del compilador, que podría almacenar datos temporalmente en registros o en la memoria local.

Con el *calificador volátil*, el compilador asume que el valor de la variable puede ser cambiado o utilizado en cualquier momento por cualquier hilo. Por lo tanto, cualquier referencia a esta variable se compila en una lectura de memoria global o escritura de memoria global que omite la caché.

______________________________________________
**Memoria Compartida vs Memoria Global**

La memoria global de la GPU se encuentra en el dispositivo de memoria (DRAM), por lo que es más lento de acceder que la memoria compartida de la GPU. Comparada con la DRAM, la memoria compartida tiene:

* De 20 a 30 veces menor latencia que la DRAM
* Más de 10 veces mayor ancho de banda que la DRAM

La granularidad de acceso a la memoria compartida también es menor.
____________________________________________________________________
**Layout de la Memoria Compartida**

Para utilizar la memoria compartida a su mayor eficiencia, es importante conocer lo siguiente:
* Arreglos Cuadrados vs Arreglos Rectangulares
* Acceso a la Fila Mayor vs Acceso a la Columna Mayor
* Declaración de Memoria Compartida Estática vs Dinámica
* Alcance del Archivo vs Alcance del Kernel
* Padding de Memoria vs No Padding de Memoria

Al diseñar un kernel que utilice la memoria compartida, tu enfoque debe de seguir los siguientes conceptos:
* Mapeo de datos entre bancos de memoria
* Mapeo del indice del hilo al desplazamiento de la memoria compartida
_______________________________________________________________
**Memoria Compartida Cuadrada**

Puedes utilizar la memoria compartida para almacenar en caché datos globales con dimensiones cuadradas de forma sencilla. La dimensionalidad simple de una matriz cuadrada facilita el cálculo de compensaciones de memoria de 1 dimensión a partir de indices de hilos de 2 dimensiones.

![Arreglo2D](/Investigacion4/Imagenes/Arreglo2D.png)

Para declarar una variable estatica 2D en memoria compartida, lo haces de la siguiente forma:

      __shared__ int tile [N][N];

Como esta variable en memoria compartida es cuadrada, podrás acceder a ella con un bloque de hilos 2D de la siguiente manera:

        tile[threadIdx.y][threadIdx.x]
Esta forma de acceder es mucho más eficiente y de menor conflicto de bancos que esta:

        tile[threadIdx.x][threadIdx.y]
Esto debido a que los hilos vecinos están accediendo a las celdas de la matriz vecina a lo largo de la dimensión más interna de la matriz. 
________________________________________________________________________________
**Acceso a la Fila Mayor vs Acceso a la Columna Mayor**

Una vez declarado nuestro arreglo 2D en la memoria compartida, lo que debemos de hacer es calcular el índice global del hilo para cada hilo a partir de se ID de hilo 2D. Como sólo se ejecutará un bloque de hilos, la conversión del índice se puede simplificar:

        unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

Almacenar el valor de *idx* al arreglo de salida nos permite visualizar el patrón de acceso del kernel basado en donde los hilos escriben su ID global.
Escribir el índice global del hilo en la memoria compartida, en orden de fila mayor, se realiza de la siguiente manera:
    
        tile[threadIdx.y][threadIdx.x] = idx;

Una vez que el punto de sincronización es alcanzado (mediante el uso de syncthreads), todos los hilos deben almacenar información en la memoria compartida, para así poder asignar valores a la memoria global desde la memoria compartida en orden fila mayor:

        out[idx] = tile[threadIdx.y][threadIdx.x];

Como todos los hilos del mismo warp tienen valores consecutivos de *threadIdx.x* y usan *threadIdx.x* como índice para acceder a la matriz de la memoria compartida, este kernel esta libre de conflictos de banco.

Por otro lado, si intercambias *threadIdx.y* y *threadIdx.x* al momento de asignar información en la memoria compartida, el acceso de memoria de un warp será de orden columna mayor. Esto causaría unos 32 conflictos de banco en dispositivos Fermi y 16 conflictos de banco en dispositivos Kepler.
_____________________________________________________________________________________________________________
**Escribir en Filas y Leer en Columnas**

Para implementar la escritura en filas de la memoria compartida, debemos poner el dimensión más interna del índice del hilo como índice de columna del tile de la memeria compartida 2D:
    
        tile[threadIdx.y][threadIdx.x] = idx;
Asignar valores a la memoria global desde la memoria compartida en orden de columna se hace al intercambiar dos índices de hilos cuando se referencian en memoria compartida:

        out[idx] = tile[threadIdx.x][threadIdx.y];

Con esto, la operación de almacenar se encuentra libre de conflictos, pero la operación de cargar reporta 16 conflictos.
_______________________________________________________________________________________________________________
**Memoria Compartida Dinámica**

Puedes implementar los mismos kernels al declarar la memoria compartida de forma dinámica. Esto lo puedes hacer tanto fuera del kernel para hacerlo global al alcance del archivo, o dentro del kernel para restringirlo a un alcance de kernel.

La memoria compartida dinámica debe ser declarada como un arreglo de una dimensión sin tamaño, por lo tanto, debes calcular los índices de acceso a la memoria en base a los índices de hilos 2D. Como vas a escribir en filas y leer en columnas en este kernel, debes mantener dos índices de la siguiente manera:
* *row_idx*: Compensación de memoria en fila 1D calculado desde índices de hilos 2D
* *col_idx*: Compensación de memoria de columnas 1D calculado desde índices de hilos 2D

Para escribir en la memoria compartida en filas utilizamos *row_idx*:

        tile[row_idx] = row_idx;

Una vez que todo este sincronizado y que la memoria compartida este llena, podemos leerla en orden de columna y asignar a la memoria global:

        out[row_idx] = tile[col_idx];

Esto se debe a que *out* esta almacenada en la memoria global y los hilos dentro de un bloque de hilos están organizados en orden de fila. Es por ello que para escribir en *out* en forma de filas debemos coordinar los hilos para garantizar almacenes fusionados.

El tamaño de la memoria compartida debe de ser especificada cuando se lance el kernel:

        setRowReadColDyn<<<grid, block, BDIMX * BDIMY * sizeof(int)>>>(d_C);
____________________________________________________________________
**Padding Estaticamente Declarado en Memoria Compartida**

Para resolver conflictos de bancos de memeria de una memoria compartida rectangular, podemos utilizar el padding. Sin embargo, para dispositivos Kepler debes calcular cuantos elementos de padding son necesarios. Para facilitar el código, utilizaremos una macro que defina el número de columnas de padding agregadas a cada fila:

        #define NPAD 2
Para declarar el padding estático en la memoria compartida se realiza así:

        __shared__ int tile[BDIMY][BDIMX + NPAD];
Cambiar el número de elementos de padding resulta en un reporte de que las operaciones de carga de la memoria compartida son atendidos por dos transacciones, lo que significa, un conflicto de banco de memoria de dos direcciones.
_____________________________________________________________________
**Padding Dinámico Declarado en Memoria Compartida**

El padding también puede ser utilizado por kernels que utilicen memoria compartida dinámica rectangular. Como el padding de la memoria compartida y la memoria global tendrán diferentes tamaños, se deben mantener tres índices por cada hilo del kernel:
* *row_idx*: Índice de fila para el padding de la memoria compartida. Al usar este índice, el warp puede acceder a una fila de la matriz.
* *col_idx*: Índice de columna para el padding de la memoria compartida. Al usar este índice, el warp puede acceder a una columna de la matriz.
* *g_idx*: Índice a la memoria global lineal. Al usar este índice, un warp puede fucionar accesos a la memoria global.

Estos índices se pueden calcular de la siguiente manera:
        
        unsigned int g_idx = threadIdx.y * blockDim.x + threadIdx.x;
        unsigned int irow = g_idx / blockDim.y;
        unsigned int icol = g_idx % blockDim.y;
        unsigned int row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;
        unsigned int col_idx = icol * (blockDim.x + IPAD) + irow;
Puedes comprobar que el padding de la memoria compartida funciona como se esperaba, reduciendo las transacciones por solicitud.
______________________________________________________________
**Rendimiento de las Memorias Compartidas Cuadradas de los Kernels**

En general, los kernels que utilizan padding de la memoria compartida ganan rendimiento al remover conflictos de bancos de memoria, y los kernels con memoria compartida dinámica reportan una menor cantidad de gastos de recursos.

______________________________________________________________
**Reduciendo el Acceso a la Memoria Global**

Una de las principales razones para utilizar la memoria compartida, es para almacenar datos en la caché del chip. De esta forma reducimos la cantidad de accesos a la memoria global en nuestro kernel. 

El objetivo es reducir los accesos a la memoria global al utilizar la memoria compartida como una caché administrada por el programa.