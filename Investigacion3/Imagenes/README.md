**memTransfer.cu**
_________________________________
Este código consiste en transferir información de el host a device y viceversa. En su análisis podemos ver que hay tres partes del código que se llevan el mayor tiempo de compilado.

![memTransfer](/Examen2/Imagenes/memTransfer.png)

El primero es el cudaMalloc, el cual obviamente requiere de bastante tiempo de compilación ya que esta función se encarga de asignar una cantidad de bytes de la memoria global.  

El segundo sería cudaMemcpy, el cual es utilizado dos veces pero con tipos de copia distintos: cudaMemcpyHostToDevice y cudaMemcpyDeviceToHost.

Esta función, con sus respectivos tipos de copia, sirve para transferir información de host a device o viceversa. Al ser una transferencia de muchos datos, esto lleva tiempo.

**pinMemTransfer.cu**
______________________________________
Este código se encarga de hacer transferencia de datos haciendo uso de la memoria fijada, tiene 3 procesos que consumen la mayor parte del tiempo de compilación.

![pinMemTransfer](/Examen2/Imagenes/pinMemTransfer.png)

El primero es cudaHostAlloc, ya que lo que hace es asignar un espacio de memoria en la memoria fijada.

La segunda es cudaMemcpy host to device, que se encarga de hacer una transferencia de información de host a device.

Y la tercera es cudaMemcpy device to host, que se encarga de hacer una transferencia de información de device a host.

**sumArrayZerocopy.cu**
_______________________________________
Este código consiste en hacer una suma de arreglos utilizando la memoria de copia cero. En este código podemos encontrar 5 procesos que se llevan el mayor tiempo de compilación.

![sumArrayZerocopy](/Examen2/Imagenes/sumArrayZerocpy.png)

El primero es el cudaMalloc, el cual ya expliqué que se encarga de asignar memoria en la memoria global y es por ello que toma tiempo, además de que es llamada 3 veces en este código.

La segunda es la función sumArraysZeroCopy, la cual es una función que se encarga de sumar arreglos con la ayuda de la memoria de cero copia. Al utilizar esta memoria, el host y el device deben de estar bien sincronizados, por lo que puede llevar tiempo.

El tercero es el cudaMemcpy de device a host, el cual al ser una transferencia de datos que se realiza dos veces durante el código, es una operación que lleva tiempo.

La cuarta es la función sumArrays, la cual es una función de suma de arreglos que se ejecuta en el host, pero al trabajar con muchos datos esta se toma su tiempo.

La quinta es cudaMemcpy de host a device, esta al ser una transferencia de información que se llama 2 veces a lo largo del código, tiene sentido que tarde un poco más.

**readSegment.cu**
_____________________________________
Este código se encarga de leer memoria con offset. Este tiene 6 puntos importantes donde se toma la mayor parte del tiempo de compilación.

![readSegment](/Examen2/Imagenes/readSegment.png)

Los primeros 3 son cudaMalloc (llamado 3 veces), cudaMemcpy device to host (llamado 1 vez) y cudaMemcpy host to device (llamado 2 veces). Estos 3 ya he explicado porque requieren de más tiempo así que voy a pasar a los otros 3.

El siguiente es cudaDeviceReset, que como su nombre lo indica, resetea al device por completo. Sabemos por experiencia que cualquier reset se toma su tiempo.

El siguiente es readOffset, función que se encarga de leer el número de offsets y por la cantidad de información puede tomar un poco más de tiempo.

Por último tenemos a la función warmup, la cual hace lo mismo que la función anterior.

**readSegmentUnroll.cu**
____________________________________
Esté código se encarga de leer memoria con offset de forma desenrollada. Este código tiene 5 funciones que se llevan la mayor parte del tiempo de compilación.

![readSegmentUnroll](/Examen2/Imagenes/readSegmentUnroll.png)

El primero es cudaMalloc, llamado 3 veces en esté código y el encargado de asignar memoria en memoria global.

El segundo es cudaMemcpy de device a host, llamado 3 veces en este código y el encargado de tranferir información de device a host.

El tercero es cudaMemcpy de host a device, llamado solo 2 veces en el código y el encargado de transferir información de host a device.

El cuarto es cudaDeviceReset, encargado de resetear la GPU al final del código.

El último es cudaMemset, llamado 4 veces en el código y el encargado de setear todos los elementos de un arreglo de CUDA al valor que deseemos.

**writeSegment.cu**
____________________________________
Esté código se encarga de la escritura en memoria con offset. Las cuatro funciones que toman el mayor tiempo de compilado son las mismas de siempre: cudaMalloc, cudaMemcpy de device a host, cudaMemcpy de host a device y cudaDeviceReset. Sin embargo, la quinta es la más interesante. 

![writeSegment](/Examen2/Imagenes/writeSegment.png)

La quinta función que más tiempo de compilación toma es writeOffset. A pesar de que solo es llamada una sola vez en todo el código, recordemos que escribir en memoria es mucho más tardado que solo leearla. Es por ello que toma un tiempo significativo de tiempo.

**simpleMathAoS.cu**
____________________________________
Este código se encarga de hacer operaciones simples con el uso de arreglo de estructuras. Tiene 3 funciones que se llevan la mayor parte del tiempo de compilación.

![simpleMathAoS](/Examen2/Imagenes/simpleMathAoS.png)

El primero es cudaMalloc, ya que se llama 2 veces y se encarga de asignar memoria en la memoria global.

La segunda es cudaMemcpy de device a host, la cual es llamada 2 veces y se encarga de transferir información de device a host.

Por último tenemos a cudaMemcpy de host a device, quien se encarga de transferir información de host a device.

**simpleMathSoA.cu**
_______________________________________
Este código se encarga de hacer operaciones simples con el uso de estructura de arreglos. Tiene 4 principales funciones que se llevan el mayor tiempo de compilación.

![simpleMathSoA](/Examen2/Imagenes/simpleMathSoA.png)

Los primeros 3 son los mismos que el anterior; cudaMalloc, cudaMemcpy de device a host y cudaMemcpy de host a device.

Pero el cuarto es cudaDeviceReset, función que se encarga de resetear la GPU y puede tomarse su tiempo.

**transpose.cu**
_______________________________________
Este código se encarga de resolver un problema básico de álgebra, transponer una matriz. Esté código tiene 5 principales funciones que se llevan la mayor parte del tiempo de compilación.

![transpose](/Examen2/Imagenes/transpose.png)

El primero es cudaMemcpy de host a device, que tomando en cuenta que estamos hablando de matrices, tiene sentido que tarde más tiempo en transferir estos datos que en códigos pasados.

El segundo es cudaMalloc, el cual es llamado 2 veces y al hablar de matrices requiere asignar mayor espacio de memoria.

El tercero es cudaDeviceReset, encargado de resetear la GPU.

La cuarta es la función copyRow, encargada de copiar los datos de una fila de la matriz para luego transponerla.

Y la quinta es la función warmup, encargada de evitar una sobrecarga en el inicio.

**sumMatrixGPUManaged.cu**
_________________________________________
Esté código se encarga de hacer sumas de matrices por medio de la GPU administrada. Este código tiene 2 funciones que se llevan el mayor tiempo de compilación.

![sumMatrixGPUManaged](/Examen2/Imagenes/sumMatrixGPUManaged.png)

El primero es la función sumMatrixGPU, la función encargada de hacer las sumas de las matrices y es por ello que se entiende porque se toma tanto tiempo.

La segunda es cudaMallocManaged, que a diferencia de cudaMalloc, esta se toma más tiempo pero esta mejor optimizada para estructura de datos más grandes y complejas.

**sumMatrixGPUManual.cu**
_________________________________________
Esté código se encarga de hacer sumas de matrices por medio de la GPU de la forma normal. Este código tiene 3 funciones que se llevan la mayor parte del tiempo de compilación.

![sumMatrixGPUManual](/Examen2/Imagenes/sumMatrixGPUManual.png)

La primera es cudaMalloc, encargada de asignar memoria a la memoria global.

La segunda es cudaMemcpy, tanto HtoD como DtoH, encargado de hacer la transferencia de información entre host y device.

Por último tenemos a la función sumMatrixGPU, la encargada de hacer la suma de matrices en la GPU. Al ser muchos calculos, es normal que le cueste tiempo hacerlos.