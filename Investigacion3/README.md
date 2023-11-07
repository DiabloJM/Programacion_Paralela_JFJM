**Modelo de Ejecución de CUDA:**
__________________________________
Un módelo de ejecución proporciona una visión operativa de cómo se ejecutan las instrucciones en una arquitectura informática específica.

El modelo de jecución de CUDA muestra una visión abstracta de la arquitectura paralela de la GPU, lo que nos permite razonar sobre la concurencia de subprocesos.

CUDA utiliza la arquitectura *Single Instruction Multiple Thread (SIMT)*, un derivado de SIMD.

**Arquitectura de Fermi:**

Fue la primera arquitectura informática de GPU completa que ofreció las funciones necesarias para las aplicaciones HPC más exigentes. Fue ampliamente adoptada para acelerar cargas de trabajo de producción.

![Fermi](/Investigacion3/Imagenes/Fermi.png)

Fermi cuenta con hasta 512 núcleos de acelerador, llamados *CUDA cores*.

Cada *CUDA core* tiene su propia ALU y su propia *floating-point unit (FPU)* la cual ejecuta una instrucción de entero o punto flotante por ciclo de reloj. Los *CUDA cores* estan organizados en 16 *streaming multiprocessors (SM)*, cada uno con 32 CUDA cores.

Fermi cuenta con 6 interfaces de memoria  de 384-bit GDDR5 DRAM, dando un total de 6 GB de memoria global, esto fue un componente clave para muchas aplicaciones, y una caché de 768 KB L2 compartida entre las 16 SMs.

Una característica clave de Fermi es la memoria configurable en chip de 64 KB, la cual esta dividida entre memoria compartida y caché L1. CUDA proporciona una API que ayuda a ajustar la cantidad de memoria compartida y caché L1.

Por último, Fermi soporta la ejecución de múltiples kernels de una misma aplicación al mismo tiempo (un máximo de 16 kernels), lo que permitía utilizar por completo la GPU

**Arquitectura de Kepler:**

La arquitectura Kepler es una arquitectura informática rápida, de alto rendemiento y altamente eficiente que ayudo a que la informática híbrida fuera más accesible.

Esta arquitectura contiene 15 SMs y 6 controladores de memoria de 64 bits. Las 3 innovaciones importantes de la arquitectura Kepler son:

* SMs mejorados
* Paralelismo dinámico
* Hyper-Q

![Kepler](/Investigacion3/Imagenes/Kepler.png)

Cada SM consistia de 192 CUDA cores de precisión simple, 64 unidades de precisión doble, 32 unidades de función especial (SFU) y 32 unidades load/store (LD/ST).

El *Paralelismo dinámico* fue una nueva característica introducida en las GPUs Kepler que le permitían a la GPU lanzar nuevas grids de forma dinámica. Con esta característica, cualquier kernel podría lanzar otro kernel y gestionar las dependencias necesarias entre kernels para realizar correctamente el trabajo adicional.

Anteriormente la CPU lanzaba cada kernel en la GPU, pero con el paralelismo dinámico, la GPU puede lanzar kernels anidados, eliminando la necesidad de comunicarse con la CPU constantemente. 

![Paralelismo_Dinamico](/Investigacion3/Imagenes/ParalelismoDinamico.png)

El *Hyper-Q* agrega más conexiones de hardware simultáneas entre la CPU y la GPU, permitiendo a los núcleos de la CPU ejecutar más tareas simultáneamente en la GPU. Esto dio como resultado que incrementara el uso de la GPU y disminuyera el tiempo de espera de la CPU.

**Entendiendo la naturaleza de la ejecución de los warp:**
_______________________________________________________________

**Warps y bloques de threads:**

Un warp es un conjunto de 32 threads los cuales trabajan con las mismas instrucciones pero cada thread trabaja con información diferente, tal como lo indica el *Single Instruction Multiple Thread (SIMT)*.

Los bloques de threads pueden ser configurados para ser de una, dos o tres dimensiones. Sin embargo, desde la perspectiva del hardware, todos los threads son unidimensionales (teniendo un ID para CUDA).

Un bloque de threads de dos o tres dimensiones puede convertirse en una dimensión física usando la dimensión X como la dimensión más interna, luego Y como la segunda dimensión y finalmente Z como la dimensión más externa.

En un bloque de threads 2D, un identificador único por cada thread en el bloque puede ser calculado de la siguiente manera:

***threadIdx.y * blockDim.x + threadIdx.x***

En uno 3D se calcula de la siguiente manera:

***threadId.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x***

El número de warps por bloque de threads puede ser determinado de la siguiente manera:

![WarpsPerBlock](/Investigacion3/Imagenes/WarpsPerBlock.png)

**Divergencia de Warps**

Las CPUs tienen poseen una *predicción de ramificación*, lo que les permite predicir en cada condicional que rama va a tomar el flujo de control. Si esta predicción es correcta, ramificar en CPU conlleva solo una pequeña penalización de rendimiento. Si es incorrecta, la CPU puede detenerse durante varios ciclos a medida que se vacía el canal de instrucciones.

Las GPUs no poseen predicción de ramificación ya que todos los hilos de un warp deben ejecutar instrucciones identicas en el mismo ciclo. Esto se llegar a ser un problema si hilos en el mismo warp toman diferentes caminos durante la aplicación.

Hilos de un mismo warps ejecutando diferentes instrucciones causa lo que se conoce como **divergencia de warp**. Esto afecta gravemente al rendimiento, ya que mientrás unos hilos trabajan con unas instrucciones, los otros esperan a que los primeros acaben para poder trabajar con otro set de instrucciones.

Para evitar esto, podemos darle condicionales diferentes a diferentes warps, así evitamos divergencia. Pero si realmente queremos evitar este problema, lo mejor es evitar situaciones que causen ramificación.

**Eficiencia de rama**

La *eficiencia de rama* se define como la relación entre ramas no divergentes y ramas totales. Se puede calcular de con la siguiente fórmula:

![BranchEfficiency](/Investigacion3/Imagenes/BranchEfficiency.png)

**Ocultar la latencia**

La latencia es el número de ciclos de reloj entre una instrucción que se emite y una instrucción que se completa.

En CUDA es de suma importancia aprender a ocultar la latencia. La latencia de las instrucciones de la GPU puede ocultarse mediante el cálculo de otros warps.

Las intrucciones de la GPU se dividen en dos:

* Instrucciones aritméticas
* Instrucciones de memoria

Es así como con estos tipos de instrucciones podemos ocultar la latencia.

**Ocupancia**

La ocupancia es la cantidad de warps que están siendo utilizados dentro de un SM. Esta se puede calcular de la siguiente forma:

![Ocupancia](/Investigacion3/Imagenes/Ocupancia.png)

Apuntamos a tener una ocupancia de 1 (o 100%), sin embargo no siempre va a ser así ya que no siempre es necesario utilizar el 100% de la GPU.

**Sincronización**

La sincronización de barrera es una primitiva común en muchos lenguajes de programación en paralelo. En CUDA, la sincronización puede ser realizada en dos niveles:

* **Nivel de Sistema:** Esperar que todos los trabajos, tanto en host como en device, sean completados
* **Nivel de Bloque:** Esperar a que todos los hilos, en un bloque de hilos, alcancen el mismo punto de ejecución en device

Ya que muchas API de CUDA fueron llamadas y todos los kernel lanzados son asincronos con respecto al host, podemos utilizar *cudaDevicesSynchronize* para bloquear al host hasta que todas las operaciones de CUDA hayan terminado. Ejemplo:

*cudaError_t  cudaDeviceSynchronize(void)*

Ya que los warps de un bloque de hilos son ejecutados sin un orden definido, CUDA nos da la habilidad de sincronizar su ejecución con una barrera de bloque local. Podemos hacerlo de la siguiente forma:

![Sincronizacion](/Investigacion3/Imagenes/Sincronizacion.png)

Cuando utilizamos *__syncthreads*, cada hilo en el mismo bloque de hilos debe de esperar a que el resto de hilos alcancen este punto de sincronización. Esta función es utilizada para coordinar la comunicación entre los hilos en el mismo bloque, sin embargo esto puede afectar de forma negativa al rendimiento al forzar a los warps a quedarse en espera.

**Condiciones de carrera**

Los hilos de un mismo bloque de hilos pueden compartir información por medio de la memoria compartida y los registros. Al compartir información entre hilos, debemos ser cuidadosos de evitar condiciones de carrera.

Las condiciones de carrera, o peligros, son accesos desordenados, por múltiples hilos, a la misma localización de memoria.

No hay sincronización de hilos entre diferentes bloques. La única forma de bloques es por medio de un punto de sincronización global al final de cada ejecución del kernel.

**Escalabilidad**

La escalabilidad implica que al proveer recursos de hardware adicionales a una aplicación paralela produce una aceleración en relación con la cantidad de recursos agregados. Ejemplo: si una aplicación de CUDA es escalada a dos SMs, su tiempo de ejecución será la mitad de lo que sería con un solo SM.

La escalabilidad implica que el rendimiento puede mejorar al agregar núcleos computacionales. Esto puede ser más importante que la eficiencia.

La habilidad de ejecutar el mismo código en una cantidad distinta de núcleos computacionales se le conoce como *escalabilidad transparente*. Una plataforma transparente escalable amplía los casos de uso para las existentes aplicaciones y reduce la carga de los desarrolladores porque pueden evitar realizar cambios para nuevos o distintos hardwares.


_____________________________________
**Memoria Global:**
_____________________________________

**Jerarquia de Memoria**

Actualmente las computadoras utilizan una jerarquía de memoria de latencia progresivamente menor, pero de menor capacidad, para optimizar el rendimiento. 

La jerarquía de memoria consiste en múltiples niveles de memoria, con difetente latencia, anchos de banda y capacidades. A medida que la latencia de un espacio de memoria aumenta, también aumenta su capacidad.

![Jerarquia](/Investigacion3/Imagenes/Jerarquia.png)

**Modelo de Memoria CUDA**

Existen dos clasificaciones de memoria:
* *Programble:* Tú contralas que datos colocas en la memoria programable
* *No Programable:* No controlas la ubicación de los datos y dependes de técnicas automáticas para lograr un buen rendimiento

CUDA ofrece distintos tipos de memoria programable:
* Registros
* Memoria Compartida
* Memoria Local
* Memoria Constante
* Memoria de Textura
* Memoria Global

**Registros**

Los registros son los espacios de memoria más rápidos en una GPU. Las variables declaradas en un kernel son generalmente almacenadas en los registros, los arreglos también pueden ser almacenados en registros siempre y cuando el tamaño del arreglo sea constante y que sus indices puedan determinarse en el momento de la compilación.

Las variables en los registros son privadas para cada hilo. Las variables en registros comparten su tiempo de vida con el kernel, una vez que el kernel complete su ejecución, no se puede volver a acceder a la variable del registro. Los registros son utilizados por el kernel para almacenar variables privadas que tienden a ser consultadas con frecuencia.

Si un kernel utiliza más registros de lo que el hardware puede proveer, los registros excedentes serán *derramados* a la memoria local. El *derrame de memoria* puede tener consecuencias en el rendimiento, además de perder el control de la ubicación de nuestros datos.

**Memoria Local**

Las variables de un kernel que sean enviadas a registros, pero que por alguna razón no quepan en un registro, serán enviadas a la memoria local. Las variables que el compilador suele enviar a la memoria local son:
* Arreglos locales a los que se hace referencia con índices cuyos valores no se pueden determinar en tiempo de compilación
* Estructuras locales muy grandes o arreglos muy grandes para un registro
* Cualquier variable que no se ajuste al límite del registro del kernel

El nombre "Memoria Local" es bastante engañoso, ya que las variables almacenadas en la memoria local se encuentran en la misma ubicación física que la memoria global. Lo que diferencia a la memoria local de la global es su alta latencia  y su bajo ancho de banda.

**Memoria Compartida**

Para enviar variables a la memoria compartida, utilizamos el atributo "__shared__".

Como la memoria compartida esta en el chip, esta tiene mucho más ancho de banda y menor latencia que la memoria local y la global. Pero esta no debe de ser sobre utilizada, ya que podrías limitar el numero de warps activos.

Cuando un bloque de hilos termina su ejecución, el espacio de memoria que ocupaban en la memoria compartida es liberada y asignadaa otro bloque de hilos.

Para acceder a la memoria compartida tenemos que estar sincronizados, es por ello que utilizamos: void __syncthreads();

Con esto creamos una especie de "check-point", donde todos los hilos de un mismo bloque deben alcanzar, y así, todos puedan continuar al mismo tiempo.

La caché L1 y la memoria compartida usan los mismos 64 KB de memoria en el chip.

**Memoria Constante**

La memoria constante se encuentra en la memoria del dispositivo y se almacena en caché en una memoria dedicada. Para declarar una variable constante, utilizamos el atributo: "__constant__".

Las variables constantes deben declararse con alcance global, fuera de cualquier kernel.

La memoria constante se declarada estaticamente y es visible por todos los kernels en la misma unidad de compilación. Recuerden que el kernel SOLO puede leer la memoria constante.

La memoria constante funciona mejor cuando todos los hilos de un warp leen de la misma dirección de memoria.

**Memoria de Textura**

La memoria de textura es un tipo de memoria global a la cual se accede a través de una caché dedicada de solo lectura. 

La memoria de tectura está optimazada para localidad espacial 2D, para que los hilos de un warp que utilicen esta memoria puedan acceder a información 2D con un gran rendimiento. Esto puede ser muy útil para algunas aplicaciones pero puede ser más lento para otras aplicaciones que utilizar la memoria global.

**Memoria Global**

La memoria global es la qué más ancho de banda tiene, la de mayor latencia y la más comunmente usada en la GPU. El termino *global* se refiere a su alcance y a su tiempo de vida. Esta puede ser accedida desde la GPU desde cada SM durante el tiempo de vida de la aplicación.

Una variable en memoria global puede ser declarada de forma estática o dinámica. Para declarar una variable global de forma estática, utilizamos el atributo: "__device__".

La memoria global es asignada por el host usando *cudaMalloc* y liberada por el host utilizando *cudaFree*. Los punteros a meoria global son utilizados en el kernel como parametros de funciones. La asignación de la memoria global existe durante toda la vida de la aplicación y es accesible por todos los hilos de todos los kernels.

**Cachés de la GPU**

Como las cachés de la CPU, las cachés de la GPU son memorias no programables. Hay cuatro tipos de caché en la GPU:
* L1
* L2
* Constante de solo lectura
* Textura de solo lectura

Hay una caché L1 para cada SM y una caché L2 compartida por todos los SMs. Ambas son utilizadas para almacenar información en memoria local y global, incluyendo derrame de registros.

**Declaración de Variables en CUDA**

La siguiente tabla muestra la declaración de variables en CUDA y su correspondiente locación de memoria, alcance, tiempo de vida y atributo.

![VariableTable](/Investigacion3/Imagenes/VariableTable.png)
![VariableTable](/Investigacion3/Imagenes/VariableTable2.png)

La cruz simboliza que la variable puede ser escalar o un arreglo.

**Asignación y Desasignación de Memoria**

Las funciones del kernel operan en el espacio de la memoria del dispositivo, CUDA ofrece funciones que permiten asignar y desasignar memoria en el dispositivo. Para asignar memoria global en el host, utilizamos la siguiente función:

    cudaError_t  cudaMalloc(void **devPtr, size_t count);

Esta función asigna una cantidad de bytes de la memoria global en el dispositivo y regresa la localización de la memoria en el puntero *devPtr*

Una vez que la aplicación ya no este utilizando ni una sola parte de la memoria global asignada, esta puede ser desasignada con la función:

    cudaError_t  cudaFree(void *devPtr);

Esta función libera la memoria global a la que apuntaba *devPtr*, la cual antes fue asignada con la función cudaMalloc.

**Transferencia de Memoria**

Una vez que la memoria global haya sido asignada, podemos transferir información del host usando la función:

    cudaError_t  cudaMemcpy(void *dst, const oid *src, size_t count, enum cudaMemcpyKind kind);

Esta función copia la *cantidad* de bytes de la locación de memoria *src* a la locación de memoria *dst*. Los *tipos* de copia son los siguientes:
* cudaMemcpyHostToHost
* cudaMemcpyHostToDevice
* cudaMemcpyDeviceToHost
* cudaMemcpyDeviceToDevice

