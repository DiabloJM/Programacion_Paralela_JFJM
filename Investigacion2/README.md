Introducción:
---------------------
CUDA se ha convertido en el lenguaje predominante para programar GPUs, esto se debe en gran medida a la liberación del NVIDIA CUDA Toolkit al público general en 2007. Con ello, se empezaron a desarrollar softwares que lograran exprimir la máxima potencia de las GPUs.

En el pasado, la búsqueda de incrementar la potencia de las computadoras se enfocaba en aumentar la velocidad de reloj. Sin embargo, la velocidad de reloj se terminó estancando alrededor de los 3.5 GHz, mientras que la cantidad de núcleos en la CPU iban en aumento. Fue entonces donde la programación en paralelo entró al juego y las GPUs ayudaron a llevar el juego al siguiente nivel, con el uso de miles o millones de hilos cooperando para realizar cálculos.

La investigación científica, toda el área médica, la creación de renders y todo el desarrollo de videojuegos requieren del máximo poder que las computadoras puedan ofrecer. Es por ello que este libro es particularmente útil para aquellas personas que puedan utilizar su computadora, equipada con una GPU, para obtener los beneficios de la programación en paralelo.

Es importante recalcar que es necesario tener una GPU de NVIDIA para poder realizar los ejemplos que vienen en este libro. En caso de que nuestro sistema operativo sea Windows, será necesario instalar Visual Studio con la extensión de C++. En caso de que utilices Linux, deberán instalar gcc o g++.

Introducción a los kernels y hardware de GPU
---------------------------------------------
A lo largo de este capítulo se nos irán presentando ejemplos de distintos problemas interesantes. Cada uno de estos ejemplos se nos presentarán de las siguientes tres formas:
* Una implementación estándar de C++ dirigido a un solo hilo de la CPU
* Una versión multihilos para la CPU
* Una versión que utilice CUDA con cientos de hilos al mismo tiempo

En la actualidad, los procesadores tienen dos, cuatro o más núcleos de procesamiento. Para obtener lo mejor de este hardware, es importante que nuestro código sea capaz de correr en paralelo en todos los recursos disponibles. Esto se logra mediante hilos, los cuales permiten lanzar subprocesos cooperativos en cada uno de los núcleos del hardware para obtener una potencial aceleración proporcional al número de núcleos que tenga nuestro hardware.

Para generar una comparativa, los hilos funcionan como un clusters. Un cluster es un sistema de procesamiento paralelo o distribuido conformado por un conjunto de computadoras interconectadas entre sí, de tal manera que funcionan como un solo recurso computacional. Siendo los hilos nuestra herramienta para crear una especie de cluster en nuestra propia computadora.

Para el primer ejemplo, tenemos un código donde se integra el seno de X de 0 a pi utilizando la regla trapezoidal.
![Código para CPU](/Investigacion2/Imagenes/Code1_1.png)
![Código para CPU](/Investigacion2/Imagenes/Code1_2.png)

La función ***sinsum*** es donde se evalúa el seno de x mediante las series de Taylor. Esta función recibe dos parámetros, el primero es el valor de x en radianes y el segundo recibe el número de términos.

Primero se inicializan algunas variables que serán necesarias para realizar la suma. Una vez hecho esto, en un ciclo for se determina el termino, el cual se sumará a la suma total de los términos. Finalmente, la función retorna la suma de todos los términos. 

En el main se declaran algunas variables que serán necesarias para solicitar al usuario la cantidad de pasos y términos que desea emplear, además de inicializar un temporizador para poder saber cuánto tiempo tarda en ejecutarse este código y posteriormente comparar el tiempo con sus otras versiones. Recordemos que esta versión se ejecutará en un solo hilo de la CPU.

Luego se empieza a realizar la integral por medio de las sumas de la serie de Taylor, con ayuda de la función ***sinsum*** y luego se hace la corrección por medio de la regla trapezoidal. Finalmente se imprime en consola el resultado de la integral, los datos ingresados por el usuario y el tiempo que se tardó en realizar el proceso.

Para la segunda versión de este código, el cual se ejecutará en múltiples núcleos de la CPU, se necesita incluir la librería OpenMP.
![Código multihilos para CPU](/Investigacion2/Imagenes/Code2_1.png)
![Código multihilos para CPU](/Investigacion2/Imagenes/Code2_2.png)

Este código es prácticamente el mismo que su versión anterior, solo que se agregan un par de líneas en el main. Se agrega la variable ***threads*** para decirle a OpenMP la cantidad de hilos con los que vamos a trabajar. En la línea 23.5 es donde le decimos a OpenMP con cuantos hilos vamos a trabajar mediante una función de la propia librería que recibe como parámetro nuestra variable ***threads***.

La línea siguiente hace que el ciclo for de la función sinsum se divida en sub-ciclos. La cantidad de sub-ciclos será proporcional a la cantidad de hilos con los que estemos trabajando y sus rangos también serán proporcionales. Cada sub-ciclo será ejecutado en un hilo del CPU de forma paralela y simultanea con los otros hilos.

Al final de cada sub-ciclo este devuelve la suma de los términos dentro de su rango, pero lo que ocupamos es la suma total de los resultados de todos los sub-ciclos. Es por ello que utilizamos una primitiva de paralelismo llamada ***operación de reducción***. Lo que esto hace es sumar los resultados de los sub-ciclos sin importar el orden en que estos terminen y el resultado final lo almacena en la variable ***opm_sum*** para posteriormente mostrarle el resultado al usuario.

Para la tercera versión vamos a utilizar la GPU y CUDA. Esto será un gran cambio ya que la GPU nos permitirá hacer un hilo para cada una de las iteraciones del ciclo for de la función ***sinsum***. 

Lo primero que debemos de hacer es incluir la librería base de CUDA para poder trabajar con él. Además, debemos de incluir la librería de empuje para poder trabajar con los vectores de empuje de la GPU.

La función ***sinsum*** es la misma que utilizamos en la primera versión de este código, sin embargo, a la declaración de esta función se le agregará ***__host__ __device__***. Esto le dirá al compilador que cree dos versiones de esta función, una para la CPU y otra para la GPU.

En la línea 15 se declara una función para el kernel de CUDA la cual llamaremos ***gpu_sin***. Como ya hemos visto en nuestra introducción a CUDA impartida en clase, las funciones de CUDA llevan la palabra clave ***__global__*** en su declaración. Las funciones del kernel pueden recibir parámetros, pero no pueden regresar valores, es por ello que estas funciones siempre deben de ser declaradas de tipo vacío (void).

Lo primero que se declara dentro de la función ***gpu_sin*** es una variable donde determinamos en donde se estará ejecutando la tarea, para así saber dónde está almacenado el resultado de dicha tarea. Y lo que hacemos en el resto de la función es sumar los resultados de todos los hilos y almacenarlos en una sola variable.

La función main es muy similar a la que ya hemos estado utilizando desde la primera versión del código, sin embargo, en esta versión se implementan dos nuevas variables: ***threads*** y ***blocks***. Estas dos variables siempre serán vistas en un programa de CUDA, ya que trabajar con hilos es esencial para la programación en paralelo y las GPUs de NVIDIA procesan los hilos en bloques.
![Código para GPU](/Investigacion2/Imagenes/Code3_1.png)
![Código para GPU](/Investigacion2/Imagenes/Code3_2.png)
![Código para GPU](/Investigacion2/Imagenes/Code3_3.png)
La variable ***threads*** define cuantos hilos habrá en cada bloque y la variable blocks define con cuantos bloques estaremos trabajando. Solo recuerde que la cantidad de hilos debe de ser múltiplo de 32.

En la línea 21.1 inicializamos una variable que funciona como arreglo de CUDA que nos servirá almacenar los resultamos de cada hilo. Posteriormente creamos un puntero que será capaz de acceder a la memoria del vector anteriormente creado.

Luego lo que hacemos es llamar a la función ***gpu_sin*** en todos nuestros bloques de hilos para empezar a hacer los cálculos. Luego utilizamos la ***operación de reducción*** para sumar todos los datos almacenados en el vector ***dsums*** y almacenarlo en una sola variable.

Esta versión del código es un poco menos precisa que las anteriores versiones. Esto se debe a que la versión de CUDA utiliza números flotantes de 4 bytes para hacer los cálculos, mientras que las versiones de la CPU son de 8 bytes. Sin embargo, el resultado de CUDA tiene una precisión de 8 cifras significativas, lo cual es bastante preciso para la mayoría de las aplicaciones científicas y además proporciona este resultado con una velocidad abismalmente mayor a las de las versiones anteriores.

**Arquitectura de la CPU:**

Es cierto que para ejecutar correctamente un código solo se necesita seguir las reglas particulares del lenguaje de programación que estemos utilizando. Sin embargo, el compilador se ejecuta directamente en el hardware. Es por ello que es importante conocer las limitaciones de nuestro hardware al momento de desarrollar software.
![Arquitectura de la CPU](/Investigacion2/Imagenes/CPU.png)
* ***Master Clock:*** Envía pulsos de reloj a una frecuencia fija a cada unidad. La velocidad de procesamiento de la CPU es directamente proporcional a esta frecuencia. Básicamente el reloj es como el director de la CPU.
* ***Memory:*** Aquí es donde se almacena los datos del programa y las instrucciones en código de maquina generadas por el compilador a partir de nuestro código. Esta información puede ser leída por la unidad Load/Save o por la unidad de búsqueda de programa, pero normalmente solo la unidad Load/Save es capaz de escribir información en la memoria.
* ***Load/Save:*** Esta unidad lee y envía información a la memoria. Esta unidad es controlada por la lógica de ejecución, que en cada paso especifica si los datos se leerán o escribirán en la memoria. La información es transferida hacia o desde uno de los registros del fichero de registros.
* ***Register File:*** El fichero de registros es el corazón de la CPU, ya que la información debe ser almacenada en uno o más registros para que pueda ser procesada por la ALU.
* ***Arithmetic Logic Unit (ALU):*** Esta unidad realiza operaciones lógicas y aritméticas en la información almacenada en los registros, cada paso la operación a realizar es especificada por la unidad de ejecución. La ALU es la pieza de hardware que realmente hace cálculos.
* ***Execute:*** La unidad de ejecución decodifica la instrucción enviada por la unidad de búsqueda de instrucciones, organiza la transferencia de datos de entrada al fichero de registros, le dice a la ALU la operación a realizar en la información y finalmente organiza la transferencia de los resultados de vuelta a la memoria.
* ***Fetch:*** La unidad de búsqueda recupera instrucciones de la memoria y las pasa a la unidad de ejecución. La unidad contiene un registro que contiene el contador de programa (PC) la cual contiene la dirección de la instrucción actual. Normalmente la PC es incrementada en 1 para que las instrucciones se sigan de forma secuencial, pero esto puede cambiar en caso de necesitarse una ramificación.

**Potencia de cálculo de la CPU**

La potencia de cálculo ha ido incrementando espectacularmente a lo largo del tiempo. La cantidad de transistores por chip ha tenido un crecimiento exponencial desde 1970 tal como lo predecía la ley de Moore. Este crecimiento exponencial se detuvo en 2002, el rendimiento por núcleo siguió en aumento, pero de forma más lenta. Por lo en vez de aumentar la frecuencia nos hemos enfocado en encontrar nuevos diseños.

Aquí es donde las GPUs entran en juego, ya que ellas no enfrentan ente problema. Aprender sobre el funcionamiento de las GPUs y la programación en paralelo nos permitirá seguir generando cambios, desde el apartado científico hasta el apartado del entretenimiento.

**Gestión de memoria de la CPU: El uso de cachés**

La memoria caché es una capa de almacenamiento de datos de alta velocidad donde se almacenan datos, generalmente transitorios, para que las consultas futuras de dichos datos sean mucho más rápidas y así evitar tener que acceder a la memoria principal.

La memoria caché suele almacenarse en hardware de acceso rápido como la RAM, así podemos aumentar el rendimiento de recuperación de datos evitando acceder a la capa subyacente de almacenamiento, la cual es más lenta.

**Arquitectura de la GPU:**

Hace dos décadas, cuando el interés por el gaming en PC comenzó, no era posible procesar los datos de imágenes en pantalla solo con la CPU. Es allí donde las tarjetas gaming surgieron como un hardware dedicado a hacer los cálculos de pixeles.

Fue entonces donde la programación en paralelo hizo que rápidamente las personas se dieran cuenta que esta tecnología podría ser muy útil para aplicaciones fuera del gaming.

**Tipos de GPU de NVIDIA:**
* La serie GeForce GTX, GeForce RTX o Titan son menos caras y van enfocadas al mercado de los videojuegos.
* La serie Tesla apunta al mercado científico de gama alta. Al no tener puerto de video de salida, estás gráficas no pueden ser utilizadas para el gaming, por lo que suelen ser utilizadas en granjas de servidores.
* La serie Quadro son básicamente tarjetas Tesla con capacidades gráficas adicionales que apuntan al mercado de estaciones de trabajo de escritorio de gama alta.

**Arquitectura Pascal:**

La arquitectura Pascal trajo consigo muchos cambios que fueron cruciales para la tecnología que tenemos hoy en día:
* El núcleo de cálculo era capaz de realizar operaciones básicas de 32 bits, tanto con números flotantes como con enteros. Sus núcleos no tenían contadores de programas individuales (PC).
* Grupos de 32 núcleos estaban agrupados juntos, lo que formó los warps. Estos son la unidad básica de ejecución en CUDA. Se implementaron las unidades de funciones específicas (SFUs) para funciones como seno o exponencial, además de implementar las unidades de doble precisión flotante (FP64).
* Los warp-engine se agrupan para formar lo que NVIDIA denomina como multiprocesadores simétricos o SMs, los cuales generalmente tienen 128 núcleos de cálculo. El SM agrega unidades de textura y varios recursos de memoria en el chip compartidos equitativamente entre los warp-engines.
* Los SMs se agrupan para formar la GPU final.

**Tipos de memoria de la GPU:**
* ***Memoria Principal:*** Aquí es donde el programa y toda la información es almacenada. La CPU puede acceder a ella para escribir o leer información.
* ***Memoria Constante:*** Un espacio de memoria dedicado de 64 KB la cual funciona como la caché de la GPU.
* ***Memoria de Textura:*** Esta se utiliza para almacenar matrices de hasta 3 dimensiones y está optimizada para el direccionamiento local de matrices 2D. Esta memoria está directamente relacionada con el procesamiento de imágenes.
* ***Memoria Local:*** Bloques de memoria privados para cada hilo en ejecución. Se utiliza como almacenamiento para variables locales en resultados temporales intermedios cuando los registros disponibles para un hilo son insuficientes.
* ***Register File:*** El fichero de registros es donde están los registros que son compartidos equitativamente por los bloques de hilos que se están ejecutando a la vez en la SM.
* ***Memoria Compartida:*** Cada SM provee de entre 32 KB a 64 KB de memoria compartida, la cual es esencial para que los hilos puedan comunicarse entre ellos de la forma más rápida.

Pensar y codificar en paralelo:
----------------------------------
**Taxonomía de Flynn:**

Es un conjunto de arquitecturas donde cada una es descrita con 4 letras:
![Flynn](/Investigacion2/Imagenes/TaxonomiaFlynn.png)
SISD representa la más "normal" de todas, donde un solo proceso se ejecuta en un solo hilo.
SIMD representa la arquitectura donde una sola instrucción puede ejecutarse en múltiples objetos de información al mismo tiempo.
MIMD es donde múltiples instrucciones se ejecutan en múltiples objetos de información. Un ejemplo claro de esa arquitectura serían los servidores o clusters.
MISD es el más raro de todos, consiste en que múltiples instrucciones se ejecuten en un solo objeto de información. Suele utilizarse en satélites, donde se requiere redundancia contra errores.
Por último, SIMT es una variación de SIMD creada por NVIDIA para describir la arquitectura de sus GPUs. La mayor diferencia entre uno y otro es que SIMT utiliza una cantidad mucho mayor de hilos para procesar objetos individuales de información.

**Sintaxis para llamar al kernel:**

La forma general de llamar al kernel de CUDA es por medio del uso de hasta cuatro argumentos dentro del siguiente espacio: <<< >>> y después los respectivos argumentos de la función.

El primer argumento define las dimensiones de la grid de bloque de hilos que va a utilizar el kernel.

El segundo argumento define el número de hilos que hay por bloque.

El tercer argumento, el cual es opcional, define el número de bytes de memoria compartida utilizada por cada bloque del kernel. No se reserva memoria compartida si este argumento es omitido es establecido en cero.

El cuarto argumento, el cual también es opcional, especifica la secuencia de CUDA con la cual se ejecutará el kernel. Este argumento solo es necesario en aplicaciones avanzadas donde se ejecutan múltiples kernels de forma simultánea.

**Parámetros del Paralelismo:**

La programación de las GPUs no solo requiere de una basta cantidad de hilos, sino de una correcta planeación de nuestro código. Para mejorar el rendimiento, es importante evitar situaciones que sean problemáticas para CUDA.

Situaciones donde un if hace que un hilo ejecute una función u otra, hace que el rendimiento se vea gravemente afectado. Ya que lo que hace el sistema es que los hilos que cumplan la condición del if ejecutan su respectiva función, mientras que los hilos que no la cumplan se mantienen pausados hasta que todos los hilos que si cumplieron la condicional terminen de ejecutar su función.

**Reduce:**
La operación ***reduce*** es el ejemplo perfecto de realizar la misma operación en un largo grupo de números. La operación en si consiste en encontrar la suma aritmética de los números, una operación que ya vimos en un ejemplo pasado.

**Ocupación:**

La ocupación es la cantidad de warps activos en la GPU en ese momento. El valor de ocupación va de 0 a 1 y esta se puede calcular de la siguiente manera:

![Ocupacion](/Investigacion2/Imagenes/Ocupacion.PNG)

Mientrás más alto sea el valor de la ocupación, más parte de la GPU esta en uso. El hecho de que no siempre sea 1 no significa que haya un error en tu código o en la GPU, simplemente no siempre es necesario utilizar el 100% de la GPU.

**Memoria Compartida:**

Como ya vimos con anterioridad, la memoria compartida es un acceso de memoria rápida de un tamaño de 64 KB disponible en cada SM. Podemos hacer que nuestros arreglos, de tipo escalar, vaya a la memoria compartida. Esto mediante el prefijo __shared__.

Ejemplo:    __shared__ float data[256];

**Restrict:**

El preijo __restrict es utilizado para los punteros y ayuda a mejorar un 44% el rendimiento, ya que le dice al compilador que el puntero no tiene un alias y la optimización agresiva es segura. Esto se hace ya que cuando un puntero es declarado de la forma tradicional, el compilador no tiene forma de saber si no hay otros punteros en la misma dirección de memoria en otra parte del código, por eso el compilador le pone un alias al punteo por si es que hay otro puntero en esa dirección de memoria. Actualmente el poner alias a los punteros es algo completamente innecesario pero es una práctica que los compiladores siguen teniendo.

Ejemplo:   float* __restrict C

Conclusión:
------------
En la actualidad los grandes programas requieren de utilizar recursos más allá de la CPU, es por ello que acudimos a la GPU. Aprender sobre la comunicación entre la CPU y la GPU, el manejo de hilos y una correcta lógica de nuestro código hace que podamos aprovechar estos recursos que tenemos a la mano.

Sin embargo, saber sobre la creación correcta de software no es suficiente. Conocer las limitaciones de nuestro hardware nos permite saber cuanto podemos exigirle, ya que el desconocimiento de este hará que desperdiciemos valiosos recursos.

El conocimiento del funcionamiento de la CPU y la GPU nos ayuda a entender como se mueve la información con la que trabajamos y como esta es operada. Saber del pasado del hardware nos ayuda a apreciar el presente de este, pero a su vez, nos ayuda a plantearnos nuevas posibilidades para el futuro de este.

Durante estos tres años de carrera nos hemos estado enfocando en el desarrollo correcto de software, y a pesar de que este no es un mal enfoque si es una versión incompleta de la industria de la que planeamos formar parte.

Si vamos a desarrollar un videojuego para una consola primero debemos conocer las capacidades de la misma, debemos comprender su funcionamiento, debemos aprender a hacer una correcta armonía entre software y hardware. Sin duda estos primeros dos capítulos han sembrado la semilla para poder lograr ese objetivo.

Bibliografía:
---------------
R. Ansorge, Programming in parrallel with CUDA: A practical guide. United Kingdom: TJ Books, 2022.

“¿Qué es la velocidad del procesador y por qué es importante?” Laptop-Computer, Desktops, Drucker, Tinte und Toner | HP® Deutschland. Accedido el 14 de octubre de 2023. [En línea]. Disponible: https://www.hp.com/cl-es/shop/tech-takes/que-es-la-velocidad-del-procesador-y-por-que-es-importante#:~:text=La%20velocidad%20del%20reloj%20del%20procesador%20de%20un%20computador%20determina,tareas%20al%20hacerlas%20más%20rápido.

“¿Qué es un cluster?” RDU - UNAM. Accedido el 14 de octubre de 2023. [En línea]. Disponible: https://www.revista.unam.mx/vol.4/num2/art3/cluster.htm

A. Kelleher. “Ley de Moore - Ahora y en el Futuro”. Intel newsroom. Accedido el 14 de octubre de 2023. [En línea]. Disponible: https://www.intel.la/content/www/xl/es/newsroom/opinion/moore-law-now-and-in-the-future.html

“Qué es el almacenamiento en caché y cómo funciona | AWS”. Amazon Web Services, Inc. Accedido el 15 de octubre de 2023. [En línea]. Disponible: https://aws.amazon.com/es/caching/
