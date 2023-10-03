<dl>
  <dt>Introducción:</dt>
  <dd>Es un entorno de desarrollo para crear aplicaciones aceleradas por GPU de alto rendimiento. </dd>
  <dt>CUDA Cores:</dt>
  <dd>Procesadores paralelos que se encargan de procesar todos los datos que entran y salen de la GPU. Sus tareas cotidianas son renderizar objetos 3D, dibujar modelos, comprender y resolver la iluminación y sombreado de una escena, etc.</dd>
  <dt>Threads:</dt>
  <dd>Es una secuencia de instrucciones las cuales el sistema operativo puede programar para su ejecución. Los threads son entidades muy pequeñas, lo cual facilita su gestionamiento.</dd>
  <dt>Blocks:</dt>
  <dd>Es un conjunto de hilos(threads) los cuales pueden ser ejecutados en serie o en paralelo. Para una mejor gestión de bloques, los tamaños de estos deben de ser múltiplos de 32, 
  ya que las deformaciones son la mejor granularidad de ejecución posible en la GPU.</dd>
  <dt>Grids:</dt>
  <dd>Conjunto de bloque de hilos. Al igual que los bloques, los grids pueden ser de 1, 2 o 3 dimensiones.</dd>
  <dt>Conclusiones:</dt>
  <dd>Ahora que conocemos los conceptos básicos para trabajar con CUDA, podemos empezar a practicar el manipulamiento básico de imagenes y como la GPU la renderiza.</dd>
  <dt>Bibliografía:</dt>
  <dd>
  The CUDA Handbook: A Comprehensive Guide to GPU Programming, Nicholas Wilt,
  Addison-Wesley Professional, 2013.
    
  CUDA by Example: An Introduction to General-Purpose GPU Programming, Jason
  Sanders, Edward Kandrot, Addison-Wesley Professional, 2010.

  Qué son los Nvidia CUDA Cores y cuál es su importancia. (s.f.). Profesional Review. 
  https://www.profesionalreview.com/2018/10/09/que-son-nvidia-cuda-core/#Que_son_los_CUDA_Cores 

  Understanding CUDA grid dimensions, block dimensions and threads organization (simple explanation). (s.f.). Stack Overflow. 
  https://stackoverflow.com/questions/2392250/understanding-cuda-grid-dimensions-block-dimensions-and-threads-organization-s
  </dd>
</dl>
