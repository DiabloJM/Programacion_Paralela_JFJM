**Hola Imagen**

_Imagen:_
Es una representación visual de objetos o elementos. Esta formada por una matriz de pixeles,
también conocido como mapa de bits.

_OpenCV:_
Es una herramienta utilizada para el procesamiento de imagenes y para realizar tareas relacionadas
a todo el apartado visual. Al ser de código abierto, suele ser utilizado para detección fácial,
rastreo de objetos, detección de puntos de referencia, etc.

_Canal de colores:_
Un canal es tu imagen en escala de grises hecha de un solo color primario. Cada valor muestra cuanta
intensidad, del color primario del canal, hay en cada pixel.

_Mapas de colores:_
Es un conjutno de valores asociados con colores. Se utilizan para mostrar un ráster de una sola banda 
de manera consistente con los mismos colores.

_Código:_
```c++
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <filesystem>

int main() {
    std::string image_path;
    std::cout << "Enter the path to the image:";
    std::cin >> image_path;

    if (!std::filesystem::exists(image_path)) {
        std::cout << "File does not exist at the specified path" << std::endl;
        return -1;
    }

    cv::Mat image = cv::imread(image_path);

    if (image.empty()) {
        std::cout << "Error loading the image" << std::endl;
        return -1;
    }
    else {
        std::cout << "Image loaded successfully" << std::endl;
    }

    cv::imshow("Image", image);

    //Wait for a keystroke in the window
    cv::waitKey(0);

    // Separate the image into its three channels
    cv::Mat bgr[3];
    cv::split(image, bgr);

    //Modify images by color maps
    cv::Mat blueChannel, greenChannel, redChannel;
    cv::applyColorMap(bgr[0], blueChannel, cv::COLORMAP_PINK);
    cv::applyColorMap(bgr[1], greenChannel, cv::COLORMAP_HSV);
    cv::applyColorMap(bgr[2], redChannel, cv::COLORMAP_INFERNO);

    // Create modified image windows
    cv::imshow("blueChannel", blueChannel);
    cv::imshow("greenChannel", greenChannel);
    cv::imshow("redChannel", redChannel);
    cv::waitKey(0);

    return 0;
}
```

_Documentación:_
La librería iostream será utilizada principalmente para pedirle al usuario la ruta de la imagen
que desea abrir.
Las librerias opencvv.hpp y core.hpp serán utilizadas para poder trabajar con OpenCV.
Y la librería filesystem será utilizada para poder acceder a los archivos de nuestra computadora.

Iniciamos el código solitandole al usuario la ruta de la imagen que desea abrir. Si la ruta que nos
brinda el usuario no existe, se le dará un mensaje al usuario de que esa ruta no existe y el programa
finalizará.

De lo contrario, la ruta será procesada y almacenada en una variable de tipo matriz. Si la ruta que nos
brindo el usuario resulta que no es una imagen, se le dirá al usuario que hubo un error al cargar la
imagen y se terminará el programa. De lo contrario, le diremos al usuario que la imagen se cargo con 
éxito y mostraremos la imagen.

El programa se mantendrá en pausa hasta que el usuario presione una tecla, después de esto, separaremos
la imagen en tres canales.

Crearemos una variable de tipo matriz por cada uno de los canales y aplicaremos un mapa de colores a cada
canal.

Finalmente, mostraremos los tres canales que creamos y el programa finalizará cuando el usuario presione 
una tecla.



