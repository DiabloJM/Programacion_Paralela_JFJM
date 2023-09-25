Multiplicación de Matrices

Definición de matriz:
Conjunto bidimensional de números o símbolos distribuidos de forma rectangular, en líneas verticales y horizontales, de manera que sus elementos se organizan en filas y columnas. Sirven para describir sistemas de ecuaciones lineales o diferenciales, así como para representar una aplicación lineal.

Definición de multiplicación de matrices:
Consiste en combinar linealmente dos o más matrices mediante la adición de sus elementos dependiendo de su situación dentro de la matriz origen respetando el orden de los factores.

Proceso para la multiplicación de matrices:
Dadas las matrices Z y Y de n filas y m columnas:

![Matrices cuadradas en orden n](https://economipedia.com/wp-content/uploads/Captura-de-pantalla-2019-10-10-a-les-15.03.40.png)

Podemos multiplicar las matrices anteriores si el número de las filas de la matriz Z es igual al número de columnas de la matriz Y.
La dimensión de la matriz resultado es la combinación de la dimensión de las matrices. En otras palabras, la dimensión de la matriz resultado serán las columnas de la primera matriz y las filas de la segunda matriz.
Una vez determinado que podemos multiplicar las matrices, multiplicamos los elementos de cada fila por cada columna y los sumamos de la forma que solo quede un número en el punto donde los óvalos azules anteriores coindicen.

![Esquema de multiplicación de matrices](https://economipedia.com/wp-content/uploads/Captura-de-pantalla-2019-10-10-a-les-15.19.33.png)

Sumamos las multiplicaciones de cada elemento y obtendremos nuestra matriz resultado.
Código:
```
#include<stdio.h>
#include<stdlib.h>

int main()
{
	int matrizA[10][10];
	int matrizB[10][10];
	int resultado[10][10];
	int suma = 0;

	//Inicializacion de las matrices
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			matrizA[i][j] = rand() % 10;
			matrizB[i][j] = rand() % 10;
		}
	}

	//Calcular la multiplicación de las matrices
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			for (int k = 0; k < 10; k++)
			{
				suma += matrizA[k][j] * matrizB[k][j];
			}
			resultado[j][i] = suma;
			suma = 0;
		}
	}

	//Imprimir la matriz resultante
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			printf("%d ", resultado[j][i]);
		}

		printf("\n");
	}

	return 0;
}
```
Documentación:
Al principio se establecen las variables de las dos matrices de 10 x 10 que se van a multiplicar y se inicializan con datos aleatorios. Se declara la variable donde el resultado de la multiplicación se va a almacenar y se declara una variable auxiliar para las sumas de las multiplicaciones necesarias por elemento de la matriz. Luego se empieza a hacer la multiplicación de matrices. Para ello se toma la fila de la primera matriz y la columna de la segunda matriz, se multiplican sus elementos y la suma de ellos se almacena en la matriz resultado. Una vez terminado todo el proceso de multiplicación, se imprime la matriz resultado para que el usuario pueda verlo.

Análisis de complejidad:
Inicialización de las matrices:
En esta parte del código podemos ver que hay dos bucles anidados, siendo la cantidad de operaciones de 10 x 10 ya que las matrices utilizadas en este ejercicio son de 10 x 10.
Al ser un valor constante, podemos determinar que su complejidad sería de O(1).
Multiplicación de matrices:
En esta parte del código podemos observar que hay tres bucles anidados, siendo la cantidad de operaciones de 10 x 10 x 10.
Esto nos daría una complejidad de O(1).
Imprimir matriz resultado:
En esta parte del código podemos observar como hay dos bucles anidados, siendo la cantidad de operaciones de 10 x 10 debido a que estas son las dimensiones de la matriz resultado.
Debido a que este valor es constante, podemos determinar que su complejidad sería de O(1).
Complejidad final:
La complejidad de este código se encuentra dominada por O(1). Por ello, la complejidad general de este código sería de O(1).
