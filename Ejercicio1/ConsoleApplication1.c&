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
