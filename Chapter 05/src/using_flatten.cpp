#include <iostream>
#include <random>

#include "book/definitions.hpp"
#include "fully_connected_layers.hpp"

int main(int, char**)
{
    srand((unsigned int) time(0));

	Tensor_3D input(10, 3, 3);
    input.setRandom();

    std::cout << "Intput dimensions are: " << input.dimensions() << "\n\n";

    auto output = flatten(input);

    std::cout << "Output dimensions are: " << output.dimensions() << "\n\n";

    return 0;

}