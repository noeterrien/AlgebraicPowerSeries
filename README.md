This module introduces an algebraic representation of potentially infinite power series and arrays of power series of any number of variables. These series are represented by PowerSeries objects, which contain the instructions to compute the coefficients of the series up to an arbitrarily high order (though that may come at the cost of some algorithmic complexity).

It was primarily developped to compute backstepping kernels (see M. Krstic and A. Smyshlyaev, Boundary Control of PDEs: A Course on Backstepping
Designs, SIAM, Philadelphia, 2008 for details on this method), but it should technically work to solve a number of well-posed Partial Differential Equations. It was tested on multiple examples (see the Examples section of the documentation), correctly reproducing results from matematica but possible bugs may remain. Users are encouraged to do their own tests to verify that everything works as expected for their personal use. **NO SUPPORT IS INTENDED!**

Documentation can be found [here](docs.pdf)
