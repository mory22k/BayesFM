# BayesFM using NumPy
My implementation of Bayesian Factorization Machines (BFM), as well as alternate least squares (ALS), using NumPy.

## How to use
1. Install Python 3.12 via pyenv. See https://github.com/pyenv/pyenv for installation instructions.
2. Install poetry. See https://python-poetry.org/docs/#installation for installation instructions.
3. Run `poetry shell` to activate the virtual environment.
4. Run `poetry install` to install dependencies.
5. Run `poetry run pytest` to run the test script. The test script will train the FM and BFM models using randomly generated data and compare the results in terms of RMSE. The result figure will be saved in the `test/out` directory.

## References
1. S. Rendle, Factorization Machines, in 2010 IEEE International Conference on Data Mining (2010), pp. 995–1000.
2. S. Rendle, Z. Gantner, C. Freudenthaler, and L. Schmidt-Thieme, Fast Context-Aware Recommendations with Factorization Machines, in Proceedings of the 34th International ACM SIGIR Conference on Research and Development in Information Retrieval (2011), pp. 635–644.
3. S. Rendle, Factorization Machines with libFM, ACM Trans. Intell. Syst. Technol. 3, 1 (2012).

## License
This project is licensed under the MIT license.
