# L-BFGS metoda optimizacije

## Opis projekta

U okviru ovog projekta implementirana je L-BFGS (Limited-memory BFGS) metoda optimizacije. L-BFGS pripada kvazi-Njutnovim metodama optimizacije drugog reda, i kao što joj ime kaže, predstavlja modifikaciju BFGS (Broyden–Fletcher–Goldfarb–Shanno) metode optimizicije koja koristi manje memorije. To se postiže uklanjanjem potrebe za skladištenjem celokupne aproksimacije inverza hesijana, umesto koje se čuva određeni broj poslednjih razlika rešenja i razlika gradijenata iz prethodnih koraka na osnovu kojih se efikasnim postupkom aproksimira proizvod inverza hesijana i gradijenta ciljne funkcije. Pored objašnjenja i implementacije L-BFGS metode, ovaj projekat bavi se i upoređivanjem performansi ove implementacije sa L-BFGS implementacijom iz biblioteke *scipy*, kao i klasičnom Njutnovom metodom iz iste biblioteke. Testiranje performansi vrši se na nekoliko problema iz [CUTEst](https://github.com/ralna/CUTEst) skupa problema za testiranje optimizacionog softvera.

Centralni deo ovog projekta čini Jupyter sveska [L-BFGS.ipynb](https://github.com/qkolj/L-BFGS/blob/master/L-BFGS.ipynb) koja sadrži tri dela: u prvom delu predstavlja se ideja L-BFGS metode; u drugom delu prolazi se kroz implementaciju ove metode optimizacije, i vrši se demonstracija rada na jednom jednostvnom primeru; poslednji deo projekta sadrži pregled rezultata testiranja nad problemima iz CUTEst skupa problema. Za pristup CUTEst problemima korišćena je Python biblioteka *pycutest*. Pošto ova biblioteka ima relativno kompleksan proces instalacije, veoma je teško instalirati je u okviru *Anaconda* okruženja. Iz tog razloga, samo testiranje nad CUTEst problemima obavlja se u okviru [cutest_tests.py](https://github.com/qkolj/L-BFGS/blob/master/cutest_tests.py), dok poslednji deo Jupyter sveske prikazuje i komentariše dobijene rezultate. Testovi se mogu ponoviti na računaru sa pravilno instaliranom *pycutest* bibliotekom ([uputstvo za instalaciju](https://jfowkes.github.io/pycutest/_build/html/install.html)) jednostavnim pokretanjem pomenutog Python skripta.

## Potrebne bilioteke

- numpy
- scipy
- matplotlib
- pycutest ([uputstvo za instalaciju](https://jfowkes.github.io/pycutest/_build/html/install.html))

## Literatura

1. Nocedal J., & Wright S.J. (2006). *Numerical Optimization* (2nd ed)
2. Nikolić M., & Zečević A. (2019). Naučno izračunavanje
3. PyCUTEst documentation. online at: https://jfowkes.github.io/pycutest/
