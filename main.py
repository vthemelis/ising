import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.misc
import scipy.signal


def initialiser(N, dimensions = 2):
    """
    Create an N^dimensions array that will be used as the inital state of the 
    lattice. The array will have balues of -1 and 1 signifying spin-down 
    and spin-up states respectively.
    
    Parameters
    ----------
    N : int
        Length of each lattice dimension.
    dimensions : int (positive) , optional
        The number of dimensions of the array
        
    Returns
    -------
    numpy.ndarray :
        The initialised randomised d-dimensional array.
    
    
    """
    
    #shape for correct dimensions
    shape = tuple([N]) * dimensions
    
    #randomise spins
    lattice = np.random.choice([1,-1], size = shape)
    
    return lattice

def deltaE(lattice, h = 0, J = 1):
    """
    Find the energy needed to flip each spin accomodated in the variable lattice.
    
    Prameters
    ---------
    lattice: int array
        the lattice of which the spin energy will be calculated.
    h: float , optional
        external magnetic field     
    J: float , optional
        coupling constant (usualy 1 for a ferromagnetic material, -1 for 
        antiferromagnetic material.)
        
    Returns
    -------
    numpy.ndarray :
        Array of the shape of lattice with the calcualted hamiltonian for each 
        spin.
        
    
    """
    dimensions = len(lattice.shape)
    
    #1D energy
    energylat = np.roll(lattice,1, axis = 0) + np.roll(lattice,-1, axis = 0)
    
    if dimensions > 1:
        for ax in range (1,dimensions):
           energylat = energylat + np.roll(lattice, 1, axis = ax)\
           + np.roll(lattice, -1, axis = ax) 

    
    #energy after flip - energy before flip (see the Hamiltonian in Ising model)
    energylat = 2 * (J * np.multiply(energylat, lattice) + h*lattice )
    
    #return an array with the energies needed to flip corresponding spins.
    return energylat

def flipper(s, rate, p, whitetile):
    """
    Compares the transition rate to a random number p, in accordance with the 
    Metropolis algorithm,    to check whether a spin of value s should flip. 
    Whitetile is introduced as part of the routine I used to make the script 
    more efficient and avoiding having the ferromagnetic lattice fall into the 
    anti-ferromagnetic lattice. 
    (see simulation() and cheessboard())
    
    Parameters
    ----------
    s : int
        value of spin
    rate : float
        the Boltzman factor
    p : float
        a random number that belons to [0,1]
    whitetile: bool
        True if the current tile is 'white' according to the chessboard model.
        
    Returns
    -------
    int :
        The final value of the spin.
    """
    if rate > p and whitetile:
        #flip spin
        return -s
    else:
        #don't flip spin
        return s
    
"""
A vectorised version of flipper()
"""    
vec_flipper = np.vectorize(flipper)

def chessboard(N, dimensions, dtype = bool):
    """
    Use a chess board scheme to seggregate the spin population of a lattice 
    in a way in which no two spins that belong in the same group (out of two;
    chessboard whites/Trues and chessboard blacks/Falses)
    
    Parameters
    ----------
    N : int
        The length of each lattice dimension
    dimensions : int
        Number of dimensions in lattice
    dtype : type
        the type of the chessboard entries (normaly bool as described above)
        
    Returns:
        numpy.ndarray :
            the chessboard-like array
    """
    
    #make a 1D chessboard
    board = [(i % 2) == 0 for i in range(N)]
    
    #generalise to higher dimension
    for i in range(1, dimensions):
        board = [board if i % 2 == 0 else np.logical_not(np.asarray(board)) \
                          for i in range(N)]
    
    numpyboard = np.asarray(board, dtype = dtype)
    
    return numpyboard


def simulation(N , T, sweeps, lattice = None, dimensions = 2, h = 0,\
               nonabsmag = False, J = 1, prelims = 50):
    """
    Find what the lattice looks like at the next step by applying
    the Metropolis Monte-Carlo algorithm on the argument lattice.
    
    Parameters
    ----------
    N : int
        The length of each lattice dimension   
    T: float
        Temperature of simulation
    sweeps : int
        number of times that the algorithm will be applied on the lattice
        that will be returned
    lattice: numpy.ndarray , optional
        feed an initial lattice on which the algorithm will be aplied
        If None is applied then a fresh lattice will be initialised 
        internally
    dimensions: int , optional
        Run the simulation on dimensionD lattice
    h: float , optional
        Run the simulation with an external magnetic field.
    nonabsmag: bool , optional
        return magnetisation values without applying the np.abs() function 
        on the obtained values.
    J: float , optional
        Coupling constant
    prelims: int , optional
        number of preliminary (warm-up) steps
        
    Returns
    -------
    tuple:
        Tuple of:
        m: the magnetisation calculated as the average of spins in the lattice 
            after every sweep. (Normally the absolute value)
        e: the energy per spin calculated before every sweep.
        l: the final state of the lattice.
    """
    
    #check if a lattice is initially fed in the function or initilise a fresh one
    if lattice is None:
        lattice = initialiser(N, dimensions=dimensions)  
    
    #collect the magnetisations in the array:
    M = np.zeros(prelims + sweeps)
    
    #collect energies
    E = np.zeros(prelims + sweeps)
    
    #chess board
    board = chessboard(N, dimensions=dimensions)
    
    #calculate energies of the lattice
    energies = deltaE(lattice, h = h, J = J)
    
    if J < 0:        
        mult = chessboard(N, dimensions,dtype=int)
        mult = 2.0 * (mult - 0.5)
    
    #itterate the prelim sweeps and then the wanted sweeps
    for itt in range(prelims + sweeps):
        
        #generate random numbers
        shape = tuple([N]) * dimensions
        p = np.random.random_sample(shape)
        
        #calculate the energy of the itteration
        #normalise by the volume of the lattice (or area for the 2D case)
        vol = np.product(np.array(lattice.shape))
        
        #find the rates for unit kB
        rates = np.exp(-energies/T)
        
        #flip white spins that should be flipped
        lattice = vec_flipper(lattice, rates, p, board)
        
        #update the values of energies after white spins have flipped
        energies = deltaE(lattice, h = h, J = J)
        
        #calculate new rates
        rates = np.exp(-energies/T)    
        
        #flip black spins (NB that we take the NOT of the chessboard bool array)
        lattice = vec_flipper(lattice, rates, p, np.logical_not(board)) 

        #update values of energies befroe calculating C and E
        energies = deltaE(lattice, h = h, J = J)               
        
        #tabulate magnetisation value
        #if J>0:
        M[itt] = np.mean(lattice)
        #else:
        #    M[itt] = np.mean(np.multiply(mult, lattice))
        E[itt] = -np.sum(energies)/2 /vol
        
    #check if we need the absolute value of Magnetisation
    if not nonabsmag:
        M = np.abs(M)
    
    return (M[prelims:],E[prelims:], lattice)

def plotMEC(dimensions = 2, J = 1, filename = None,N = [20], \
            anneal = True, Tlim = [1,4], prelims = 50, sweeps = 200, \
            plots = True, plainlines = False, steps = 150):
    """
    Plot magnetisation and energy as a function of temperature.
    
    The Magnetisation values are obtained for the highest temperatures first
    and then obtained lattices are fed back to the simulator for lower 
    tempereature to help a smooth transition between temperatures and 
    avoiding getting stuck in metastable states while calculating 
    magnetisation for lower temperature states.
    
    Can also be used to obtain magnetisation, energy, or heat capacity as a
    function of temperature.
    
    Returns:
        T: the temperatures used
        Ms: The arrays of magnetisation plotted for corresponding N
    
    Parameters
    ----------
    dimensions: int , optional
        run the code for a dimensionsD lattice
    J: float, optional
        coupling constant
    filename: str, optional
        filename used for the saved plots.
    N: list , optional
        the lattice sizes for which results are plot.
    anneal : bool , optional
        True to use simulated annealing
    Tlim : list , optional
        conatains the temperature boundaries of the plot
    prelims : int , optional
        number of warm-up steps
    sweeps : int , optional
        number of main steps of simulation
    plots : bool , optional
        True to crate plots
    plainlines : bool , optional
        True to create plots that lack marks on points.
    steps : int , optional
        number of temperature points to be created.
        
    Returns
    -------
    tuple:
        Tuple of:
        T: temperatures array
        Ms: Magnetisations array
        Cs: Heat Capacity array
    """
    
    #temperature linespace
    T = np.linspace(Tlim[0],Tlim[1], steps)
    
    #tabulated magnetisation arry list
    Ms = []
    
    #tabulated energy array list
    Es = []
    
    #tabulated heat capacities
    Cs = []
    
    #labels used for datasets in the plots
    labels = []
    
    #critical exponent function used to fit data.
    def f (x, p1, p2, p3) : return p1*(((p2-x)/p2) ** p3)
    
    
    
    #itterate over wanted values of N
    for k in range(len(N)):
        
        #magnetisations and energies for N(i)
        M = np.zeros(T.shape)
        E = np.zeros(T.shape)
        C = np.zeros(T.shape)
        
        #lattice for N(i)
        lattice = initialiser(N[k],dimensions = dimensions)


        
        #itterate over all temperatures, highest first
        for i in range(len(T)):
            #highest first
            index = len(T) - i - 1
            
            #run simulation
            (Mi,Ei,l) = simulation(N[k],T[index],sweeps, lattice,\
            dimensions = dimensions, J = J, prelims = prelims)
                
            #tabulate obtained data
            M[index] = np.abs(np.mean(Mi))
            E[index] = np.mean(Ei)
            Ci = (np.std(Ei)/T[index] * N[k] /2)**2
            C[index] = np.mean(Ci)
            
            #change lattice that will be fed to the next simulation
            if anneal:
                lattice = l           
        
        #tabulate data for N(i)
        Ms.append(M)
        Es.append(E)
        Cs.append(C)
        
        labels.append("N = "+str(N[k]))
   
    if plots:
        orderpar = "Magnetisation" if J>0 else "Staggered Magnetisation"
        
        #plot data
        magfigure = makeplot(T,Ms, labels, "Temperature / $[J/k_B]$", orderpar,\
                             plainlines=plainlines)
        
        magfigure.show()
        
        enfigure = makeplot(T,Es, labels, "Temperature / $[J/k_B]$", "Energy per spin / $[J]$",\
                plainlines=plainlines)
        enfigure.show()
        
        cfigure = makeplot(T, Cs, labels, "Temperature / $[J/k_B]$", "Heat Capacity / $[k_B]$",\
                plainlines=plainlines)
        cfigure.show()
        
        #save plots
        if filename is not None:
            magfigure.savefig(filename+".svg")
            enfigure.savefig(filename+"E.svg")
            cfigure.savefig(filename+"C.svg")
        
    return(T, Ms, Cs)
    
def autocovariance(magnetisation):
    """
    Calculates the autocovariance for the magnetisation vector.
    
    Prameters
    ---------
    magnetisation : numpy.ndarray
        The magnetisation per timestep vector to be used.
        
    Returns
    -------
    numpy.ndarray :
        The created autocovariance vector with length = len(magnetisation)//20
    """
    
    #length of input vector
    length = len(magnetisation)
    
    #maximum degree of retardation tau
    taulength = length // 20
    
    #linespace
    tau = np.array(range(taulength))
    
    #average magnetisation
    averagemag = np.average(magnetisation)
    
    #late snap of the magnetisation vector
    lateM = magnetisation[taulength:]
    
    #M'(t+tau)
    lateMprime = lateM - averagemag
    
    #collect A values
    A = np.zeros(taulength)
    
    #<M'(t)M'(t)>
    A[0] = np.mean(np.square(lateMprime))
    
    for i in range(1,taulength):
        #M'(t)
        Mprime = magnetisation[taulength - tau[i]: -tau[i]] - averagemag
        
        #<M'(t+tau)M'(t)>
        A[i] = np.mean(np.multiply(Mprime, lateMprime))
    
    #obtain autocovariance
    a = A / A[0]
    
    #return autocovariance
    return a



def plotAutocov(filename = None, T = 1, sweeps = 10000):
    """
    Plot autocavariance at different values of N, for temperature T.
    
    Parameters
    ----------
    filename: str, optional
        filename used for the saved plots.
    T: float, optional
        Temperature of simulation
    sweeps : int , optional
        number of main steps of simulation    
    """
    
    #values of lattice size
    N = [10,40,70]
    
    #tabulate calculated a
    a = []
    
    #labels for plots
    labels = []
    
    #iterate of Ni's
    for i in range(len(N)):
        #anneal initial lattice from a high temperature.
        lattice = anneal(initialiser(N[i]), T, 200)
        
        #run simulation
        (m,e,l) = simulation(N[i], T, sweeps, lattice = lattice, prelims=1000)
        
        #calculate autocovariance
        ai = autocovariance(m)
        
        #tabulate calculated autocovariance vector
        a.append(ai)
        
        #add a label
        labels.append("N = " + str(N[i]))
        
        steps = np.array(range(len(a[0])))
        autoc = a[i]
        stepsles = steps[autoc < np.exp(-1)]
        print(str(N[i])+" "+ str(stepsles[0])+"\n")

    print(a)
    print(labels)
    #plot data
    f = makeplot(np.arange(0,len(a[0])),a, labels, "Time delay",\
                 "Autocorrelation", plainlines = True)
    
    

    
    #find e-fold points and save them in a text file along with plots if filenme is not empty
    if filename is not None:
        #save figure
        f.savefig(filename+".svg")
        
        #open file
        fil= open(filename+".txt","w+")
        steps = np.array(range(len(a[0])))
        
        #calculate e-fold points and save file.
        for i  in range(len(N)):
            autoc = a[i]
            stepsles = steps[autoc < np.exp(-1)]
            fil.write(str(N[i])+" "+ str(stepsles[0])+"\n")
        fil.close()

def Tc (x,infin, a, nu):
    """
    Finite size lattice scaling function.
    """
    return infin + a* (x ** (-1/nu))

def fitdata(X, Y, func, mask, p0 = None, filename = "fit", sigma = None,\
            xlabel = None, ylabel = None, scatter = False):
    """
    Fit given data using the curve_fit() routine from scipy.optimize.
    
    Parameters
    ----------
    X : np.ndarray
        X axis values
    Y : np.ndarray
        Y axis values
    func: function
        the function of which the parameters are to be calculated.
    mask : np.ndarray
        contains the minimum and maximum value of X that will be considered
        for the fitting of the function
    p0 : np.ndarray , optional
        initial guess of the parameters
    filename : string , optional
        name of output files
    sigma : np.ndarray , optional
        errors in the measurements of Y.
    xlabel : str , optional
        label of x-axis on the figure
    ylabel : str , optional
        label of y-axis on the figure
    scatter : bool , optional
        True if the points in the plot should be presented as in a scatterplot.    
    """
    
    #gather indices that are within the masking region   
    index = (X>mask[0]) & (X<mask[1])
    
    #get parameters of the fitting of the function using scipy
    (p, er) = optimize.curve_fit(func, X[index], Y[index], p0 = p0,\
    sigma = sigma)
    
    #print out the parameter values and the errors
    print("###################")
    print(p)
    print(np.sqrt(np.diag(er)))
    print("###################")
    
    #create scatter plot if required
    if scatter:
        f = plt.figure()
        ax1 = f.add_subplot(111)
        ax1.scatter(X, Y, linewidth = 1, label = "data", marker = 'x')
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
    else:
        f = makeplot(X, [Y], ["data"], xlabel, ylabel, plainlines = True)
    f.axes[0].plot(np.linspace(X[index][0],X[index][-1],1000), \
          func(np.linspace(X[index][0],X[index][-1],1000), p[0],p[1],p[2]), \
          'r-', label = "fit")
    f.axes[0].errorbar(X, Y, yerr = sigma, fmt = 'bx', elinewidth = 1,\
          ecolor = 'black', capsize = 2)
    f.axes[0].legend(loc = 'best')
    f.axes[0].grid()
    f.savefig(filename+".svg")

def anneal(lattice, temp, steps):
    """
    anneals a lattice to a low temperature slowly from a high temperature
    (T = 4), helping avoid getting stuck in metastable states.
    
    Parameters
    ----------
    lattice : np.ndarray
        the lattice that will get annealed.
    temp : float
        the final temperature of the lattice
    steps : int
        number of steps until reaching the final temperature.
        
    Returns
    -------
    np.ndarray :
        The lattice at temperature temp.   
    
    """
    #temperature linespace
    annealT = np.linspace(temp, 4, 20)
    
    for i in range(len(annealT)):
        index = len(annealT) - i - 1
        (m,e,l) = simulation(lattice.shape[0], annealT[index], 200, lattice)
        lattice = l
    
    return lattice

def hysteresis(T = 1, dimensions = 2, J = 1, filename = "hist", hmax = 2.5):
    """
    plots magnetisation vs external magnetic field strength.
    
    Parameters
    ----------
    T : float , optional
        The temperature of the simulation
    dimensions : int , optional 
        The number of dimensions of the lattice
    J : float , optional
        The coupling constant
    filename : str , optional
        The name of output files
    hmax : float , optional
        The maxumum absolute value of the external field.
    """
    h = np.linspace(-hmax, hmax, 100)
    
    #size of lattice
    N = 20
    
    #forward tabulated magnetisations and backward going
    Mforward = np.zeros(h.shape)
    Mbackward = np.zeros(h.shape)
    
    #initial lattice
    lattice = initialiser(N, dimensions = dimensions)
    
    #anneal lattice
    lattice = anneal(lattice, T, 20)

    #forward scan over different values of strength
    for i in range(len(h)):
        (m,e,l) = simulation(N, T, 200, lattice, h = h[i], nonabsmag=True,\
                dimensions= dimensions, J = J)
        Mforward[i] = np.mean(m)
        lattice = l
    
    #backward scan over different values of strength    
    for i in range(len(h)):
        index = len(h) - 1 - i
        (m,e,l) = simulation(N, T, 200, lattice, h = h[index], nonabsmag=True,\
                dimensions = dimensions, J = J)
        Mbackward[index] = np.mean(m)
        lattice = l
    
    #plot data
    f = makeplot(h, [Mforward, Mbackward], ["Increasing h", "Decreasing h"],\
                 "External field, h $[J]$", "Magnetisation")
    f.show()
    f.savefig(filename+".svg")

def makeplot(x, ys, labels, xlabel, ylabel, plainlines = False, figure = None,\
             filename = None, sigmas = None, logy = False, logx = False):
    """
    Template for creating a plot
    
    Parameters
    ----------
    x: np.ndarray
        x-values
    ys: np.ndarray
        list of y values
    labels: list
        labels for the y data to be plotted
    xlabel: str
        label of x axis
    ylabel: str
        label of y axis
    plainlines : bool , optional
        If true, the data is ploted without a marker
    figure : matplotlib.figure.Figure
        An initial canvas on which further plots can be drawn
    filename : str
        The name of the output files
    sigmas : list
        errorbars
    logy : bool
        scales y-axis as log if True
    logx : bool
        scales x-axis as log if True
    
    Returns
    -------
    matplotlib.figure.Figure :
        The figure on which the plot was drawn
    """
    
    #initialise a pyplot figure if needed
    if figure is None:
        f = plt.figure()
        #add axis
        a = f.add_subplot(111)
    else:
        a = f.axes[0]
    
    #styles for plotted data
    styles = ['rx-','yx-','gx-','mx-','rx-']
    formats = ['rx','yx','gx','mx','rx']
    
    #plain line styles
    if plainlines:
        styles = ['k-','r-','g-','y-','m-']
    
    #plot . . .
    for i in range(len(ys)):
        a.plot(x, ys[i], styles[i], label = labels[i])
    if sigmas is not None:
        for i in range(len(ys)):
            a.errorbar(x, ys[i],yerr = sigmas[i], fmt = formats[i], elinewidth = 1,\
                    ecolor = 'black', capsize = 2)    
    if logx:
        a.set_xscale('log')
    if logy:
        a.set_yscale('log')
    
    #set labels
    a.set_xlabel(xlabel)
    a.set_ylabel(ylabel)
    
    #add legend
    a.legend(loc = 'best')
    
    #save
    if filename is not None:
        f.savefig(filename+".svg")
    
    return f  
	
def trysavgol(window, order, data, xaxis):
    """
    Try the savitzky-golay filter with specific parameters on data.
    
    Parameters
    ----------
    window : int
        window of data (must be odd)
    order : int
        order of fitted polynomial
    data : list
        y axis of data to be filtered
    xaxis : list
        x axis of data to be filtered
    """
    for datum in data:
        filt = scipy.signal.savgol_filter(datum,window,order)
        f = makeplot(xaxis, [datum, filt],["data","filter"], "Temperature / $[J/k_B]$",\
                "Heat Capacity / $[k_B]$", plainlines = True)
        f.show()
        f.savefig("savgol.svg")
        Tc = np.mean(xaxis[filt == max(filt)])
        print("The predicted value of Tc is: "+str(Tc))
