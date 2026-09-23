
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(15489998) # Meu nUSP como seed de aleatoriedade

def Mapa(x0, b, n=1, nT=0): # Funções do diretório do professor, levemente modificadas para se adequarem ao mapa utilizado nos exercícios 3 e 4
  x = x0
  X = []
  for i in range(-nT,n):
    x = (1-(b*(x**2))) # Mapa de nUSP par
    if i >= 0:
      X.append(x)
  return np.asarray(X)
#

def Lyapunov( X, b, fPrimeMinimo=1e-10 ): # Levemente alterado para se adequar ao novo mapa considerado
  L = np.zeros_like(b)
  for x in X:
    fPrime = np.maximum( np.abs(2*b*x), fPrimeMinimo )
    L += np.log(fPrime)
  return L/X.shape[0]
#

def FazFiguraDiagramaBifurcacao( Bs, X, L, \
              figsize = None, alpha = 0.2, ylimL = None, \
              corL = 'b', corX = 'k', \
              SHOW = True, DEVOLVE_fig_axs = False, \
              fig = None, axs=None ): # Levemente alterado para se adequar ao novo mapa considerado. Já é a versão atualizada que leva o ContadorPeriodo em consideração
  if fig is None:
    if figsize is not None:
      fig = plt.figure(figsize=figsize)
    else:
      fig = plt.figure()

  if axs is None:
    axs = []
    axs.append( plt.subplot(3,1,(1,2)) )
    axs.append( plt.subplot(3,1,3) )
    plt.subplots_adjust(hspace=0.5)
    axs[1].sharex(axs[0])
    axs[1].set_xlim([min(Bs), max(Bs)])
    axs[1].axhline( 0, c='r', lw=0.5 )

  # separar entre periódico e caótico
  #axs[0].plot( As, X.T, ',k', alpha=alpha )
  Ps = contadorPeriodo( X ); 
  ondePeriodico = np.flatnonzero( Ps>0 )
  ondeCaotico = np.flatnonzero( Ps==0 )
  axs[0].plot( Bs[ondePeriodico], X[:,ondePeriodico].T, ',', c=corX, alpha=1 )
  axs[0].plot( Bs[ondeCaotico], X[:,ondeCaotico].T, ',', c=corX, alpha=alpha )

  axs[0].axhline (0, c = 'c', linestyle = '-', alpha = 1)              
  axs[0].set_xlabel( 'b' )
  axs[0].set_ylabel( 'x' )
  axs[1].plot( Bs, L, ',', c=corL )
  axs[1].set_xlabel( 'b' )
  axs[1].set_ylabel( 'L' )
  if ylimL is not None:
    axs[1].set_ylim(ylimL)
  if SHOW:
    plt.show()
  if DEVOLVE_fig_axs:
    return fig, axs
#


def contadorPeriodo( X, delta=1e-5, pMax=np.inf, nVerMin=0 ): # Mesmo contador de período feito para as atividades em sala
  if X.ndim==1:
    X = np.atleast_2d(X).T
  Ps = []
  for x in X.T:
    repetidos = np.nonzero( abs(x-x[0])<delta )[0]
    p = 0
    if len(repetidos)>1: # o x contem o x[0]
      pi = np.diff(repetidos)
      if ( max(pi)==min(pi) ) and ( pi[0]<=np.min((pMax,(len(x)-nVerMin))) ):
        p = pi[0]
        for i in range(p):
          if max( abs(x[i::p]-x[i]) )>delta:
            p = 0
            break
    Ps.append( p )
  return np.asarray(Ps)
#
