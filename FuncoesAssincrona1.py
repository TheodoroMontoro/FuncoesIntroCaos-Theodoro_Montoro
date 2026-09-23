# Aqui foram definidas todas as funções utilizadas para a Atividade Assíncrona 1 do curso de Introdução ao Caos

# A maioria são cópias ou versões levemente modificadas das funções criadas em conjunto com o professor durante as aulas. As principais mudanças
# estão na parte dos Mapas, em que foi necessário adaptar as funções para o mapa abordado ao invés do Mapa Logistico.




# Imports gerais
import numpy as np
import matplotlib.pyplot as plt

# Meu nUSP como seed de aleatoriedade
np.random.seed(15489998) 


#                          Funções dos Exercícios 1 e 2 (Pontos Fixos, Espaços de Fase etc)


#Import da função solve_ivp, que sera usada para resolver os sistemas
from scipy.integrate import solve_ivp


#Função que obtem os pontos maximos de x como eventos, que é integrada na função faz_Figura. Criada pelo professor em sala.
def evMax( t, xy ): 
  # y é a derivada de x, então o máximo de x ocorre quando y=0, com derivada negativa
  x, y = xy
  return y
evMax.terminal = 2 # Para parar a integração quando ocorrer duas vezes o evento (ou no final do t_span)
evMax.direction = -1 # Para que a derivada da função evMax seja negativa


OPCOES = dict( dense_output=True, events = [evMax],\
               atol=1e-9, rtol=1e-6 )


#Função para plotar as figuras relacionadas aos sistemas dinamicos abordados. Criada pelo professor em sala.
def faz_Figura( SOL, MOSTRA_T=True, MOSTRA_tempo=True, MOSTRA_fase=True, \
                CRIA_fig = True, SHOW_fig = True, LIMITES_x=None ):  
  x0y0 = SOL.y[:,0]
  t = np.linspace(SOL.t[0], SOL.t[-1], 10_001)
  xy = SOL.sol(t)
  if MOSTRA_T:
    tMax = SOL.t_events[0]
    xyMax = SOL.sol(tMax)
    T = tMax[1]-tMax[0]
    print( f'{T = :f}, y0 = {x0y0[1]:f}' )
    print( f'{T}\t{x0y0[1]}'.replace('.',',') )
  if MOSTRA_tempo:
    if CRIA_fig:
      plt.subplots(nrows=2, ncols=1, sharex=True)
    plt.subplot(2,1,1)
    plt.plot( t, xy[0], '.-' )
    #plt.plot( tMax, xyMax[0], 'or' )
    plt.ylabel( 'x' )
    plt.subplot(2,1,2)
    plt.plot( t, xy[1], '.-' )
    #plt.plot( tMax, xyMax[1], 'or' )
    plt.ylabel( 'y' )
    plt.xlabel( 't' )
    if SHOW_fig:
      plt.show()

  if MOSTRA_fase:
    if CRIA_fig:
      plt.figure()
    if LIMITES_x is None:
      plt.plot( xy[0], xy[1], '.-' )
      plt.plot( xy[0][0], xy[1][0], 'og' )
    else:
      xNovo = (xy[0] - LIMITES_x[0]) % (LIMITES_x[-1] - LIMITES_x[0]) + \
              LIMITES_x[0]
      plt.plot( xNovo, xy[1], '.' )
      plt.plot( xNovo[0], xy[1][0], 'og' )
    plt.xlabel( 'x' )
    plt.ylabel( 'y' )
    if SHOW_fig:
      plt.show()

# Função que plota apenas o espaço de fase do sistema abordado. Criada pelo professor em sala.
def fazFase( SOL, ehUnico = False, **kwrd ): 
  faz_Figura( SOL, MOSTRA_T=False, MOSTRA_tempo=False, MOSTRA_fase=True, \
                CRIA_fig = ehUnico, SHOW_fig = ehUnico, **kwrd )



# Sistemas da questão 1.

# 1.a) 

def Sistema1a(t, xy):
  x,y = xy
  dxdt = -3*x
  dydt = 3*x - 2*y
  return [dxdt,dydt]

#1 b) 

def Sistema1b(t, xy):
  x,y = xy
  dxdt = -x + 4*y
  dydt = -2*x + 5*y
  return [dxdt,dydt]


#1 c) 

def Sistema1c(t, xy):
    x, y = xy
    dxdt = x*(3 - x - 2*y)
    dydt = y*(2 - x - y)
    return [dxdt, dydt]


# Sistemas da questão 2:

# Sistema (1)
def Sistema2_1(t, xy):
  x,y = xy
  dxdt = y
  dydt = -np.sin(x)
  return [dxdt,dydt]


# Sistema (2)
def Sistema2_2(t,xy):
  x,y = xy
  dxdt = y
  dydt = -x + (1/6)* x**3
  return[dxdt,dydt]






#                                       Funções dos exercícios 3 e 4 (parte de Mapas,Bifurcações, Lyapunov, etc)


# Função criada pelo professor para abordar o Mapa Logistico (presente no seu repositório do github), levemente modificada para abordar o Mapa em questão
def Mapa(x0, b, n=1, nT=0): 
  x = x0
  X = []
  for i in range(-nT,n):
    x = (1-(b*(x**2))) # Mapa de nUSP par, com parametro "b" ao inves de "a"
    if i >= 0:
      X.append(x)
  return np.asarray(X)



# Função criada pelo professor, levemente modificada para o novo Mapa.
def Lyapunov( X, b, fPrimeMinimo=1e-10 ):  
  L = np.zeros_like(b)
  for x in X:
    fPrime = np.maximum( np.abs(-2*b*x), fPrimeMinimo ) # Aqui, considerado a derivada do Mapa de nUSP par, ao invés da derivada do mapa logístico
    L += np.log(fPrime)
  return L/X.shape[0]




# Levemente alterado para se adequar ao novo mapa considerado. Já é a versão atualizada que leva o ContadorPeriodo em consideração
def FazFiguraDiagramaBifurcacao( Bs, X, L, \
              figsize = None, alpha = 0.2, ylimL = None, \
              corL = 'b', corX = 'k', \
              SHOW = True, DEVOLVE_fig_axs = False, \
              fig = None, axs=None ): 
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

# Mesma função contadora de período feita pelo professor para as atividades em sala
def contadorPeriodo( X, delta=1e-5, pMax=np.inf, nVerMin=0 ): 
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



 # Função para o diagrama de CobWeb feita pelo professor em sala, também levemente alterada para se adequar ao novo Mapa
def CobWeb(x0, N, b ):
  def f(x):
    return (1-b*x**2) # Mapa de nUSP par
  plt.figure()
  xs = np.linspace(-1,1,2_001)
  plt.plot( xs, f(xs).T, '-m' )
  plt.plot( xs, xs, '-k' )
  plt.xlabel( 'x' )
  plt.ylabel( 'f(x)' )
  plt.xlim([-1.2,1.2])
  plt.ylim([-1.2,1.2]) 
  plt.grid()

  # 1o passo (subir na vertical):
  plt.plot( [x0,x0], [0, f(x0)], '-r' )
  xA = x0
  for _ in range(N):
    xN = f(xA)
    # linha horizontal
    plt.plot( [xA,xN], [xN, xN], '-g' )
    # linha vertical
    plt.plot( [xN, xN], [xN, f(xN)], '-b' )
    xA = xN
  plt.show()


#  Não foram definidas aqui as funções relacionadas as figuras interativas dos exercicios 3 e 4. Elas foram definidas diretamente no ipynb.
