from simulations import simulations
from markovDecision import markovDecision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax


Square_nb=np.linspace(1,14,14)

layout1=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]


def plotfun_theoretical_vs_empirical(layout,boolean,number_of_sim,strategy):

    Square_nb=np.linspace(1,14,14)
    layout1=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    emp = np.zeros(len(number_of_sim))
    theor = markovDecision(layout,boolean)[0]
    for i in range(len(number_of_sim)):
        emp[i] = simulations(layout,boolean,number_of_sim[i],strategy)
    
    
    fig=plt.figure()
    plt.plot(Square_nb,emp[0],'#e41a1c',linewidth=0.7, marker='o')
    plt.plot(Square_nb,emp[2],'#377eb8',linewidth=0.7, marker='o')
    plt.plot(Square_nb,emp[4],'#4daf4a',linewidth=0.7, marker='o')    
    plt.plot(Square_nb,theor,'#984ea3',linewidth=0.7, marker='o')
    plt.xlabel('Square Number',fontsize=16)
    plt.xticks(Square_nb,fontsize=14)
    plt.ylabel('Expected cost',fontsize=16)
    plt.yticks(fontsize=14)
    plt.legend(('100 Simulations','10000 Simulations','1000000 Simulations','Theoretical'),fontsize=16)
    plt.show


def plotfun_strategies_comparison(layout,boolean,num_of_sim,strategies):

    fig=plt.figure()
    emp = np.zeros(len(strategies))
    for i in range(len(strategies)):
        emp[i] = simulations(layout,boolean,num_of_sim,strategies[i])

    
    plt.plot(Square_nb,emp[0],'#e41a1c',linewidth=0.7, marker='o')
    plt.plot(Square_nb,emp[1],'#377eb8',linewidth=0.7, marker='o')
    plt.plot(Square_nb,emp[2],'#4daf4a',linewidth=0.7, marker='o')
    plt.plot(Square_nb,emp[3],'#984ea3',linewidth=0.7, marker='o')
    plt.xlabel('Square Number',fontsize=16)
    plt.xticks(Square_nb, fontsize=14)
    plt.ylabel('Expected cost',fontsize=16)
    plt.yticks(fontsize=14)
    plt.legend((strategies[0],strategies[1],strategies[2],strategies[3]),fontsize=16)
    plt.show


def plotfun_relative_error(layout,boolean,number_of_sim,strategy):

    Square_nb=np.linspace(1,14,14)
    layout1=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    emp = np.zeros(len(number_of_sim))
    theor = markovDecision(layout,boolean)[0]
    for i in range(len(number_of_sim)):
        emp[i] = simulations(layout,boolean,number_of_sim[i],strategy)

    fig=plt.figure()
    plt.plot(Square_nb,np.divide(np.subtract(theor,emp[0]),theor),'#e41a1c',linewidth=0.7, marker='o')
    plt.plot(Square_nb,np.divide(np.subtract(theor,emp[1]),theor),'#377eb8',linewidth=0.7, marker='o')
    plt.plot(Square_nb,np.divide(np.subtract(theor,emp[2]),theor),'#4daf4a',linewidth=0.7, marker='o')
    plt.plot(Square_nb,np.divide(np.subtract(theor,emp[3]),theor),'#984ea3',linewidth=0.7, marker='o')
    plt.plot(Square_nb,np.divide(np.subtract(theor,emp[4]),theor),'#ff7f00',linewidth=0.7, marker='o')
    plt.ylim(-0.3,0.3)
    plt.xlabel('Square Number',fontsize=16)
    plt.xticks(Square_nb, fontsize=14)
    plt.ylabel('Relative error',fontsize=16)
    plt.yticks(fontsize=14)
    plt.legend((strategy[0],strategy[1],strategy[2],strategy[3],strategy[4]),fontsize=16)
    plt.show




##layout 1 - circle - Theoretical vs Empirical #####
theor=markovDecision(layout1,True)[0]
emp1=simulations(layout1,True,100,'optimal')
emp2=simulations(layout1,True,1000,'optimal')
emp3=simulations(layout1,True,10000,'optimal')
emp4=simulations(layout1,True,100000,'optimal')
#emp5=simulations(layout1,True,1000000,'optimal')


fig=plt.figure()
plt.plot(Square_nb,emp3,'#e41a1c',linewidth=0.7, marker='o')
plt.plot(Square_nb,emp4,'#377eb8',linewidth=0.7, marker='o')
#plt.plot(Square_nb,emp5,'#4daf4a',linewidth=0.7, marker='o')
plt.plot(Square_nb,theor,'#984ea3',linewidth=0.7, marker='o')
plt.xlabel('Square Number',fontsize=16)
plt.xticks(Square_nb,fontsize=14)
plt.ylabel('Expected cost',fontsize=16)
plt.yticks(fontsize=14)
plt.legend(('100 Simulations','10000 Simulations','1000000 Simulations','Theoretical'),fontsize=16)
plt.show()
#plt.savefig('Figures/theorVSemp_circle.eps', format='eps')


##### layout 1 - circle - Relative error #####
fig=plt.figure()
plt.plot(Square_nb,np.divide(np.subtract(theor,emp1),theor),'#e41a1c',linewidth=0.7, marker='o')
plt.plot(Square_nb,np.divide(np.subtract(theor,emp2),theor),'#377eb8',linewidth=0.7, marker='o')
plt.plot(Square_nb,np.divide(np.subtract(theor,emp3),theor),'#4daf4a',linewidth=0.7, marker='o')
#plt.plot(Square_nb,np.divide(np.subtract(theor,emp4),theor),'#984ea3',linewidth=0.7, marker='o')
#plt.plot(Square_nb,np.divide(np.subtract(theor,emp5),theor),'#ff7f00',linewidth=0.7, marker='o')
plt.ylim(-0.3,0.3)
plt.xlabel('Square Number',fontsize=16)
plt.xticks(Square_nb, fontsize=14)
plt.ylabel('Relative error',fontsize=16)
plt.yticks(fontsize=14)
plt.legend(('100 Simulations','1000 Simulations','10000 Simulations','100000 Simulations','1000000 Simulations'),fontsize=16)
plt.show()
#plt.savefig('Figures/relative_error_circle.eps', format='eps')

##### layout 1 - circle - Strategies comparison #####
emp6=simulations(layout1,True,10000,'optimal')
emp7=simulations(layout1,True,10000,'security_only')
emp8=simulations(layout1,True,10000,'normal_only')
emp9=simulations(layout1,True,10000,'risky_only')


fig=plt.figure()
plt.plot(Square_nb,emp6,'#e41a1c',linewidth=0.7, marker='o')
plt.plot(Square_nb,emp7,'#377eb8',linewidth=0.7, marker='o')
plt.plot(Square_nb,emp8,'#4daf4a',linewidth=0.7, marker='o')
plt.plot(Square_nb,emp9,'#984ea3',linewidth=0.7, marker='o')
plt.xlabel('Square Number',fontsize=16)
plt.xticks(Square_nb, fontsize=14)
plt.ylabel('Expected cost',fontsize=16)
plt.yticks(fontsize=14)
plt.legend(('Optimal','Security Die','Normal Die','Risky Die only'),fontsize=16)
plt.show()
#plt.savefig('Figures/strategies_circle.eps', format='eps')


##### layout 1 - not circle - Theoretical vs Empirical #####
theor=markovDecision(layout1,False)[0]
emp1=simulations(layout1,False,100,'optimal')
emp2=simulations(layout1,False,1000,'optimal')
emp3=simulations(layout1,False,10000,'optimal')
#emp4=simulations(layout1,False,100000,'optimal')
#emp5=simulations(layout1,False,1000000,'optimal')


fig=plt.figure()
plt.plot(Square_nb,emp1,'#e41a1c',linewidth=0.7, marker='o')
plt.plot(Square_nb,emp3,'#377eb8',linewidth=0.7, marker='o')
#plt.plot(Square_nb,emp5,'#4daf4a',linewidth=0.7, marker='o')
plt.plot(Square_nb,theor,'#984ea3',linewidth=0.7, marker='o')
plt.xlabel('Square Number',fontsize=16)
plt.xticks(Square_nb,fontsize=14)
plt.ylabel('Expected cost',fontsize=16)
plt.yticks(fontsize=14)
plt.legend(('100 Simulations','10000 Simulations','1000000 Simulations','Theoretical'),fontsize=16)
plt.show()
#plt.savefig('Figures/theorVSemp.eps', format='eps')


##### layout 1 - not circle - Relative error #####
fig=plt.figure()
plt.plot(Square_nb,np.divide(np.subtract(theor,emp1),theor),'#e41a1c',linewidth=0.7, marker='o')
plt.plot(Square_nb,np.divide(np.subtract(theor,emp2),theor),'#377eb8',linewidth=0.7, marker='o')
plt.plot(Square_nb,np.divide(np.subtract(theor,emp3),theor),'#4daf4a',linewidth=0.7, marker='o')
#plt.plot(Square_nb,np.divide(np.subtract(theor,emp4),theor),'#984ea3',linewidth=0.7, marker='o')
#plt.plot(Square_nb,np.divide(np.subtract(theor,emp5),theor),'#ff7f00',linewidth=0.7, marker='o')
plt.ylim(-0.3,0.3)
plt.xlabel('Square Number',fontsize=16)
plt.xticks(Square_nb, fontsize=14)
plt.ylabel('Relative error',fontsize=16)
plt.yticks(fontsize=14)
plt.legend(('100 Simulations','1000 Simulations','10000 Simulations','100000 Simulations','1000000 Simulations'),fontsize=16)
plt.show()
#plt.savefig('Figures/relative_error.eps', format='eps')

##### layout 1 - not circle - Strategies comparison #####
emp6=simulations(layout1,False,10000,'optimal')
emp7=simulations(layout1,False,10000,'security_only')
emp8=simulations(layout1,False,10000,'normal_only')
emp9=simulations(layout1,False,10000,'risky_only')


fig=plt.figure()
plt.plot(Square_nb,emp6,'#e41a1c',linewidth=0.7, marker='o')
plt.plot(Square_nb,emp7,'#377eb8',linewidth=0.7, marker='o')
plt.plot(Square_nb,emp8,'#4daf4a',linewidth=0.7, marker='o')
plt.plot(Square_nb,emp9,'#984ea3',linewidth=0.7, marker='o')
plt.xlabel('Square Number',fontsize=16)
plt.xticks(Square_nb, fontsize=14)
plt.ylabel('Expected cost',fontsize=16)
plt.yticks(fontsize=14)
plt.legend(('Optimal','Security Die only','Normal Die only','Riksy Die only'),fontsize=16)
plt.show()
#plt.savefig('Figures/strategies.eps', format='eps')