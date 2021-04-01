import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from simulations import simulations
from markovDecision import markovDecision

Square_nb=np.linspace(1,14,14)

layout1=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]


##layout 1 - circle - Theoretical vs Empirical #####
theor=markovDecision(layout1,True)[0]
emp1=simulations(layout1,True,100,'Markov Decision')
emp2=simulations(layout1,True,1000,'Markov Decision')
emp3=simulations(layout1,True,10000,'Markov Decision')
emp4=simulations(layout1,True,100000,'Markov Decision')
emp5=simulations(layout1,True,1000000,'Markov Decision')


fig=plt.figure()
plt.plot(Square_nb,emp1,'#e41a1c',linewidth=0.7, marker='o')
plt.plot(Square_nb,emp3,'#377eb8',linewidth=0.7, marker='o')
plt.plot(Square_nb,emp5,'#4daf4a',linewidth=0.7, marker='o')
plt.plot(Square_nb,theor,'#984ea3',linewidth=0.7, marker='o')
plt.xlabel('Square Number',fontsize=16)
plt.xticks(Square_nb,fontsize=14)
plt.ylabel('Expected cost',fontsize=16)
plt.yticks(fontsize=14)
plt.legend(('100 Simulations','10000 Simulations','1000000 Simulations','Theoretical'),fontsize=16)
plt.savefig('Figures/theorVSemp_circle.eps', format='eps')


##### layout 1 - circle - Relative error #####
fig=plt.figure()
plt.plot(Square_nb,np.divide(np.subtract(theor,emp1),theor),'#e41a1c',linewidth=0.7, marker='o')
plt.plot(Square_nb,np.divide(np.subtract(theor,emp2),theor),'#377eb8',linewidth=0.7, marker='o')
plt.plot(Square_nb,np.divide(np.subtract(theor,emp3),theor),'#4daf4a',linewidth=0.7, marker='o')
plt.plot(Square_nb,np.divide(np.subtract(theor,emp4),theor),'#984ea3',linewidth=0.7, marker='o')
plt.plot(Square_nb,np.divide(np.subtract(theor,emp5),theor),'#ff7f00',linewidth=0.7, marker='o')
plt.ylim(-0.3,0.3)
plt.xlabel('Square Number',fontsize=16)
plt.xticks(Square_nb, fontsize=14)
plt.ylabel('Relative error',fontsize=16)
plt.yticks(fontsize=14)
plt.legend(('100 Simulations','1000 Simulations','10000 Simulations','100000 Simulations','1000000 Simulations'),fontsize=16)
plt.savefig('Figures/relative_error_circle.eps', format='eps')

##### layout 1 - circle - Strategies comparison #####
emp6=simulations(layout1,True,10000,'Markov Decision')
emp7=simulations(layout1,True,10000,'Altered Optimality')
emp8=simulations(layout1,True,10000,'Die1')
emp9=simulations(layout1,True,10000,'Die2')
emp10=simulations(layout1,True,10000,'Random')
emp11=simulations(layout1,True,10000,'Regular Player')
emp12=simulations(layout1,True,10000,'Offensive')
emp13=simulations(layout1,True,10000,'Defensive')

fig=plt.figure()
plt.plot(Square_nb,emp6,'#e41a1c',linewidth=0.7, marker='o')
plt.plot(Square_nb,emp7,'#377eb8',linewidth=0.7, marker='o')
plt.plot(Square_nb,emp8,'#4daf4a',linewidth=0.7, marker='o')
plt.plot(Square_nb,emp9,'#984ea3',linewidth=0.7, marker='o')
plt.plot(Square_nb,emp10,'#ff7f00',linewidth=0.7, marker='o')
plt.plot(Square_nb,emp11,'#ffff33',linewidth=0.7, marker='o')
plt.plot(Square_nb,emp12,'#a65628',linewidth=0.7, marker='o')
plt.plot(Square_nb,emp13,'#f781bf',linewidth=0.7, marker='o')
plt.xlabel('Square Number',fontsize=16)
plt.xticks(Square_nb, fontsize=14)
plt.ylabel('Expected cost',fontsize=16)
plt.yticks(fontsize=14)
plt.legend(('Optimal','Quasi-optimal','Security Die only','Normal Die only','Random','Intuitive Player','Offensive Player', 'Defensive Player'),fontsize=16)
plt.savefig('Figures/strategies_circle.eps', format='eps')


##### layout 1 - not circle - Theoretical vs Empirical #####
theor=markovDecision(layout1,False)[0]
emp1=simulations(layout1,False,'Markov Decision',100)
emp2=simulations(layout1,False,'Markov Decision',1000)
emp3=simulations(layout1,False,'Markov Decision',10000)
emp4=simulations(layout1,False,'Markov Decision',100000)
emp5=simulations(layout1,False,'Markov Decision',1000000)


fig=plt.figure()
plt.plot(Square_nb,emp1,'#e41a1c',linewidth=0.7, marker='o')
plt.plot(Square_nb,emp3,'#377eb8',linewidth=0.7, marker='o')
plt.plot(Square_nb,emp5,'#4daf4a',linewidth=0.7, marker='o')
plt.plot(Square_nb,theor,'#984ea3',linewidth=0.7, marker='o')
plt.xlabel('Square Number',fontsize=16)
plt.xticks(Square_nb,fontsize=14)
plt.ylabel('Expected cost',fontsize=16)
plt.yticks(fontsize=14)
plt.legend(('100 Simulations','10000 Simulations','1000000 Simulations','Theoretical'),fontsize=16)
plt.savefig('Figures/theorVSemp.eps', format='eps')


##### layout 1 - not circle - Relative error #####
fig=plt.figure()
plt.plot(Square_nb,np.divide(np.subtract(theor,emp1),theor),'#e41a1c',linewidth=0.7, marker='o')
plt.plot(Square_nb,np.divide(np.subtract(theor,emp2),theor),'#377eb8',linewidth=0.7, marker='o')
plt.plot(Square_nb,np.divide(np.subtract(theor,emp3),theor),'#4daf4a',linewidth=0.7, marker='o')
plt.plot(Square_nb,np.divide(np.subtract(theor,emp4),theor),'#984ea3',linewidth=0.7, marker='o')
plt.plot(Square_nb,np.divide(np.subtract(theor,emp5),theor),'#ff7f00',linewidth=0.7, marker='o')
plt.ylim(-0.3,0.3)
plt.xlabel('Square Number',fontsize=16)
plt.xticks(Square_nb, fontsize=14)
plt.ylabel('Relative error',fontsize=16)
plt.yticks(fontsize=14)
plt.legend(('100 Simulations','1000 Simulations','10000 Simulations','100000 Simulations','1000000 Simulations'),fontsize=16)
plt.savefig('Figures/relative_error.eps', format='eps')

##### layout 1 - not circle - Strategies comparison #####
emp6=simulations(layout1,False,'Markov Decision',10000)
emp7=simulations(layout1,False,'Altered Optimality',10000)
emp8=simulations(layout1,False,'Die1',10000)
emp9=simulations(layout1,False,'Die2',10000)
emp10=simulations(layout1,False,'Random',10000)
emp11=simulations(layout1,False,'Regular Player',10000)
emp12=simulations(layout1,False,'Offensive',10000)
emp13=simulations(layout1,False,'Defensive',10000)

fig=plt.figure()
plt.plot(Square_nb,emp6,'#e41a1c',linewidth=0.7, marker='o')
plt.plot(Square_nb,emp7,'#377eb8',linewidth=0.7, marker='o')
plt.plot(Square_nb,emp8,'#4daf4a',linewidth=0.7, marker='o')
plt.plot(Square_nb,emp9,'#984ea3',linewidth=0.7, marker='o')
plt.plot(Square_nb,emp10,'#ff7f00',linewidth=0.7, marker='o')
plt.plot(Square_nb,emp11,'#ffff33',linewidth=0.7, marker='o')
plt.plot(Square_nb,emp12,'#a65628',linewidth=0.7, marker='o')
plt.plot(Square_nb,emp13,'#f781bf',linewidth=0.7, marker='o')
plt.xlabel('Square Number',fontsize=16)
plt.xticks(Square_nb, fontsize=14)
plt.ylabel('Expected cost',fontsize=16)
plt.yticks(fontsize=14)
plt.legend(('Optimal','Quasi-optimal','Security Die only','Normal Die only','Random','Intuitive Player','Offensive Player', 'Defensive Player'),fontsize=16)
plt.savefig('Figures/strategies.eps', format='eps')