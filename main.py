from deap import base, algorithms
from deap import creator
from deap import tools
import graycode

import algelitism

from prettytable import PrettyTable

import random
import matplotlib.pyplot as plt
import numpy as np
import time
import math

LOW, UP = -10, 10
ETA = 20
LENGTH_CHROM = 2    # длина хромосомы, подлежащей оптимизации

# константы генетического алгоритма
POPULATION_SIZE = 25   # количество индивидуумов в популяции
P_CROSSOVER = 0.2       # вероятность скрещивания
P_MUTATION_MAX = 0.8        # вероятность мутации индивидуума
P_MUTATION_MIN = 0.2        # вероятность мутации индивидуума
MAX_GENERATIONS = 50  # максимальное количество поколений
HALL_OF_FAME_SIZE = 25

def func(individial):
    return ((individial[0]+2*individial[1]-7)**2+(2*individial[0]+individial[1]-5)**2),
    #return (math.sin((individial[0]**2 + individial[1]**2)**(0.5))**2)/((individial[0]**2 + individial[1]**2)**0.5),

def funcBin(individial):
    point_1 = int(individial[0])
    point_2 = int(individial[1])
    return((point_1+2*point_2-7)**2+(2*point_1+point_2-5)**2),
    #return (math.sin((point_1**2 + point_2**2)**(0.5))**2)/((point_1**2 + point_2**2))+100,


def metrics(minvalues):
    count = 0
    avg = 0
    for i in minvalues:
        if i < 0.0001:
            count += 1
    return [count/len(minvalues), len(minvalues) - count]

    

def show(ax, xgrid, ygrid, f):

    ax.clear()
    ax.contour(xgrid, ygrid, f)
    ax.scatter(*zip(*population), color='black', s=3, zorder=1)

    plt.draw()
    plt.gcf().canvas.flush_events()

    time.sleep(0.01)
    
def showBin(ax, xgrid, ygrid, f):

    ax.clear()
    ax.contour(xgrid, ygrid, f)
    populationTmp = population
    print(len(populationTmp))
    
    for i in range(len(populationTmp)):
        populationTmp[i][0] = int(populationTmp[i][0])
        populationTmp[i][1] = int(populationTmp[i][1])
    ax.scatter(*zip(*populationTmp), color='black', s=3, zorder=1)

    plt.draw()
    plt.gcf().canvas.flush_events()

    time.sleep(0.01)

def randomPoint(a, b):
    return [random.uniform(a, b), random.uniform(a, b)]

def randomPoint_bin(a, b):
    return [random.randint(a, b), random.randint(a, b)]

if __name__ == '__main__':

    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    RANDOM_SEED = 1245
    random.seed(RANDOM_SEED)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    creator.create("Individual", list, fitness=creator.FitnessMin)
    best_num = []
    best_bin = []
    best_grey = []
    
    table = PrettyTable()
    
    table.field_names = ["Способ отбора родителей/Метод рекомбинации", "Тип рекомбинации", "Значение вероятности", "Тип числа", "Результат точка 1", "Результат точка 2", "Функция","Надежность", "Ср. число итераций"]

    for i in range(3): ## Тип числа Вещ/Бин/Грея
        for j in range(3): ## Cелекция
            for k in range(3): ## Кросс
                for m in range(2): ## Мутация
                    toolbox = base.Toolbox()
                
                    if i == 0:
                        toolbox.register("randomPoint", randomPoint, LOW, UP)
                        toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomPoint)
                        toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
                    if i == 1:
                        toolbox.register("randomPoint", randomPoint_bin, LOW, UP)
                        toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomPoint)
                        toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
                    if i == 2:
                        toolbox.register("randomPoint", randomPoint_bin, LOW, UP)
                        toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomPoint)
                        toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
                    population = toolbox.populationCreator(n=POPULATION_SIZE)

                    if i == 0:
                        toolbox.register("evaluate", func)
                    else:
                        toolbox.register("evaluate", funcBin)
                    
                    if j == 0:
                        toolbox.register("select", tools.selBest)
                    if j == 1:
                        toolbox.register("select", tools.selRoulette)
                    if j == 2:
                        toolbox.register("select", tools.selTournament, tournsize=3)
                        
                    if k == 0:
                        toolbox.register("mate", tools.cxOnePoint)
                    if k == 1:
                        toolbox.register("mate", tools.cxTwoPoint)
                    if k == 2:
                        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=LOW, up=UP, eta=ETA)
                    
                    if m == 0:        
                        toolbox.register("mutate", tools.mutPolynomialBounded, low=LOW, up=UP, eta=ETA, indpb=1.0/LENGTH_CHROM)
                    if m == 1:                        
                        toolbox.register("mutate", tools.mutPolynomialBounded, low=LOW, up=UP, eta=ETA, indpb=1.0/LENGTH_CHROM)
                    
                    
                    stats = tools.Statistics(lambda ind: ind.fitness.values)
                    stats.register("min", np.min)
                    stats.register("avg", np.mean)

                    x = np.arange(-10, 10, 0.1)
                    y = np.arange(-10, 10, 0.1)
                    xgrid, ygrid = np.meshgrid(x, y)

                    f_himmelbalu = (1-np.sin((xgrid**2+ygrid**2)**0.5)**2)/(1+0.001*(xgrid**2+ygrid**2)) + 100


                    algelitism.eaSimpleElitism
                    algorithms.eaSimple
                    if i == 0:
                        try: population_1, logbook_1 = algelitism.eaSimpleElitism(population, toolbox,
                                                                        cxpb=P_CROSSOVER,
                                                                        mutpb=P_MUTATION_MAX if m==0 else P_MUTATION_MIN,
                                                                        ngen=MAX_GENERATIONS,
                                                                        halloffame=hof,
                                                                        stats=stats,
                                                                        #callback=(show, (ax, xgrid, ygrid, f_himmelbalu)),
                                                                        verbose=True)
                        except:
                            continue
                    else:
                        try: population_1, logbook_1 = algelitism.eaSimpleElitism(population, toolbox,
                                                            cxpb=P_CROSSOVER,
                                                            mutpb=P_MUTATION_MAX if m==0 else P_MUTATION_MIN,
                                                            ngen=MAX_GENERATIONS,
                                                            halloffame=hof,
                                                            stats=stats,
                                                            #callback=(showBin, (ax, xgrid, ygrid, f_himmelbalu)),
                                                            verbose=True)
                        except:
                            continue

                    maxFitnessValues, meanFitnessValues, genValues = logbook_1.select("min", "avg", "gen")

                    best = hof.items[0]
                    result_str = []
                    if j == 0:
                        result_str.append('Случайная')
                    if j == 1:
                        result_str.append('Рулетка')
                    if j == 2:
                        result_str.append('Турнир')
                    if k == 0:
                        result_str.append('Одноточечная')
                    if k == 1:
                        result_str.append('Двуточечная')
                    if k == 2:
                        result_str.append('cxSimulatedBinaryBounded ')
                    if m == 0:
                        result_str.append('Большая вероятность')
                    if m == 1:
                        result_str.append('Малая вероятность')
                    if i == 0:
                        result_str.append('Число')
                        result_str.append(best[0])
                        result_str.append(best[1])
                        result_str.append(func(best))
                    if i == 1:
                        result_str.append('Бин')
                        result_str.append(format(int(best[0]), '04b'))
                        result_str.append(format(int(best[1]), '04b'))
                        # result_str.append(bin(int(best[0]))[2:])
                        # result_str.append(bin(int(best[1]))[2:])
                        result_str.append(funcBin([best[0], best[1]]))
                    if i == 2:
                        result_str.append('Код грея')
                        result_str.append('{:03b}'.format(graycode.tc_to_gray_code(int(best[0]))))
                        result_str.append('{:03b}'.format(graycode.tc_to_gray_code(int(best[1]))))
                        result_str.append(funcBin([best[0], best[1]]))
                    coef = metrics(maxFitnessValues)
                    result_str.append(coef[0])
                    result_str.append(coef[1])
                    
                    table.add_row(result_str)
    print(table)
    with open('res.txt', 'w') as file:
        file.write(str(table))

    
    
    
    
    
    
    
    
    
    # toolbox = base.Toolbox()
    # toolbox.register("randomPoint", randomPoint, LOW, UP)
    # toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomPoint)
    # toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

    # population = toolbox.populationCreator(n=POPULATION_SIZE)
 
    # toolbox.register("evaluate", func)
    # toolbox.register("select", tools.selRandom)
    # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=LOW, up=UP, eta=ETA)
    # toolbox.register("mutate", tools.mutPolynomialBounded, low=LOW, up=UP, eta=ETA, indpb=1.0/LENGTH_CHROM)

    # stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("min", np.min)
    # stats.register("avg", np.mean)

    # x = np.arange(-10, 10, 0.1)
    # y = np.arange(-10, 10, 0.1)
    # xgrid, ygrid = np.meshgrid(x, y)

    # f_himmelbalu = (1-np.sin((xgrid**2+ygrid**2)**0.5)**2)/(1+0.001*(xgrid**2+ygrid**2)) + 100

    # plt.ion()
    # fig, ax = plt.subplots()
    # fig.set_size_inches(5, 5)

    # ax.set_xlim(LOW, UP + 3)
    # ax.set_ylim(LOW, UP + 3)

    # algelitism.eaSimpleElitism
    # algorithms.eaSimple
    # population_1, logbook_1 = algelitism.eaSimpleElitism(population, toolbox,
    #                                                   cxpb=P_CROSSOVER,
    #                                                   mutpb=P_MUTATION_MAX,
    #                                                   ngen=MAX_GENERATIONS,
    #                                                   halloffame=hof,
    #                                                   stats=stats,
    #                                                   callback=(show, (ax, xgrid, ygrid, f_himmelbalu)),
    #                                                   verbose=True)

    # maxFitnessValues, meanFitnessValues, genValues = logbook_1.select("min", "avg", "gen")

    # best_num = hof.items[0]
    # print(best_num)

    # plt.ioff()
    # plt.show()

    # plt.plot(maxFitnessValues, color='blue')
    # plt.plot(meanFitnessValues, color='green')
    # plt.xlabel('Поколение')
    # plt.ylabel('Макс/средняя приспособленность')
    # plt.title('minF(x) и avgF(x)')
    # plt.show()
    
    # input()
    
    # ####Бинарная
    # toolbox = base.Toolbox()
    # toolbox.register("randomPoint", randomPoint_bin, LOW, UP)
    # toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomPoint)
    # toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

    # population = toolbox.populationCreator(n=POPULATION_SIZE)
 
    # toolbox.register("evaluate", funcBin)
    # toolbox.register("select", tools.selRandom)
    # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=LOW, up=UP, eta=ETA)
    # toolbox.register("mutate", tools.mutPolynomialBounded, low=LOW, up=UP, eta=ETA, indpb=1.0/LENGTH_CHROM)

    # stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("min", np.min)
    # stats.register("avg", np.mean)

    # x = np.arange(-10, 10, 0.1)
    # y = np.arange(-10, 10, 0.1)
    # xgrid, ygrid = np.meshgrid(x, y)

    # f_himmelbalu = (1-np.sin((xgrid**2+ygrid**2)**0.5)**2)/(1+0.001*(xgrid**2+ygrid**2)) + 100

    # plt.ion()
    # fig, ax = plt.subplots()
    # fig.set_size_inches(5, 5)

    # ax.set_xlim(LOW, UP + 3)
    # ax.set_ylim(LOW, UP + 3)

    # algelitism.eaSimpleElitism
    # algorithms.eaSimple
    # population_1, logbook_1 = algelitism.eaSimpleElitism(population, toolbox,
    #                                                   cxpb=P_CROSSOVER,
    #                                                   mutpb=P_MUTATION_MAX,
    #                                                   ngen=MAX_GENERATIONS,
    #                                                   halloffame=hof,
    #                                                   stats=stats,
    #                                                   callback=(showBin, (ax, xgrid, ygrid, f_himmelbalu)),
    #                                                   verbose=True)

    # maxFitnessValues, meanFitnessValues, genValues = logbook_1.select("min", "avg", "gen")

    # best_bin = hof.items[0]
    # print([int(best_bin[0]),int(best_bin[1])])

    # plt.ioff()
    # plt.show()

    # plt.plot(maxFitnessValues, color='blue')
    # plt.plot(meanFitnessValues, color='green')
    # plt.xlabel('Поколение')
    # plt.ylabel('Макс/средняя приспособленность')
    # plt.title('minF(x) и avgF(x)')
    # plt.show()
    
    # input()
    
    # ####Код грея
    # toolbox = base.Toolbox()
    # toolbox.register("randomPoint", randomPoint_bin, LOW, UP)
    # toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomPoint)
    # toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

    # population = toolbox.populationCreator(n=POPULATION_SIZE)
 
    # toolbox.register("evaluate", funcBin)
    # toolbox.register("select", tools.selRandom)
    # toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=LOW, up=UP, eta=ETA)
    # toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/LENGTH_CHROM)

    # stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("min", np.min)
    # stats.register("avg", np.mean)

    # x = np.arange(-10, 10, 0.1)
    # y = np.arange(-10, 10, 0.1)
    # xgrid, ygrid = np.meshgrid(x, y)

    # f_himmelbalu = (1-np.sin((xgrid**2+ygrid**2)**0.5)**2)/(1+0.001*(xgrid**2+ygrid**2)) + 100

    # plt.ion()
    # fig, ax = plt.subplots()
    # fig.set_size_inches(5, 5)

    # ax.set_xlim(LOW, UP + 3)
    # ax.set_ylim(LOW, UP + 3)

    # # algelitism.eaSimpleElitism
    # # algorithms.eaSimple
    # population_1, logbook_1 = algelitism.eaSimpleElitism(population, toolbox,
    #                                                  cxpb=P_CROSSOVER,
    #                                                  mutpb=P_MUTATION_MIN,
    #                                                  ngen=MAX_GENERATIONS,
    #                                                  halloffame=hof,
    #                                                  stats=stats,
    #                                                  callback=(showBin, (ax, xgrid, ygrid, f_himmelbalu)),
    #                                                  verbose=True)

    # maxFitnessValues, meanFitnessValues, genValues = logbook_1.select("min", "avg", "gen")

    # best_grey = hof.items[0]
    
    # print([int(best_grey[0]),int(best_grey[1])])
    # print(best_grey[0], '{:03b}'.format(graycode.tc_to_gray_code(best_grey[0])), '|', best_grey[1], '{:03b}'.format(graycode.tc_to_gray_code(best_grey[1])))
    
    
    # plt.ioff()
    # plt.show()

    # plt.plot(maxFitnessValues, color='blue')
    # plt.plot(meanFitnessValues, color='green')
    # plt.xlabel('Поколение')
    # plt.ylabel('Макс/средняя приспособленность')
    # plt.title('minF(x) и avgF(x)')
    # plt.show()
    
