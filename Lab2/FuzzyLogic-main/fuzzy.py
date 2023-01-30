"""
authors: Konrad Chrzanowski, Łukasz Reinke
emails: s17404@pjwstk.edu.pl , s15037@pjwstk.edu.pl
task: Car Wash

Zadaniem tego programu jest wyliczenie ceny za myjnię samochodową.
Mamy 3 wejścia
    Wielkość samochodu (car_size) : od 0 - 100, gdzie 100 Van a 0 Hatchback. 
    Ilość odwiedzeń w miesiącu (visit_count) : od 0 - 10, gdzie 10 to jest 10 odwiedzeń
    Wielkość zabrudzenia (dirt_level) : od 0 - 10, gdzie 0 to lekko brudny

I 1 wyjście
    Cena za usługe (price) : od 0 do 100, gdzie 100 to 50 zł

Przypadek testowy to duży samochód, klient z dużą ilością odwiedzeń i mała wielkość zabrudzenia
    car_size = 90
    visit_count = 10
    dirt_level = 1

Żeby uruchomić program trzeba zainstalować
pip install scikit-fuzzy
pip install matplotlib
pip install numpy


"""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Create membership functions
car_size = ctrl.Antecedent(np.arange(0, 101, 10), 'car_size') 
visit_count = ctrl.Antecedent(np.arange(0, 11, 1), 'visit_count') 
dirt_level = ctrl.Antecedent(np.arange(0, 11, 1), 'dirt_level')  
#Create output
price = ctrl.Consequent(np.arange(0, 101, 1), 'price')  

# Populate the universe
car_size.automf(3)
visit_count.automf(3)
dirt_level.automf(3)

# Triangular function generator, [start, maximum, end]
price['low'] = fuzz.trimf(price.universe, [0, 0, 40])
price['medium'] = fuzz.trimf(price.universe, [0, 30, 60])
price['high'] = fuzz.trimf(price.universe, [50, 100, 100])
# Visual representation of the Rule, highlighted sections that we are going to use
car_size['good'].view()
visit_count['good'].view()
dirt_level['poor'].view()
price.view()
# Create a rule based on inputs

rule1 = ctrl.Rule((car_size['good'] & visit_count['average'] ) | 
                  (dirt_level['good'] & car_size['average']) , price['high'])

rule2 = ctrl.Rule((car_size['good'] & visit_count['good'] & dirt_level['poor']) |
                 (car_size['average'] & dirt_level['average']), price['medium'])

rule3 = ctrl.Rule((car_size['poor'] & visit_count['good']) | 
                (car_size['average'] & dirt_level['poor'] & visit_count['average'])
                , price['low'])

price_ctrl = ctrl.ControlSystem([rule1, rule2])
# Create simulation
price_sim = ctrl.ControlSystemSimulation(price_ctrl)
# Set input values in simulation
price_sim.input['car_size'] = 90
price_sim.input['visit_count'] = 10
price_sim.input['dirt_level'] = 1
# Compute the numbers
price_sim.compute()
# print results
print(price_sim.output['price'])
price.view(sim=price_sim)
print(price_sim.output['price'])
plt.show()