import random
import json
import numpy as np
from matplotlib import pyplot as plt

class CSFaker:
    citizenships = ['ar_EG', 'ar_PS', 'ar_SA', 'bg_BG', 'cs_CZ', 'de_DE', 'dk_DK', 'el_GR', 'en_AU', 'en_CA', 'en_GB', 'en_US', 'es_ES', 'es_MX', 'et_EE', 'fa_IR', 'fi_FI', 'fr_FR', 'hi_IN', 'hr_HR', 'hu_HU', 'it_IT', 'ja_JP', 'ko_KR', 'lt_LT', 'lv_LV', 'ne_NP', 'nl_NL', 'no_NO', 'pl_PL', 'pt_BR', 'pt_PT', 'ro_RO', 'ru_RU', 'sl_SI', 'sv_SE', 'tr_TR', 'uk_UA', 'zh_CN', 'zh_TW', 'ka_GE']

    def generator(self):
        #Name
        name = 'a' #self.person()['name']
        #Citizenship
        citizenship = random.choice(self.citizenships)
        #Age
        age = random.randint(18, 100)
        #Education
        educ = ['High School', 'college', 'masters', 'phd', 'genius']
        education = random.choice(educ)
        #Race
        race = ['almond', 'oreo', 'quaker']
        race = random.choice(race)
        #Gender
        gender = random.randint(0, 1)
        #CreditScore
        base = 100

        if(education == 'High School'):
            base *= 0.5
        elif (education == 'college'):
            base *= 0.75
        elif(education == 'masters'):
            base *= 1
        elif(education == 'phd'):
            base *= 1.25
        else:
            base *= 1.5
        
        if(race == 'almond'):
            base*= 0.5
        elif (race == 'quaker'):
            base *= 1.5
        else:
            pass
        
        if(gender == 1):
            base *= 1.25
        else:
            base*= 0.75

        base += 2*age
        credit_score = base

        return {"name": name, "citizenship": citizenship, 'age': age, 'education': education, 'race': race, 'gender': gender, 'credit_score': credit_score}

        def generate():
            dataset = []

            for i in range(100000):
                print(i)
                c = CSFaker()
                dataset.append(c.generator())

            credit_scores = np.asarray([iter['credit_score'] for iter in dataset])
            max_score = np.amax(credit_scores)
            min_score = np.amin(credit_scores)
            mean_score = np.mean(credit_scores)
            median_score = np.median(credit_scores)
            std_deviation = np.std(credit_scores)
            stats = {'mean': mean_score, 'max': max_score, 'min': min_score, 'median': median_score,
                     'stdev': std_deviation, 'quartile1': (mean_score - std_deviation),
                     'quartile3': (mean_score + std_deviation)}
            dataset.append(stats)

            with open('CreditScoreData.json', 'w') as outfile:
                json.dump(dataset, outfile)

            print(dataset)

            print(dataset[-1])

            plt.hist(credit_scores, bins='auto')  # arguments are passed to np.histogram
            plt.title("Simulated Credit score on simulated data")
            plt.xlabel("credit score")
            plt.ylabel("frequency")
            plt.show()